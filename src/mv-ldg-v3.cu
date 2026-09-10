/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*
 * LDG3 SpMV: LDG's eight-way 2D launch geometry, LDG_V2's two-stage register
 * pipeline and W rows/thread, over a selectable cache policy and access width.
 *
 * Within one call the matrix is read exactly once, so streaming it with __ldcs
 * keeps it from evicting x, which is reused through the gather. Across calls
 * that reasoning only holds while the matrix is too large to have survived in
 * cache anyway. At the coarse MG levels it is not -- level 3 is some 46 MiB --
 * and there __ldcs discards reuse a plain cached load would have kept. Both
 * policies are therefore built and the autotuner picks per level.
 *
 * The vector paths reinterpret the column array as int2, so this family is
 * available only when local_int_t is 32-bit; INDEX_64 is not supported.
 */

#ifdef USE_CUDA

#include "SparseMatrix.hpp"
#include "intdiv.hh"
#include "ldg-loads.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#ifndef INDEX_64

namespace
{

// WIDE is the family's second variant rather than a replacement: with it off,
// the loads are exactly what measured 2262.5 GFLOP/s, and with it on they are as
// wide as the hardware allows. Which is better is not obvious and is left to the
// autotuner, because width is not free -- it is paid for in the registers that
// buy resident warps, and this kernel is bandwidth-bound.
//
// Only the loads change. The store is identical in both variants, so a
// difference between them is attributable to load width alone.
//
// CS is the cache policy, and it is the axis on which this family differs from
// LDG_V2 rather than from LDG. LDG already loads the matrix with __ldcs; LDG_V2
// loads it through plain const __restrict__ dereferences, which become
// ld.global.nc and keep the line. Fixing LDG3 to __ldcs therefore made it a
// strict subset of neither parent, and left it unable to express the policy
// that wins wherever the matrix is small enough to stay resident. With CS a
// parameter, the family spans both parents on this axis and the autotuner
// decides per level.
//
// The wide (WIDE && W%4==0) path below is exactly ldgload's LoadCols/LoadVals,
// used instead of a second hand-written copy of the same PTX: LDG_V2 needed
// that PTX for the same reason (__ldcs bottoms out at 2-wide for doubles), and
// keeping one copy means LDG3 gets ldgload's ".nc" qualifier on every streaming
// access and its true single-instruction 256-bit column load at W=8, neither of
// which this file's own hand-written version had.
//
// The narrow path (the "else" below, forced 2-wide access even when W is 4 or
// 8) stays local: it is LDG3's own experiment against the wide path above, and
// ldgload has no equivalent since LoadCols/LoadVals there always pick the
// widest legal access for a given W.
template <bool CS>
__device__ __forceinline__ int2 Load2i(const local_int_t* __restrict__ p)
{
    const int2* q = reinterpret_cast<const int2*>(p);
    if constexpr (CS)
        return __ldcs(q);
    else
        return *q;
}

template <bool CS>
__device__ __forceinline__ double2 Load2d(const double* __restrict__ p)
{
    const double2* q = reinterpret_cast<const double2*>(p);
    if constexpr (CS)
        return __ldcs(q);
    else
        return *q;
}

// Alignment comes from the launcher's guards rather than luck. in_slice is a
// multiple of W and the k stride is slice_size, which the launcher requires to
// be a multiple of W; with W a multiple of 4 that makes every offset below a
// multiple of 4 elements. The launcher additionally checks the base pointers,
// since a wide access to a misaligned address faults instead of degrading.
template <int W, bool WIDE, bool CS>
__device__ __forceinline__ void LoadColsV3(const local_int_t* __restrict__ p, int (&c)[W])
{
    if constexpr (W == 1 || W == 2 || (WIDE && W % 4 == 0))
    {
        ldgload::LoadCols<W, CS>(p, c);
    }
    else
    {
#pragma unroll
        for (int w = 0; w < W; w += 2)
        {
            const int2 t = Load2i<CS>(p + w);
            c[w] = t.x;
            c[w + 1] = t.y;
        }
    }
}

template <int W, bool WIDE, bool CS>
__device__ __forceinline__ void LoadValsV3(const double* __restrict__ p, double (&v)[W])
{
    if constexpr (W == 1 || W == 2 || (WIDE && W % 4 == 0))
    {
        ldgload::LoadVals<W, CS>(p, v);
    }
    else
    {
#pragma unroll
        for (int w = 0; w < W; w += 2)
        {
            const double2 t = Load2d<CS>(p + w);
            v[w] = t.x;
            v[w + 1] = t.y;
        }
    }
}

// Clamping k here is what lets the caller's loop overshoot the end of the row
// without a branch. A clamped slot gets column -1, which ConsumeBlock turns into
// a zero contribution without touching x.
template <int UNROLL, int W, bool WIDE, bool CS>
__device__ __forceinline__ void LoadBlockV3(int kb, int max_row_len, int slice_size,
    const local_int_t* __restrict__ cp, const double* __restrict__ vp, int (&cols)[UNROLL][W],
    double (&vals)[UNROLL][W])
{
#pragma unroll
    for (int ki = 0; ki < UNROLL; ++ki)
    {
        const int k = kb + ki;
        if (k < max_row_len)
        {
            const size_t off = (size_t) k * slice_size;
            LoadColsV3<W, WIDE, CS>(&cp[off], cols[ki]);
            LoadValsV3<W, WIDE, CS>(&vp[off], vals[ki]);
        }
        else
        {
#pragma unroll
            for (int w = 0; w < W; ++w)
            {
                cols[ki][w] = -1;
                vals[ki][w] = 0.0;
            }
        }
    }
}

// The column test covers both the overshoot above and the SELL builder's own
// padding: rows shorter than their slice's max are padded with column -1 by
// createSellLUColumnsValues_kernel, and those slots keep the -1.0 that
// setLUValues_kernel wrote, not a zero. Selecting 0.0 rather than multiplying is
// what makes them contribute nothing, and it also means a padded lane never
// loads x at all.
template <int UNROLL, int W>
__device__ __forceinline__ void ConsumeBlock(
    double (&sum)[W], const int (&cols)[UNROLL][W], const double (&vals)[UNROLL][W], const double* __restrict__ x)
{
#pragma unroll
    for (int ki = 0; ki < UNROLL; ++ki)
    {
        double b[W];
#pragma unroll
        for (int w = 0; w < W; ++w)
        {
            const int col = cols[ki][w];
            b[w] = (col >= 0) ? x[col] : 0.0;
        }
#pragma unroll
        for (int w = 0; w < W; ++w)
            sum[w] += vals[ki][w] * b[w];
    }
}

// The partition count is gridDim.x and the rows per partition arrive as an
// argument, so this kernel is agnostic to how many partitions there are. That
// keeps the split a runtime knob costing one more argument, rather than a
// template parameter that would multiply the 272 instantiations below.
template <int BLKDIM, int UNROLL, int W, bool WIDE, bool CS>
__global__ __launch_bounds__(BLKDIM) void mv_sell_ldgv3(int partition_rows, double alpha, double beta,
    double* __restrict__ y, const double* __restrict__ x, const slice_ptr_t* __restrict__ slice_offsets,
    const local_int_t* __restrict__ col_idx, const double* __restrict__ values, int slice_size,
    intdiv32_t slice_size_div)
{
    // LDG's geometry: PART row partitions along x, rows within a partition along
    // y. Because blockIdx.x moves fastest, the blocks that get scheduled together
    // land in PART well-separated row ranges, which spreads the column stream
    // across the memory system. PART=1 collapses this to LDG_V2's plain 1D walk.
    const int row_in_partition = (blockIdx.y * BLKDIM + threadIdx.x) * W;
    const int base_row = blockIdx.x * partition_rows + row_in_partition;
    const int partition_end = (blockIdx.x + 1) * partition_rows;
    if (base_row >= partition_end)
        return;

    int slice, in_slice;
    intdiv32_divmod(base_row, slice_size, slice_size_div, &slice, &in_slice);
    const size_t row_start = (size_t) slice_offsets[slice] + (size_t) in_slice;
    int max_row_len;
    intdiv32_div((int32_t) (slice_offsets[slice + 1] - slice_offsets[slice]), slice_size_div, &max_row_len);

    const local_int_t* cp = col_idx + row_start;
    const double* vp = values + row_start;

    int cA[UNROLL][W], cB[UNROLL][W];
    double vA[UNROLL][W], vB[UNROLL][W];

    LoadBlockV3<UNROLL, W, WIDE, CS>(0, max_row_len, slice_size, cp, vp, cA, vA);

    double sum[W];
#pragma unroll
    for (int w = 0; w < W; ++w)
        sum[w] = 0.0;

    // LDG_V2's two-stage A/B pipeline: no branch in the loop body and no scalar
    // epilogue, because LoadBlockV3 makes running past the row end harmless.
    for (int kb = 0; kb < max_row_len; kb += 2 * UNROLL)
    {
        LoadBlockV3<UNROLL, W, WIDE, CS>(kb + UNROLL, max_row_len, slice_size, cp, vp, cB, vB);
        ConsumeBlock<UNROLL, W>(sum, cA, vA, x);
        LoadBlockV3<UNROLL, W, WIDE, CS>(kb + 2 * UNROLL, max_row_len, slice_size, cp, vp, cA, vA);
        ConsumeBlock<UNROLL, W>(sum, cB, vB, x);
    }

#pragma unroll
    for (int w = 0; w < W; ++w)
    {
        const int r = base_row + w;
        if (r < partition_end)
            y[r] = (beta == 0.0) ? (alpha * sum[w]) : (beta * y[r] + alpha * sum[w]);
    }
}

template <int BLKDIM, int UNROLL, int W, bool WIDE, bool CS>
bool LaunchMvLdgV3(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const slice_ptr_t* slice_offsets, const local_int_t* columns, const double* values, cudaStream_t stream, int part)
{
    const local_int_t m = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;
    if (part < 1 || part > 65535)
        return false;
    // An uneven split would leave the last partition short, and the kernel's
    // bound is per-partition rather than global, so the remainder rows would
    // never be written. Refusing here makes the autotuner skip the pairing.
    if (m % part != 0)
        return false;
    const local_int_t partition_rows = m / part;
    if (slice_size % W != 0 || partition_rows % W != 0)
        return false;

    // A wide access to a misaligned address faults rather than degrading. A
    // cudaMalloc base is far more aligned than the 32 bytes needed, but a vector
    // that is a view into a larger allocation need not be, so the base pointers
    // are checked instead of assumed. Returning false makes the autotuner skip
    // this configuration rather than the run die.
    if (WIDE && W % 4 == 0)
    {
        if (((uintptr_t) values % 32u) != 0 || ((uintptr_t) columns % 16u) != 0)
            return false;
    }

    const intdiv32_t sdiv = intdiv32_gen(slice_size);
    const local_int_t nthreads = partition_rows / W;
    const unsigned int grid_y = (unsigned int) ((nthreads + BLKDIM - 1) / BLKDIM);
    if (grid_y > 65535u)
        return false;
    mv_sell_ldgv3<BLKDIM, UNROLL, W, WIDE, CS><<<dim3((unsigned int) part, grid_y, 1), BLKDIM, 0, stream>>>(
        (int) partition_rows, alpha, beta, y, x, slice_offsets, columns, values, slice_size, sdiv);
    return true;
}

}

// wide selects the access width and cached selects the cache policy. Only W of 4
// and 8 have a wide form -- one and two rows per thread are 8 and 16 bytes,
// already at or below the narrow width -- so asking for wide at W of 1 or 2 is
// refused rather than silently served by the narrow kernel, which would make the
// two variants look identical for reasons that have nothing to do with the
// hardware. Cached applies at every W, since the policy question is independent
// of how many bytes each access moves.
bool MvLdgV3SellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    slice_ptr_t* sell_block_offset, local_int_t* sell_columns, double* sell_values, cudaStream_t stream, int blk,
    int unroll, int w, bool wide, bool cached, int part)
{
#define TRY(B, U, WI, WD, CA)                                                                                          \
    if (blk == (B) && unroll == (U) && w == (WI) && wide == (WD) && cached == (CA))                                    \
        return LaunchMvLdgV3<B, U, WI, WD, !(CA)>(                                                                     \
            A, alpha, beta, x, y, sell_block_offset, sell_columns, sell_values, stream, part);
#define POL(B, U, WI, WD) TRY(B, U, WI, WD, false) TRY(B, U, WI, WD, true)
#define ROW(B, U)                                                                                                      \
    POL(B, U, 1, false)                                                                                                \
    POL(B, U, 2, false)                                                                                                \
    POL(B, U, 4, false)                                                                                                \
    POL(B, U, 8, false)                                                                                                \
    POL(B, U, 4, true)                                                                                                 \
    POL(B, U, 8, true)
#define BLK_ROWS(B) ROW(B, 1) ROW(B, 2) ROW(B, 3) ROW(B, 4)

// Depth, offered only at small W. LDG takes level 2 at 128/14 and 256/10 with
// one row per thread, and until now LDG3 could not express anything like it.
// The two families do not mean the same thing by unroll: LDG advances Unroll
// k-steps per loop iteration, where LDG3's A/B pair advances 2*UNROLL, so LDG's
// 14 and 10 correspond to UNROLL 7 and 5 here. The range below brackets both.
//
// It stops where the register file does. The A/B pair costs 2*UNROLL*W*12 bytes
// per thread, so depth is affordable exactly where W is not: at W=1, UNROLL=14
// is about 84 registers, and at W=2, UNROLL=8 is about 96. Going deeper at W of
// 4 or 8 would only trade resident warps for a longer body in a kernel that is
// bandwidth-bound, which is the trade the original cap was right to refuse.
//
// Wide is absent because it needs W of at least 4. Both cache policies are kept,
// since depth and policy answer different questions.
#define DEEP(B, U, WI) POL(B, U, WI, false)
#define BLK_DEEP(B)                                                                                                    \
    DEEP(B, 5, 1) DEEP(B, 6, 1) DEEP(B, 7, 1) DEEP(B, 8, 1) DEEP(B, 10, 1) DEEP(B, 14, 1)                              \
    DEEP(B, 5, 2) DEEP(B, 6, 2) DEEP(B, 7, 2) DEEP(B, 8, 2)
    BLK_ROWS(32)
    BLK_ROWS(64)
    BLK_ROWS(128)
    BLK_ROWS(256)
    BLK_DEEP(32)
    BLK_DEEP(64)
    BLK_DEEP(128)
    BLK_DEEP(256)
#undef BLK_DEEP
#undef DEEP
#undef BLK_ROWS
#undef ROW
#undef POL
#undef TRY
    return false;
}

#else // INDEX_64

bool MvLdgV3SellCfg(const SparseMatrix&, double, double, const double*, double*, slice_ptr_t*, local_int_t*, double*,
    cudaStream_t, int, int, int, bool, bool, int)
{
    return false;
}

#endif // !INDEX_64
#endif
