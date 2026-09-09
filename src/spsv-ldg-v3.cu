//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 @file spsv-ldg-v3.cu

 LDG3 SymGS triangular solve: LDG's per-color launch, LDG_V2's two-stage
 register pipeline and W rows/thread, over a selectable cache policy and
 access width.

 Within one colour launch the matrix is read exactly once, so streaming it
 with __ldcs keeps it from evicting x, which is reused through the gather.
 Across the colour launches of a sweep, and across sweeps, that only holds
 while the matrix is too large to have survived in cache anyway. At the coarse
 MG levels it is not, and there __ldcs discards reuse a plain cached load would
 have kept. Both policies are built and the autotuner picks per level.

 The vector paths reinterpret the column array as int2, so this family takes
 32-bit columns (idx32_t), which is what the index-mode dispatch already
 enforces. Instantiated for both slice-offset widths, like the rest of the
 explicit path.
 */

#ifdef USE_CUDA
#ifdef EXPLICIT_KERNELS

#include "CudaKernels.hpp"
#include "IndexMode.hpp"
#include "SparseMatrix.hpp"
#include "intdiv.hh"
#include "ldg-loads.cuh"

#include <cstdint>
#include <cuda_runtime.h>

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
__device__ __forceinline__ int2 Load2i(const idx32_t* __restrict__ p)
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
__device__ __forceinline__ void LoadColsV3(const idx32_t* __restrict__ p, int (&c)[W])
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
__device__ __forceinline__ void LoadBlockV3(int kb, int max_row_len, local_int_t slice_size,
    const idx32_t* __restrict__ cp, const double* __restrict__ vp, int (&cols)[UNROLL][W], double (&vals)[UNROLL][W])
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
    double (&sum)[W], const int (&cols)[UNROLL][W], const double (&vals)[UNROLL][W], const double* x)
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

template <class OffsetT, int BLKDIM, int UNROLL, int W, bool WIDE, bool CS>
__global__ __launch_bounds__(BLKDIM) void spsv_sell_ldgv3(local_int_t color_str, local_int_t color_end,
    double* __restrict__ x, const double* __restrict__ rhs, const OffsetT* __restrict__ slice_offsets,
    const idx32_t* __restrict__ col_idx, const double* __restrict__ values, const double* __restrict__ diag,
    double alpha, local_int_t slice_size, intdiv32_t slice_size_div)
{
    const local_int_t base_row = ((local_int_t) blockIdx.x * BLKDIM + (local_int_t) threadIdx.x) * W + color_str;
    if (base_row >= color_end)
        return;

    int slice, in_slice;
    intdiv32_divmod((int32_t) base_row, (int32_t) slice_size, slice_size_div, &slice, &in_slice);
    // 64-bit only when the slice offsets are, since a flat offset is bounded by
    // the same padded nonzero count those offsets hold -- see FlatOffsetT.
    const FlatOffsetT<OffsetT> row_start
        = (FlatOffsetT<OffsetT>) slice_offsets[slice] + (FlatOffsetT<OffsetT>) in_slice;
    // Per-slice nnz is bounded by slice_size * HPCG_MAX_ROW_LEN and so fits in
    // int32. Narrowing the difference before dividing keeps this in the 32-bit
    // reciprocal above instead of the 64-bit form a wider OffsetT would force.
    int max_row_len;
    intdiv32_div((int32_t) (slice_offsets[slice + 1] - slice_offsets[slice]), slice_size_div, &max_row_len);

    const idx32_t* cp = col_idx + row_start;
    const double* vp = values + row_start;

    int cA[UNROLL][W], cB[UNROLL][W];
    double vA[UNROLL][W], vB[UNROLL][W];

    LoadBlockV3<UNROLL, W, WIDE, CS>(0, max_row_len, slice_size, cp, vp, cA, vA);

    // rhs and diag read here rather than at their point of use in the epilogue.
    // Down there nothing is left in the thread to overlap them with: every warp
    // arrives at the tail together, both loads miss, and the division cannot
    // start until diag lands. Read before the loop, that latency is spent
    // against the k iterations instead, which is what spsv-tma.cu does.
    //
    // The cost is 4*W registers held across the loop -- two doubles a row --
    // competing with the same budget the A/B prefetch draws on. It is charged
    // whether or not the row is live, so the guard picks the value rather than
    // the load: 1.0 for a dead lane's diag, since it is still divided by.
    //
    // The diagonal follows the same policy as the matrix it belongs to. Each
    // colour touches a disjoint slice of it, so under CS it is read once and
    // dropped; under the cached policy the whole vector is small enough to be
    // worth keeping across the colour launches of one sweep. LoadScalarRo is
    // ldgload's read-only-once-per-call scalar load -- the same helper LDG_V2
    // uses for rhs/diag in spsv-ldg-v2.cu, for the identical reason.
    double rhs_m[W], diag_m[W];
#pragma unroll
    for (int w = 0; w < W; ++w)
    {
        const local_int_t r = base_row + w;
        const bool live = r < color_end;
        rhs_m[w] = live ? rhs[r] : 0.0;
        diag_m[w] = live ? ldgload::LoadScalarRo<CS>(&diag[r]) : 1.0;
    }

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
        const local_int_t r = base_row + w;
        if (r < color_end)
            x[r] = (alpha * rhs_m[w] - sum[w]) / diag_m[w];
    }
}

template <class OffsetT, int BLKDIM, int UNROLL, int W, bool WIDE, bool CS>
bool LaunchSpsvLdgV3(bool forward, const SparseMatrix& A, const double* rv, double* xv, const OffsetT* slice_offsets,
    const idx32_t* columns, const double* values, cudaStream_t stream)
{
    const local_int_t rows = A.localNumberOfRows;
    const local_int_t slice_size = A.slice_size;
    if (A.totalColors <= 0 || rows % A.totalColors != 0)
        return false;
    const local_int_t color_size = rows / A.totalColors;

    if (slice_size % W != 0 || color_size % W != 0)
        return false;

    // See the MV launcher: a wide access faults on a misaligned address, a
    // vector that is a view into a larger allocation need not be aligned even
    // though a cudaMalloc base is, and the column requirement scales with W
    // because the access does.
    if (WIDE && W % 4 == 0)
    {
        constexpr unsigned int kColAlign = (unsigned int) (W * sizeof(idx32_t));
        if (((uintptr_t) values % 32u) != 0 || ((uintptr_t) columns % kColAlign) != 0)
            return false;
    }

    const intdiv32_t sdiv = intdiv32_gen((int32_t) slice_size);
    const double alpha = 1.0;
    const local_int_t nthreads = color_size / W;
    const local_int_t grid = (nthreads + BLKDIM - 1) / BLKDIM;

    auto kernel = spsv_sell_ldgv3<OffsetT, BLKDIM, UNROLL, W, WIDE, CS>;
    if (forward)
    {
        for (int color = 0; color < A.totalColors; ++color)
        {
            const local_int_t cs = color * color_size;
            const local_int_t ce = cs + color_size;
            kernel<<<grid, BLKDIM, 0, stream>>>(
                cs, ce, xv, rv, slice_offsets, columns, values, A.diagonal, alpha, slice_size, sdiv);
        }
    }
    else
    {
        for (int color = A.totalColors - 1; color >= 0; --color)
        {
            const local_int_t cs = color * color_size;
            const local_int_t ce = cs + color_size;
            kernel<<<grid, BLKDIM, 0, stream>>>(
                cs, ce, xv, rv, slice_offsets, columns, values, A.diagonal, alpha, slice_size, sdiv);
        }
    }
    return true;
}

} // namespace

// wide selects the access width and cached selects the cache policy, exactly as
// in the SpMV of this family. There is no partition knob here: the colour loop
// already fixes the grid, so the solve has nothing corresponding to the SpMV's
// row partitions.
template <class OffsetT>
bool SpsvLdgV3SellCfg(bool forward, const SparseMatrix& A, const double* rv, double* xv, const OffsetT* slice_offsets,
    const idx32_t* columns, const double* values, cudaStream_t stream, int blk, int unroll, int w, bool wide,
    bool cached)
{
#define HPCG_SV_LDGV3_CASE(B, U, WI, WD, CA)                                                                           \
    if (blk == (B) && unroll == (U) && w == (WI) && wide == (WD) && cached == (CA))                                    \
        return LaunchSpsvLdgV3<OffsetT, B, U, WI, WD, !(CA)>(                                                          \
            forward, A, rv, xv, slice_offsets, columns, values, stream);
#define HPCG_SV_LDGV3_POL(B, U, WI, WD) HPCG_SV_LDGV3_CASE(B, U, WI, WD, false) HPCG_SV_LDGV3_CASE(B, U, WI, WD, true)
#define HPCG_SV_LDGV3_ROW(B, U)                                                                                        \
    HPCG_SV_LDGV3_POL(B, U, 1, false)                                                                                  \
    HPCG_SV_LDGV3_POL(B, U, 2, false)                                                                                  \
    HPCG_SV_LDGV3_POL(B, U, 4, false)                                                                                  \
    HPCG_SV_LDGV3_POL(B, U, 8, false)                                                                                  \
    HPCG_SV_LDGV3_POL(B, U, 4, true)                                                                                   \
    HPCG_SV_LDGV3_POL(B, U, 8, true)
#define HPCG_SV_LDGV3_BLK(B)                                                                                           \
    HPCG_SV_LDGV3_ROW(B, 1) HPCG_SV_LDGV3_ROW(B, 2) HPCG_SV_LDGV3_ROW(B, 3) HPCG_SV_LDGV3_ROW(B, 4)

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
#define HPCG_SV_LDGV3_DEEP(B, U, WI) HPCG_SV_LDGV3_POL(B, U, WI, false)
#define HPCG_SV_LDGV3_BLK_DEEP(B)                                                                                      \
    HPCG_SV_LDGV3_DEEP(B, 5, 1)                                                                                        \
    HPCG_SV_LDGV3_DEEP(B, 6, 1)                                                                                        \
    HPCG_SV_LDGV3_DEEP(B, 7, 1)                                                                                        \
    HPCG_SV_LDGV3_DEEP(B, 8, 1)                                                                                        \
    HPCG_SV_LDGV3_DEEP(B, 10, 1)                                                                                       \
    HPCG_SV_LDGV3_DEEP(B, 14, 1)                                                                                       \
    HPCG_SV_LDGV3_DEEP(B, 5, 2)                                                                                        \
    HPCG_SV_LDGV3_DEEP(B, 6, 2) HPCG_SV_LDGV3_DEEP(B, 7, 2) HPCG_SV_LDGV3_DEEP(B, 8, 2)
    HPCG_SV_LDGV3_BLK(32)
    HPCG_SV_LDGV3_BLK(64)
    HPCG_SV_LDGV3_BLK(128)
    HPCG_SV_LDGV3_BLK(256)
    HPCG_SV_LDGV3_BLK_DEEP(32)
    HPCG_SV_LDGV3_BLK_DEEP(64)
    HPCG_SV_LDGV3_BLK_DEEP(128)
    HPCG_SV_LDGV3_BLK_DEEP(256)
#undef HPCG_SV_LDGV3_BLK_DEEP
#undef HPCG_SV_LDGV3_DEEP
#undef HPCG_SV_LDGV3_BLK
#undef HPCG_SV_LDGV3_ROW
#undef HPCG_SV_LDGV3_POL
#undef HPCG_SV_LDGV3_CASE
    return false;
}

// Both slice-offset widths, 32-bit columns: --mi 0 and --mi 1 respectively.
template bool SpsvLdgV3SellCfg<idx32_t>(bool, const SparseMatrix&, const double*, double*, const idx32_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int, bool, bool);
template bool SpsvLdgV3SellCfg<idx64_t>(bool, const SparseMatrix&, const double*, double*, const idx64_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int, bool, bool);

#endif // EXPLICIT_KERNELS
#endif // USE_CUDA
