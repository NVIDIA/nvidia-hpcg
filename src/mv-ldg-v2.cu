/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*
 * LDG_V2 SpMV: two-stage register pipeline, W rows/thread, wide (128/256-bit)
 * sliced-ELL gather.
 *
 * Cache policy is selected by kStreamMatrix / kStreamVector below. Streaming
 * uses inline PTX (see ldg-loads.cuh) so the wide accesses survive -- __ldcs has
 * no 4-wide double form, which is the width LDG3 had to give up to get .cs.
 *
 * The streaming paths load columns as .u32, so this family is compiled only when
 * local_int_t is 32-bit. INDEX_64 is not supported and gets a stub that reports
 * every config as unavailable. (The previous int2/int4 reinterprets had the same
 * 32-bit requirement but no guard.)
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

// Streaming (.cs, evict-first) policy knobs, split so they can be bisected.
//
//   kStreamMatrix: the column/value stream is read exactly once per call, so .cs
//   keeps it from evicting x, which IS reused across the gather. This is the
//   LDG3 result, here without giving up the wide accesses. Set false to recover
//   the original wide+caching LDG_V2 for an A/B.
//
//   kStreamVector: the y write, and its read-back under beta != 0, is not
//   touched again by this kernel. Marking it evict-first bets nothing downstream
//   wants it in L2 either -- clearly true at L0 where y far exceeds L2, less
//   obviously so on the coarse levels. Set false to isolate.
constexpr bool kStreamMatrix = true;
constexpr bool kStreamVector = false;

template <int UNROLL, int W>
__device__ __forceinline__ void LoadBlock(int kb, int max_row_len, int slice_size, const local_int_t* __restrict__ cp,
    const double* __restrict__ vp, int (&cols)[UNROLL][W], double (&vals)[UNROLL][W])
{
#pragma unroll
    for (int ki = 0; ki < UNROLL; ++ki)
    {
        const int k = kb + ki;
        if (k < max_row_len)
        {
            const size_t off = (size_t) k * slice_size;
            ldgload::LoadCols<W, kStreamMatrix>(&cp[off], cols[ki]);
            ldgload::LoadVals<W, kStreamMatrix>(&vp[off], vals[ki]);
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

// `parts` splits the row space into that many equal partitions and hands one to
// each blockIdx.x, with blockIdx.y walking rows inside a partition. Because
// blocks are dispatched x-fastest, the co-resident blocks then sit at the same
// offset in `parts` widely separated regions instead of forming one contiguous
// sweep -- this is LDG3's geometry, generalised off its hard-coded 8.
//
// parts == 1 keeps the original flat 1-D grid, where the row index lives in
// blockIdx.x. That case is kept separate deliberately: gridDim.y caps at 65535,
// so folding it into the 2-D shape would reject nearly every config at the
// finest level (m/(W*BLKDIM) there runs to ~295k).
//
// `parts` is a plain kernel argument, not a template parameter: its only uses
// are a host-side launch dimension and partition_rows, which the host passes in
// precomputed. Templating it would multiply the 64 existing instantiations by
// the sweep length for byte-identical device code -- and the runtime form is
// strictly better, since it removes the m/parts division rather than
// strength-reducing it.
template <int BLKDIM, int UNROLL, int W>
__global__ __launch_bounds__(BLKDIM) void mv_sell_ldgv2(int m, int parts, int partition_rows, double alpha, double beta,
    double* __restrict__ y, const double* __restrict__ x, const slice_ptr_t* __restrict__ slice_offsets,
    const local_int_t* __restrict__ col_idx, const double* __restrict__ values, int slice_size,
    intdiv32_t slice_size_div)
{
    int base_row, row_end;
    if (parts == 1)
    {
        base_row = ((int) blockIdx.x * BLKDIM + (int) threadIdx.x) * W;
        row_end = m;
    }
    else
    {
        const int row_in_partition = ((int) blockIdx.y * BLKDIM + (int) threadIdx.x) * W;
        base_row = (int) blockIdx.x * partition_rows + row_in_partition;
        row_end = ((int) blockIdx.x + 1) * partition_rows;
    }
    if (base_row >= row_end)
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

    LoadBlock<UNROLL, W>(0, max_row_len, slice_size, cp, vp, cA, vA);

    double sum[W];
#pragma unroll
    for (int w = 0; w < W; ++w)
        sum[w] = 0.0;

    for (int kb = 0; kb < max_row_len; kb += 2 * UNROLL)
    {
        LoadBlock<UNROLL, W>(kb + UNROLL, max_row_len, slice_size, cp, vp, cB, vB);
        ConsumeBlock<UNROLL, W>(sum, cA, vA, x);
        LoadBlock<UNROLL, W>(kb + 2 * UNROLL, max_row_len, slice_size, cp, vp, cA, vA);
        ConsumeBlock<UNROLL, W>(sum, cB, vB, x);
    }

#pragma unroll
    for (int w = 0; w < W; ++w)
    {
        const int r = base_row + w;
        if (r < row_end)
        {
            const double prev = (beta == 0.0) ? 0.0 : beta * ldgload::LoadScalarRw<kStreamVector>(&y[r]);
            ldgload::StoreScalar<kStreamVector>(&y[r], prev + alpha * sum[w]);
        }
    }
}

template <int BLKDIM, int UNROLL, int W>
bool LaunchMvLdgV2(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const slice_ptr_t* slice_offsets, const local_int_t* columns, const double* values, cudaStream_t stream, int parts)
{
    const local_int_t m = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;
    if (parts < 1 || slice_size % W != 0 || m % W != 0)
        return false;

    const intdiv32_t sdiv = intdiv32_gen(slice_size);

    if (parts == 1)
    {
        const local_int_t grid = (m / W + BLKDIM - 1) / BLKDIM;
        mv_sell_ldgv2<BLKDIM, UNROLL, W><<<dim3((unsigned int) grid, 1, 1), BLKDIM, 0, stream>>>(
            (int) m, 1, (int) m, alpha, beta, y, x, slice_offsets, columns, values, slice_size, sdiv);
        return true;
    }

    // Partitions must divide the rows exactly, and each partition must still be
    // a whole number of W-row thread chunks.
    if (m % parts != 0)
        return false;
    const local_int_t partition_rows = m / parts;
    if (partition_rows % W != 0)
        return false;
    const unsigned int grid_y = (unsigned int) ((partition_rows / W + BLKDIM - 1) / BLKDIM);
    if (grid_y > 65535u)
        return false;
    mv_sell_ldgv2<BLKDIM, UNROLL, W><<<dim3((unsigned int) parts, grid_y, 1), BLKDIM, 0, stream>>>(
        (int) m, parts, (int) partition_rows, alpha, beta, y, x, slice_offsets, columns, values, slice_size, sdiv);
    return true;
}

}

bool MvLdgV2SellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    slice_ptr_t* sell_block_offset, local_int_t* sell_columns, double* sell_values, cudaStream_t stream, int blk,
    int unroll, int w, int parts)
{
#define TRY(B, U, WI)                                                                                                  \
    if (blk == (B) && unroll == (U) && w == (WI))                                                                      \
        return LaunchMvLdgV2<B, U, WI>(                                                                                \
            A, alpha, beta, x, y, sell_block_offset, sell_columns, sell_values, stream, parts);
#define ROW(B, U) TRY(B, U, 1) TRY(B, U, 2) TRY(B, U, 4) TRY(B, U, 8)
#define BLK_ROWS(B) ROW(B, 1) ROW(B, 2) ROW(B, 3) ROW(B, 4)
    BLK_ROWS(32)
    BLK_ROWS(64)
    BLK_ROWS(128)
    BLK_ROWS(256)
#undef BLK_ROWS
#undef ROW
#undef TRY
    return false;
}

#else // INDEX_64

bool MvLdgV2SellCfg(const SparseMatrix&, double, double, const double*, double*, slice_ptr_t*, local_int_t*, double*,
    cudaStream_t, int, int, int, int)
{
    return false;
}

#endif // !INDEX_64
#endif
