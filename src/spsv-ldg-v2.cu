/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*
 * LDG_V2 SymGS triangular solve: two-stage register pipeline, W rows/thread,
 * wide (128/256-bit) sliced-ELL gather.
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
//   kStreamVector: covers rhs and diag, which are read once per sweep, AND the
//   x[r] write. Note x is NOT write-once across a SymGS sweep: colour c writes
//   [color_str, color_end) and colours c+1.. gather from it, and on the coarse
//   levels the whole x fits in L2 (1.2 MB at L3). So evict-first on that store
//   can cost more than it saves -- if SymGS regresses, flip this first.
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

template <int BLKDIM, int UNROLL, int W>
__global__ __launch_bounds__(BLKDIM) void spsv_sell_ldgv2(int color_str, int color_end, double* __restrict__ x,
    const double* __restrict__ rhs, const slice_ptr_t* __restrict__ slice_offsets,
    const local_int_t* __restrict__ col_idx, const double* __restrict__ values, const double* __restrict__ diag,
    double alpha, int slice_size, intdiv32_t slice_size_div)
{
    const int base_row = (blockIdx.x * BLKDIM + threadIdx.x) * W + color_str;
    if (base_row >= color_end)
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
        if (r < color_end)
        {
            const double rv = ldgload::LoadScalarRo<kStreamVector>(&rhs[r]);
            const double dv = ldgload::LoadScalarRo<kStreamVector>(&diag[r]);
            ldgload::StoreScalar<kStreamVector>(&x[r], (alpha * rv - sum[w]) / dv);
        }
    }
}

template <int BLKDIM, int UNROLL, int W>
bool LaunchColorLoopV2(int forward, const SparseMatrix& A, double* rv, double* xv, const slice_ptr_t* slice_offsets,
    const local_int_t* columns, const double* values, cudaStream_t stream)
{
    const local_int_t rows = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;
    if (A.totalColors <= 0 || rows % A.totalColors != 0)
        return false;
    const local_int_t color_size = rows / A.totalColors;

    if (slice_size % W != 0 || color_size % W != 0)
        return false;

    const intdiv32_t sdiv = intdiv32_gen(slice_size);
    const double alpha = 1.0;
    const local_int_t nthreads = color_size / W;
    const local_int_t grid = (nthreads + BLKDIM - 1) / BLKDIM;

    auto kernel = spsv_sell_ldgv2<BLKDIM, UNROLL, W>;
    if (forward)
    {
        for (int color = 0; color < A.totalColors; ++color)
        {
            const int cs = color * color_size;
            const int ce = cs + (int) color_size;
            kernel<<<grid, BLKDIM, 0, stream>>>(
                cs, ce, xv, rv, slice_offsets, columns, values, A.diagonal, alpha, slice_size, sdiv);
        }
    }
    else
    {
        for (int color = A.totalColors - 1; color >= 0; --color)
        {
            const int cs = color * color_size;
            const int ce = cs + (int) color_size;
            kernel<<<grid, BLKDIM, 0, stream>>>(
                cs, ce, xv, rv, slice_offsets, columns, values, A.diagonal, alpha, slice_size, sdiv);
        }
    }
    return true;
}

}

bool SpsvLdgV2SellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, cudaStream_t stream, int blk, int unroll, int w)
{

#define TRY(B, U, WI)                                                                                                  \
    if (blk == (B) && unroll == (U) && w == (WI))                                                                      \
        return LaunchColorLoopV2<B, U, WI>(forward, A, rv, xv, sell_block_offset, sell_columns, sell_values, stream);
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

bool SpsvLdgV2SellCfg(int, const SparseMatrix&, double*, double*, slice_ptr_t*, local_int_t*, double*, cudaStream_t,
    int, int, int)
{
    return false;
}

#endif // !INDEX_64
#endif
