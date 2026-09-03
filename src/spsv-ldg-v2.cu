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
 @file spsv-ldg-v2.cu

 LDG_V2 Sliced-ELL triangular solve: two-stage register pipeline, W rows per
 thread, wide (128/256-bit) gather. Instantiated for 32-bit slice offsets and
 32-bit columns only, like the rest of the explicit path.
 */

#ifdef USE_CUDA
#ifdef EXPLICIT_KERNELS

#include "CudaKernels.hpp"
#include "IndexMode.hpp"
#include "SparseMatrix.hpp"
#include "ldg-loads.cuh"

#include <cuda_runtime.h>

namespace
{

// Streaming (.cs, evict-first) policy knobs, split so they can be bisected.
//
//   kStreamMatrix: the column/value stream is read exactly once per call, so
//   .cs keeps it from evicting x, which IS reused across the gather.
//
//   kStreamVector: covers rhs and diagonal, which are read once per sweep, AND
//   the x[r] write. Note x is NOT write-once across a SymGS sweep: color c
//   writes [color_str, color_end) and colors c+1.. gather from it, and on the
//   coarse levels the whole x fits in L2. So evict-first on that store can cost
//   more than it saves.
constexpr bool kStreamMatrix = true;
constexpr bool kStreamVector = false;

template <int UNROLL, int W>
__device__ __forceinline__ void LoadBlock(int kb, int max_row_len, local_int_t slice_size,
    const idx32_t* __restrict__ cp, const double* __restrict__ vp, int (&cols)[UNROLL][W], double (&vals)[UNROLL][W])
{
#pragma unroll
    for (int ki = 0; ki < UNROLL; ++ki)
    {
        const int k = kb + ki;
        if (k < max_row_len)
        {
            // Widen: the flat element offset of a column stripe exceeds 2^31 for
            // large local problems even though the row and stripe indices do not.
            const slice_ptr_t off = (slice_ptr_t) k * slice_size;
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

/*
    GPU Kernel
    Sliced-ELL triangular solve for the rows of a single color, W consecutive
    rows per thread. Rows of one color have no dependence on each other, so the
    host serializes the colors and each launch is a plain SpMV plus a diagonal
    division. x is both the source and the destination: the entries it gathers
    belong to earlier colors, already solved by earlier launches.
*/
template <class OffsetT, int BLKDIM, int UNROLL, int W>
__global__ __launch_bounds__(BLKDIM) void spsv_sell_ldgv2(local_int_t color_str, local_int_t color_end,
    double* __restrict__ x, const double* __restrict__ rhs, const OffsetT* __restrict__ slice_offsets,
    const idx32_t* __restrict__ col_idx, const double* __restrict__ values, const double* __restrict__ diag,
    double alpha, local_int_t slice_size)
{
    const local_int_t base_row = ((local_int_t) blockIdx.x * BLKDIM + (local_int_t) threadIdx.x) * W + color_str;
    if (base_row >= color_end)
        return;

    const local_int_t slice = base_row / slice_size;
    const local_int_t in_slice = base_row - slice * slice_size;
    // Flat element offsets into the column/value arrays exceed 2^31 for large
    // local problems, so widen before they enter the pointer arithmetic even
    // when OffsetT itself is 32-bit.
    const slice_ptr_t row_start = (slice_ptr_t) slice_offsets[slice] + in_slice;
    // Per-slice nnz is bounded by slice_size * HPCG_MAX_ROW_LEN and so fits in
    // int. Narrowing before the divide keeps this a 32-bit idiv instead of the
    // emulated 64-bit one a wider OffsetT would otherwise force.
    const int max_row_len = (int) (slice_offsets[slice + 1] - slice_offsets[slice]) / (int) slice_size;

    const idx32_t* cp = col_idx + row_start;
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
        const local_int_t r = base_row + w;
        if (r < color_end)
        {
            const double rv = ldgload::LoadScalarRo<kStreamVector>(&rhs[r]);
            const double dv = ldgload::LoadScalarRo<kStreamVector>(&diag[r]);
            ldgload::StoreScalar<kStreamVector>(&x[r], (alpha * rv - sum[w]) / dv);
        }
    }
}

template <class OffsetT, int BLKDIM, int UNROLL, int W>
bool LaunchSpsvLdgV2(bool forward, const SparseMatrix& A, const double* rv, double* xv, const OffsetT* slice_offsets,
    const idx32_t* columns, const double* values, cudaStream_t stream)
{
    const local_int_t rows = A.localNumberOfRows;
    const local_int_t slice_size = A.slice_size;
    // Colors of unequal size would put a color boundary inside a thread's W-row
    // chunk, so the solve is only expressible here when they divide the rows.
    if (A.totalColors <= 0 || rows % A.totalColors != 0)
        return false;
    const local_int_t color_size = rows / A.totalColors;

    // A W-element vector access must be naturally aligned, and it spans W whole
    // rows. Slice offsets are multiples of slice_size, so a row's offset into
    // the arrays is a multiple of W exactly when slice_size and the row index
    // both are; the row index here starts at a color boundary, hence color_size.
    if (slice_size % W != 0 || color_size % W != 0)
        return false;

    const double alpha = 1.0;
    const local_int_t nthreads = color_size / W;
    const local_int_t grid = (nthreads + BLKDIM - 1) / BLKDIM;

    const local_int_t first = forward ? 0 : A.totalColors - 1;
    const local_int_t step = forward ? 1 : -1;
    for (local_int_t i = 0; i < A.totalColors; ++i)
    {
        const local_int_t color = first + i * step;
        const local_int_t color_str = color * color_size;
        spsv_sell_ldgv2<OffsetT, BLKDIM, UNROLL, W><<<grid, BLKDIM, 0, stream>>>(color_str, color_str + color_size, xv,
            rv, slice_offsets, columns, values, A.diagonal, alpha, slice_size);
    }
    return true;
}

} // namespace

/*
    Turn the runtime block size, unroll depth and rows-per-thread knobs into the
    matching kernel instantiation. Returns false when the triple, or the shape it
    implies, is not one this family can serve, so the caller can say so rather
    than substitute a different kernel.
*/
template <class OffsetT>
bool SpsvLdgV2SellCfg(bool forward, const SparseMatrix& A, const double* rv, double* xv, const OffsetT* slice_offsets,
    const idx32_t* columns, const double* values, cudaStream_t stream, int blk, int unroll, int w)
{
#define HPCG_SV_LDGV2_CASE(B, U, WI)                                                                                   \
    if (blk == (B) && unroll == (U) && w == (WI))                                                                      \
        return LaunchSpsvLdgV2<OffsetT, B, U, WI>(                                                                     \
            forward, A, rv, xv, slice_offsets, columns, values, stream);
#define HPCG_SV_LDGV2_ROW(B, U)                                                                                        \
    HPCG_SV_LDGV2_CASE(B, U, 1) HPCG_SV_LDGV2_CASE(B, U, 2) HPCG_SV_LDGV2_CASE(B, U, 4) HPCG_SV_LDGV2_CASE(B, U, 8)
#define HPCG_SV_LDGV2_BLK(B)                                                                                           \
    HPCG_SV_LDGV2_ROW(B, 1) HPCG_SV_LDGV2_ROW(B, 2) HPCG_SV_LDGV2_ROW(B, 3) HPCG_SV_LDGV2_ROW(B, 4)
    HPCG_SV_LDGV2_BLK(32)
    HPCG_SV_LDGV2_BLK(64)
    HPCG_SV_LDGV2_BLK(128)
    HPCG_SV_LDGV2_BLK(256)
#undef HPCG_SV_LDGV2_BLK
#undef HPCG_SV_LDGV2_ROW
#undef HPCG_SV_LDGV2_CASE
    return false;
}

template bool SpsvLdgV2SellCfg<idx32_t>(bool, const SparseMatrix&, const double*, double*, const idx32_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int);

#endif // EXPLICIT_KERNELS
#endif // USE_CUDA
