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
 @file mv-ldg-v2.cu

 LDG_V2 Sliced-ELL SpMV: two-stage register pipeline, W rows per thread, wide
 (128/256-bit) gather. Instantiated for 32-bit slice offsets and 32-bit columns
 only, like the rest of the explicit path.
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
//   kStreamVector: the y write, and its read-back under beta != 0, is not
//   touched again by this kernel. Marking it evict-first bets nothing
//   downstream wants it in L2 either -- clearly true at the finest level where
//   y far exceeds L2, less obviously so on the coarse levels.
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
    y = alpha * A * x + beta * y over a Sliced-ELL operator, W consecutive rows
    per thread, the gather widened to one W-element vector access per stripe.

    `parts` splits the row space into that many equal partitions and hands one
    to each blockIdx.x, with blockIdx.y walking rows inside a partition. Because
    blocks are dispatched x-fastest, the co-resident blocks then sit at the same
    offset in `parts` widely separated regions instead of forming one contiguous
    sweep.

    parts == 1 keeps the flat 1D grid, where the row index lives in blockIdx.x.
    That case is kept separate deliberately: gridDim.y caps at 65535, so folding
    it into the 2D shape would reject nearly every launch at the finest level,
    where m/(W*BLKDIM) runs into the hundreds of thousands.

    `parts` is a plain kernel argument, not a template parameter: its only uses
    are a host-side launch dimension and partition_rows, which the host passes
    in precomputed. Templating it would multiply the instantiation count for
    byte-identical device code.
*/
template <class OffsetT, int BLKDIM, int UNROLL, int W>
__global__ __launch_bounds__(BLKDIM) void mv_sell_ldgv2(local_int_t m, int parts, local_int_t partition_rows,
    double alpha, double beta, double* __restrict__ y, const double* __restrict__ x,
    const OffsetT* __restrict__ slice_offsets, const idx32_t* __restrict__ col_idx,
    const double* __restrict__ values, local_int_t slice_size)
{
    local_int_t base_row, row_end;
    if (parts == 1)
    {
        base_row = ((local_int_t) blockIdx.x * BLKDIM + (local_int_t) threadIdx.x) * W;
        row_end = m;
    }
    else
    {
        const local_int_t row_in_partition = ((local_int_t) blockIdx.y * BLKDIM + (local_int_t) threadIdx.x) * W;
        base_row = (local_int_t) blockIdx.x * partition_rows + row_in_partition;
        row_end = ((local_int_t) blockIdx.x + 1) * partition_rows;
    }
    if (base_row >= row_end)
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
        if (r < row_end)
        {
            const double prev = (beta == 0.0) ? 0.0 : beta * ldgload::LoadScalarRw<kStreamVector>(&y[r]);
            ldgload::StoreScalar<kStreamVector>(&y[r], prev + alpha * sum[w]);
        }
    }
}

template <class OffsetT, int BLKDIM, int UNROLL, int W>
bool LaunchMvLdgV2(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const OffsetT* slice_offsets, const idx32_t* columns, const double* values, cudaStream_t stream, int parts)
{
    const local_int_t m = A.localNumberOfRows;
    const local_int_t slice_size = A.slice_size;
    // A W-element vector access must be naturally aligned, and it spans W whole
    // rows. Slice offsets are multiples of slice_size, so a row's offset into
    // the arrays is a multiple of W exactly when slice_size and the row index
    // both are, and m % W == 0 keeps the last chunk from running off the end of
    // the stored rows.
    if (parts < 1 || slice_size % W != 0 || m % W != 0)
        return false;

    if (parts == 1)
    {
        const local_int_t grid = (m / W + BLKDIM - 1) / BLKDIM;
        mv_sell_ldgv2<OffsetT, BLKDIM, UNROLL, W><<<dim3((unsigned int) grid, 1, 1), BLKDIM, 0, stream>>>(
            m, 1, m, alpha, beta, y, x, slice_offsets, columns, values, slice_size);
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
    mv_sell_ldgv2<OffsetT, BLKDIM, UNROLL, W><<<dim3((unsigned int) parts, grid_y, 1), BLKDIM, 0, stream>>>(
        m, parts, partition_rows, alpha, beta, y, x, slice_offsets, columns, values, slice_size);
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
bool MvLdgV2SellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const OffsetT* slice_offsets, const idx32_t* columns, const double* values, cudaStream_t stream, int blk,
    int unroll, int w, int parts)
{
#define HPCG_MV_LDGV2_CASE(B, U, WI)                                                                                   \
    if (blk == (B) && unroll == (U) && w == (WI))                                                                      \
        return LaunchMvLdgV2<OffsetT, B, U, WI>(                                                                       \
            A, alpha, beta, x, y, slice_offsets, columns, values, stream, parts);
#define HPCG_MV_LDGV2_ROW(B, U)                                                                                        \
    HPCG_MV_LDGV2_CASE(B, U, 1) HPCG_MV_LDGV2_CASE(B, U, 2) HPCG_MV_LDGV2_CASE(B, U, 4) HPCG_MV_LDGV2_CASE(B, U, 8)
#define HPCG_MV_LDGV2_BLK(B)                                                                                           \
    HPCG_MV_LDGV2_ROW(B, 1) HPCG_MV_LDGV2_ROW(B, 2) HPCG_MV_LDGV2_ROW(B, 3) HPCG_MV_LDGV2_ROW(B, 4)
    HPCG_MV_LDGV2_BLK(32)
    HPCG_MV_LDGV2_BLK(64)
    HPCG_MV_LDGV2_BLK(128)
    HPCG_MV_LDGV2_BLK(256)
#undef HPCG_MV_LDGV2_BLK
#undef HPCG_MV_LDGV2_ROW
#undef HPCG_MV_LDGV2_CASE
    return false;
}

template bool MvLdgV2SellCfg<idx32_t>(const SparseMatrix&, double, double, const double*, double*, const idx32_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int, int);

#endif // EXPLICIT_KERNELS
#endif // USE_CUDA
