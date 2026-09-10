/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

// SpMV from the explicit-HPCG / cuSPARSE TMA lineage, ported here so the
// autotuner can price it against the other families on equal terms. The kernel
// body is the explicit branch's sellmv_v1_tma_2D_tensor_kernel_double_int32:
// static shared double buffers, an elected lane in warp 0 issuing
// cp.async.bulk.tensor, and one row per thread over an 8-way row partition.
//
// What is deliberately not carried over is the caller's behaviour on an
// unsupported shape. There, an unusable configuration either aborts the launch
// or silently drops to a different kernel. A family in this framework must
// instead report infeasibility, so every shape requirement below returns false
// and lets the autotuner record the level as n/a.

#ifdef USE_CUDA

#include "SparseMatrix.hpp"

#include <cstdio>
#include <cuda.h>
#include <cuda/ptx>

namespace
{
namespace ptx = cuda::ptx;

__device__ __forceinline__ bool IsWarpZero()
{
    const unsigned int warp_id = threadIdx.x / 32;
    return __shfl_sync(0xffffffffu, warp_id, 0) == 0;
}

__device__ __forceinline__ bool IsElectedLane()
{
    uint32_t elected;
    asm volatile("{\n"
                 "  .reg .pred p;\n"
                 "  elect.sync _|p, 0xffffffff;\n"
                 "  selp.b32 %0, 1, 0, p;\n"
                 "}"
                 : "=r"(elected));
    return elected != 0;
}

template <typename T>
__device__ __forceinline__ void Swap(T& a, T& b)
{
    T t = a;
    a   = b;
    b   = t;
}

template <int BLKDIM, int UNROLL>
__global__ __launch_bounds__(BLKDIM) void mv_sell_tma_ex(const __grid_constant__ CUtensorMap col_map,
    const __grid_constant__ CUtensorMap val_map, int m, double alpha, double beta, int slice_size,
    const slice_ptr_t* __restrict__ slice_offsets, const double* __restrict__ x, double* __restrict__ y)
{
    const int ty  = blockIdx.x; // which of the 8 row partitions
    const int row = threadIdx.x + BLKDIM * blockIdx.y + ty * (m / 8);

    const int row_in_slice = row % slice_size;
    const int row_slice    = row / slice_size;

    __shared__ uint64_t bar[2];
    uint64_t* bar_curr = &bar[0];
    uint64_t* bar_next = &bar[1];

    __shared__ __align__(128) char a_vals[UNROLL * BLKDIM * sizeof(double)];
    __shared__ __align__(128) char a_cols[UNROLL * BLKDIM * sizeof(int)];
    __shared__ __align__(128) char b_vals[UNROLL * BLKDIM * sizeof(double)];
    __shared__ __align__(128) char b_cols[UNROLL * BLKDIM * sizeof(int)];

    double* curr_vals = reinterpret_cast<double*>(a_vals);
    int* curr_cols    = reinterpret_cast<int*>(a_cols);
    double* next_vals = reinterpret_cast<double*>(b_vals);
    int* next_cols    = reinterpret_cast<int*>(b_cols);

    __shared__ slice_ptr_t slice_start_shared, slice_end_shared;

    if (IsWarpZero() && IsElectedLane())
    {
        ptx::mbarrier_init(bar_curr, 1);
        ptx::mbarrier_init(bar_next, 1);
        ptx::fence_proxy_async(ptx::space_shared);

        slice_start_shared = slice_offsets[row_slice];
        slice_end_shared   = slice_offsets[row_slice + 1];
    }
    __syncthreads();

    // The block owns one BLKDIM-wide column of the slice; which one is fixed by
    // the caller having checked that slice_size and m/8 are both multiples of
    // BLKDIM, so this offset is the block's true position inside its slice.
    const int x_offset  = blockIdx.y % (slice_size / BLKDIM);
    const int lane_slot = row_in_slice % BLKDIM;

    // Cast before the divide: slice_ptr_t may be 64-bit and a 64-bit idiv is
    // emulated on device.
    const int slice_nnz  = (int) (slice_end_shared - slice_start_shared);
    int max_row_len      = slice_nnz / slice_size;
    const int entry_base = (int) (slice_start_shared / slice_size);
    const int unroll_nz  = (max_row_len + UNROLL - 1) / UNROLL;

    double sell_sum = 0.0;
    constexpr int kTileBytes = BLKDIM * UNROLL * (sizeof(double) + sizeof(int));

    if (IsWarpZero() && IsElectedLane())
    {
        int32_t coords[2] = {x_offset * BLKDIM, entry_base};
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, curr_vals, &val_map, coords, bar_curr);
        ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, curr_cols, &col_map, coords, bar_curr);
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_curr, kTileBytes);
    }

    int stage = 0;
    stage ^= 1;
    __syncthreads();
    int parity = 0;

#pragma unroll 1
    for (int q = 1; q < unroll_nz - 1; q++)
    {
        if (IsWarpZero() && IsElectedLane())
        {
            int32_t coords[2] = {x_offset * BLKDIM, entry_base + q * UNROLL};
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, next_vals, &val_map, coords, bar_next);
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, next_cols, &col_map, coords, bar_next);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_next, kTileBytes);
        }

        stage ^= 1;
        while (!ptx::mbarrier_try_wait_parity(bar_curr, parity))
        {
        }
        parity ^= stage;

#pragma unroll UNROLL
        for (int k = 0; k < UNROLL; k++)
        {
            const int slot = lane_slot + k * BLKDIM;
            if (curr_cols[slot] >= 0)
                sell_sum += curr_vals[slot] * x[curr_cols[slot]];
        }

        // Required: the swap below hands these buffers to the next fetch.
        __syncthreads();

        Swap(bar_curr, bar_next);
        Swap(curr_vals, next_vals);
        Swap(curr_cols, next_cols);

        max_row_len -= UNROLL;
    }

    if (1 < unroll_nz)
    {
        max_row_len -= UNROLL;
        if (IsWarpZero() && IsElectedLane())
        {
            int32_t coords[2] = {x_offset * BLKDIM, entry_base + (unroll_nz - 1) * UNROLL};
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, next_vals, &val_map, coords, bar_next);
            ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, next_cols, &col_map, coords, bar_next);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_next, kTileBytes);
        }

        stage ^= 1;
        while (!ptx::mbarrier_try_wait_parity(bar_curr, parity))
        {
        }
        parity ^= stage;

#pragma unroll UNROLL
        for (int k = 0; k < UNROLL; k++)
        {
            const int slot = lane_slot + k * BLKDIM;
            if (curr_cols[slot] >= 0)
                sell_sum += curr_vals[slot] * x[curr_cols[slot]];
        }

        Swap(bar_curr, bar_next);
        Swap(curr_vals, next_vals);
        Swap(curr_cols, next_cols);
    }

    stage ^= 1;
    while (!ptx::mbarrier_try_wait_parity(bar_curr, parity))
    {
    }
    parity ^= stage;

#pragma unroll UNROLL
    for (int k = 0; k < UNROLL; k++)
    {
        const int slot = lane_slot + k * BLKDIM;
        if (curr_cols[slot] >= 0 && k < max_row_len)
            sell_sum += curr_vals[slot] * x[curr_cols[slot]];
    }

    if (beta == 0.0)
        y[row] = alpha * sell_sum;
    else
        y[row] = beta * y[row] + alpha * sell_sum;
}

bool MakeMap(CUtensorMap* map, void* addr, CUtensorMapDataType dtype, size_t elem, uint64_t total_entries, int box_rows,
    int unroll, int slice_size)
{
    const uint64_t gdim[2] = {(uint64_t) slice_size, total_entries};
    const uint64_t gstr[1] = {(uint64_t) slice_size * elem};
    const uint32_t bdim[2] = {(uint32_t) box_rows, (uint32_t) unroll};
    const uint32_t estr[2] = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(map, dtype, 2, addr, gdim, gstr, bdim, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS)
    {
        const char* msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "[mv-tma-ex] cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "?");
        return false;
    }
    return true;
}

template <int BLKDIM, int UNROLL>
bool LaunchMvTmaEx(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    slice_ptr_t* slice_offsets, local_int_t* columns, double* values, slice_ptr_t last_nnz, cudaStream_t stream)
{
    const local_int_t m  = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;

    // Static shared, as in the original. 48 KiB is the ceiling without opt-in.
    constexpr size_t smem = 2 * (size_t) UNROLL * BLKDIM * (sizeof(double) + sizeof(int)) + 2 * sizeof(uint64_t);
    if constexpr (smem > 48 * 1024)
        return false;

    // The kernel has no row bounds test and derives its tile coordinate from
    // blockIdx.y alone, so a partition that is not slice-aligned would read a
    // window belonging to other rows. Both divisions must come out exact.
    if (slice_size % BLKDIM != 0 || m % 8 != 0)
        return false;
    const local_int_t part = m / 8;
    if (part % BLKDIM != 0 || part % slice_size != 0)
        return false;
    if (last_nnz % slice_size != 0)
        return false;

    CUtensorMap col_map, val_map;
    const uint64_t total_entries = (uint64_t) (last_nnz / slice_size);
    if (!MakeMap(&val_map, (void*) values, CU_TENSOR_MAP_DATA_TYPE_FLOAT64, sizeof(double), total_entries, BLKDIM,
            UNROLL, slice_size))
        return false;
    if (!MakeMap(&col_map, (void*) columns, CU_TENSOR_MAP_DATA_TYPE_INT32, sizeof(local_int_t), total_entries, BLKDIM,
            UNROLL, slice_size))
        return false;

    const dim3 grid(8, (unsigned int) (part / BLKDIM), 1);
    mv_sell_tma_ex<BLKDIM, UNROLL><<<grid, BLKDIM, 0, stream>>>(
        col_map, val_map, (int) m, alpha, beta, slice_size, slice_offsets, x, y);
    return true;
}

} // namespace

bool MvTmaExSellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    slice_ptr_t* sell_block_offset, local_int_t* sell_columns, double* sell_values, slice_ptr_t last_nnz,
    cudaStream_t stream, int blk, int unroll, int rpt)
{
    // One row per thread is inherent to this kernel; the rpt axis stays at 1.
    if (rpt != 1)
        return false;

#define TRY(B, U)                                                                                                      \
    if (blk == (B) && unroll == (U))                                                                                   \
        return LaunchMvTmaEx<B, U>(A, alpha, beta, x, y, sell_block_offset, sell_columns, sell_values, last_nnz, stream);
    TRY(64, 1) TRY(64, 4) TRY(64, 6) TRY(64, 7) TRY(64, 8)
    TRY(128, 1) TRY(128, 4) TRY(128, 6) TRY(128, 7) TRY(128, 8)
    TRY(256, 1) TRY(256, 4) TRY(256, 6) TRY(256, 7)
#undef TRY
    return false;
}

#endif
