/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

// SpSV counterpart to mv-tma-ex.cu: the explicit branch's
// ex_spsv_sell_single_color_v1_tma_kernel, one colour per launch, one row per
// thread, cp.async.bulk.tensor into static double-buffered shared memory.
//
// The original caller drops to a plain (64, 4) kernel when the colour stride is
// not a multiple of the slice size. That substitution would be invisible in a
// family time here -- the autotuner would attribute another kernel's cost to
// this one -- so the shape test below returns false instead and the level is
// reported as n/a.

#ifdef USE_CUDA

#include "SparseMatrix.hpp"

#include <algorithm>
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
__global__ __launch_bounds__(BLKDIM) void spsv_sell_tma_ex(const __grid_constant__ CUtensorMap col_map,
    const __grid_constant__ CUtensorMap val_map, int slice_size, int color_str, int color_end, double* __restrict__ x,
    const double* __restrict__ rhs, const slice_ptr_t* __restrict__ slice_offsets, const double* __restrict__ diag,
    double alpha)
{
    const int row = blockIdx.x * BLKDIM + threadIdx.x + color_str;

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

    // Valid only because the caller checked that the colour stride is a whole
    // number of slices and that BLKDIM divides the slice.
    const int x_offset  = blockIdx.x % (slice_size / BLKDIM);
    const int lane_slot = row_in_slice % BLKDIM;

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

    if (row < color_end)
    {
        const double d = __ldcs(&diag[row]);
        x[row]         = ((alpha * rhs[row]) - sell_sum) / d;
    }
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
        fprintf(stderr, "[spsv-tma-ex] cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "?");
        return false;
    }
    return true;
}

template <int BLKDIM, int UNROLL>
bool LaunchSpsvTmaEx(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* slice_offsets,
    local_int_t* columns, double* values, slice_ptr_t last_nnz, cudaStream_t stream)
{
    const local_int_t rows = A.localNumberOfRows;
    const int slice_size   = (int) A.slice_size;

    constexpr size_t smem = 2 * (size_t) UNROLL * BLKDIM * (sizeof(double) + sizeof(int)) + 2 * sizeof(uint64_t);
    if constexpr (smem > 48 * 1024)
        return false;

    if (A.totalColors <= 0 || rows % A.totalColors != 0)
        return false;
    const local_int_t color_size = rows / A.totalColors;

    // Every block derives its tile coordinate from blockIdx.x, which is only
    // its position within the slice if each colour starts on a slice boundary.
    // On the coarsest grid it does not, and that is where the original falls
    // back to a different kernel.
    if (slice_size % BLKDIM != 0 || color_size % slice_size != 0)
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

    const local_int_t grid = color_size / BLKDIM;
    const double alpha     = 1.0;

    if (forward)
    {
        for (int color = 0; color < A.totalColors; ++color)
        {
            const int cs = color * color_size;
            const int ce = (int) std::min<local_int_t>((color + 1) * (local_int_t) color_size, rows);
            spsv_sell_tma_ex<BLKDIM, UNROLL><<<grid, BLKDIM, 0, stream>>>(
                col_map, val_map, slice_size, cs, ce, xv, rv, slice_offsets, A.diagonal, alpha);
        }
    }
    else
    {
        for (int color = A.totalColors - 1; color >= 0; --color)
        {
            const int cs = color * color_size;
            const int ce = (int) std::min<local_int_t>((color + 1) * (local_int_t) color_size, rows);
            spsv_sell_tma_ex<BLKDIM, UNROLL><<<grid, BLKDIM, 0, stream>>>(
                col_map, val_map, slice_size, cs, ce, xv, rv, slice_offsets, A.diagonal, alpha);
        }
    }
    return true;
}

} // namespace

bool SpsvTmaExSellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, slice_ptr_t last_nnz, cudaStream_t stream, int blk, int unroll,
    int rpt)
{
    if (rpt != 1)
        return false;

#define TRY(B, U)                                                                                                      \
    if (blk == (B) && unroll == (U))                                                                                   \
        return LaunchSpsvTmaEx<B, U>(                                                                                  \
            forward, A, rv, xv, sell_block_offset, sell_columns, sell_values, last_nnz, stream);
    TRY(64, 4) TRY(64, 6) TRY(64, 7) TRY(64, 8) TRY(64, 10) TRY(64, 12)
    TRY(128, 4) TRY(128, 6) TRY(128, 8)
    TRY(256, 4) TRY(256, 6) TRY(256, 7)
#undef TRY
    return false;
}

#endif
