/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#ifdef USE_CUDA

#include "SparseMatrix.hpp"
#include "intdiv.hh"

#include <cstdio>
#include <cstdlib>
#include <cuda/ptx>
#include <cuda_runtime.h>

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

template <int BLKDIM, int UNROLL, int RPT>
__global__ __launch_bounds__(BLKDIM) void mv_sell_tma(int m, double alpha, double beta, double* __restrict__ y,
    const double* __restrict__ x, const slice_ptr_t* __restrict__ slice_offsets,
    const local_int_t* __restrict__ col_idx, const double* __restrict__ values, int slice_size,
    intdiv32_t slice_size_div)
{
    constexpr int ROWS = BLKDIM * RPT;

    extern __shared__ __align__(16) char smem_raw[];
    local_int_t* s_col = reinterpret_cast<local_int_t*>(smem_raw);
    double* s_val = reinterpret_cast<double*>(smem_raw + (size_t) 2 * UNROLL * ROWS * sizeof(local_int_t));
    uint64_t* bar = reinterpret_cast<uint64_t*>(
        smem_raw + (size_t) 2 * UNROLL * ROWS * (sizeof(local_int_t) + sizeof(double)));

    auto scol = [&](int buf, int e) -> local_int_t* { return s_col + ((size_t) buf * UNROLL + e) * ROWS; };
    auto sval = [&](int buf, int e) -> double* { return s_val + ((size_t) buf * UNROLL + e) * ROWS; };

    const int tid = threadIdx.x;
    const int cta_row_base = blockIdx.x * ROWS;

    int slice, in_slice_base;
    intdiv32_divmod(cta_row_base, slice_size, slice_size_div, &slice, &in_slice_base);
    const size_t block_base = (size_t) slice_offsets[slice] + (size_t) in_slice_base;
    int max_row_len;
    intdiv32_div((int32_t) (slice_offsets[slice + 1] - slice_offsets[slice]), slice_size_div, &max_row_len);

    if (IsWarpZero() && IsElectedLane())
    {
        ptx::mbarrier_init(&bar[0], 1);
        ptx::mbarrier_init(&bar[1], 1);
        ptx::fence_proxy_async(ptx::space_shared);
    }
    __syncthreads();

    auto load = [&](int buf, int kb)
    {
        if (IsWarpZero() && IsElectedLane())
        {
            int valid = 0;
#pragma unroll
            for (int e = 0; e < UNROLL; ++e)
            {
                const int k = kb + e;
                if (k < max_row_len)
                {
                    const size_t off = block_base + (size_t) k * slice_size;
                    ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, scol(buf, e), col_idx + off,
                        static_cast<uint32_t>(ROWS * sizeof(local_int_t)), &bar[buf]);
                    ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, sval(buf, e), values + off,
                        static_cast<uint32_t>(ROWS * sizeof(double)), &bar[buf]);
                    ++valid;
                }
            }
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[buf],
                static_cast<uint32_t>(valid * ROWS * (sizeof(local_int_t) + sizeof(double))));
        }
    };

    auto consume = [&](int buf, int kb, double (&sum)[RPT])
    {
#pragma unroll
        for (int e = 0; e < UNROLL; ++e)
        {
            if (kb + e < max_row_len)
            {
                const local_int_t* c = scol(buf, e);
                const double* v = sval(buf, e);
                local_int_t cols[RPT];
                double b[RPT];
#pragma unroll
                for (int mm = 0; mm < RPT; ++mm)
                    cols[mm] = c[tid + mm * BLKDIM];
#pragma unroll
                for (int mm = 0; mm < RPT; ++mm)
                    b[mm] = (cols[mm] >= 0) ? x[cols[mm]] : 0.0;
#pragma unroll
                for (int mm = 0; mm < RPT; ++mm)
                    sum[mm] += v[tid + mm * BLKDIM] * b[mm];
            }
        }
    };

    double sum[RPT];
#pragma unroll
    for (int mm = 0; mm < RPT; ++mm)
        sum[mm] = 0.0;

    int parity0 = 0, parity1 = 0;

    load(0, 0);
    for (int kb = 0; kb < max_row_len; kb += 2 * UNROLL)
    {
        load(1, kb + UNROLL);
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar[0], parity0)) { }
        parity0 ^= 1;
        ptx::fence_proxy_async(ptx::space_shared);
        consume(0, kb, sum);
        __syncthreads();

        load(0, kb + 2 * UNROLL);
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, &bar[1], parity1)) { }
        parity1 ^= 1;
        ptx::fence_proxy_async(ptx::space_shared);
        consume(1, kb + UNROLL, sum);
        __syncthreads();
    }

#pragma unroll
    for (int mm = 0; mm < RPT; ++mm)
    {
        const int row = cta_row_base + tid + mm * BLKDIM;
        if (row < m)
            y[row] = (beta == 0.0) ? (alpha * sum[mm]) : (beta * y[row] + alpha * sum[mm]);
    }
}

int MaxOptinSmem()
{
    static int v = [] {
        int dev = 0, s = 48 * 1024;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return s;
    }();
    return v;
}

template <int BLKDIM, int UNROLL, int RPT>
bool LaunchMvTma(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const slice_ptr_t* slice_offsets, const local_int_t* columns, const double* values, cudaStream_t stream)
{
    constexpr int ROWS = BLKDIM * RPT;
    const local_int_t m = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;
    if (ROWS > slice_size || slice_size % ROWS != 0 || m % ROWS != 0)
        return false;

    constexpr size_t smem
        = (size_t) 2 * UNROLL * ROWS * (sizeof(local_int_t) + sizeof(double)) + 2 * sizeof(uint64_t);
    if ((int) smem > MaxOptinSmem())
        return false;

    auto kernel = mv_sell_tma<BLKDIM, UNROLL, RPT>;
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr_set = true;
    }

    const intdiv32_t sdiv = intdiv32_gen(slice_size);
    const local_int_t grid = m / ROWS;
    kernel<<<grid, BLKDIM, smem, stream>>>(
        (int) m, alpha, beta, y, x, slice_offsets, columns, values, slice_size, sdiv);
    return true;
}

}

bool MvTmaSellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    slice_ptr_t* sell_block_offset, local_int_t* sell_columns, double* sell_values, cudaStream_t stream, int blk,
    int unroll, int rpt)
{
#define TRY(B, U, R)                                                                                                   \
    if (blk == (B) && unroll == (U) && rpt == (R))                                                                     \
        return LaunchMvTma<B, U, R>(A, alpha, beta, x, y, sell_block_offset, sell_columns, sell_values, stream);
#define ROW(B, U) TRY(B, U, 1) TRY(B, U, 2) TRY(B, U, 4) TRY(B, U, 8)
#define BLK_ROWS(B) ROW(B, 1) ROW(B, 2) ROW(B, 3) ROW(B, 4) ROW(B, 6) ROW(B, 8)
    BLK_ROWS(32)
    BLK_ROWS(64)
    BLK_ROWS(128)
    BLK_ROWS(256)
#undef BLK_ROWS
#undef ROW
#undef TRY
    return false;
}

#endif
