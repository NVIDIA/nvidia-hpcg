/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*
 * TMA2D is disabled under INDEX_64.
 *
 * MakeMap describes the column array to cuTensorMapEncodeTiled with a hardcoded
 * CU_TENSOR_MAP_DATA_TYPE_INT32 while deriving its stride from
 * sizeof(local_int_t). Under INDEX_64 those disagree -- 8-byte elements
 * described as 4-byte ones -- and the TMA silently reads wrong columns: it
 * compiles, raises no error, and just produces bad results.
 *
 * A 32-bit-index build is fine: INT32 and sizeof(local_int_t) agree. The slice
 * offsets are unconditionally 64-bit and are carried as slice_ptr_t / uint64_t
 * throughout this file.
 *
 * Fixing INDEX_64 properly means picking the tensor-map data type from
 * sizeof(local_int_t) instead of hardcoding it; until then the stubs below
 * report every config unavailable and the autotuner prints TMA2D as n/a.
 */

#ifdef USE_CUDA

#include "SparseMatrix.hpp"
#include "intdiv.hh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda/ptx>

#ifndef INDEX_64

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

struct __align__(32) double4a
{
    double x, y, z, w;
};

template <int N>
__device__ __forceinline__ void LoadVecD(const double* __restrict__ p, double (&v)[N])
{
    if constexpr (N == 4)
        *(double4a*) v = *(const double4a*) p;
    else if constexpr (N == 2)
        *(double2*) v = *(const double2*) p;
    else
        for (int i = 0; i < N; ++i)
            v[i] = p[i];
}

template <int N>
__device__ __forceinline__ void StoreVecD(double* __restrict__ p, const double (&v)[N])
{
    if constexpr (N == 4)
        *(double4a*) p = *(const double4a*) v;
    else if constexpr (N == 2)
        *(double2*) p = *(const double2*) v;
    else
        for (int i = 0; i < N; ++i)
            p[i] = v[i];
}

template <int BLKDIM, int UNROLL, int RPT>
__global__ __launch_bounds__(BLKDIM) void spsv_sell_tma2d(int color_str, int color_end, double* __restrict__ x,
    const double* __restrict__ rhs, const double* __restrict__ diag, const slice_ptr_t* __restrict__ slice_offsets,
    double alpha, int slice_size, intdiv32_t slice_size_div, const __grid_constant__ CUtensorMap val_map,
    const __grid_constant__ CUtensorMap col_map)
{
    constexpr int ROWS  = BLKDIM * RPT;
    constexpr int TROWS = (ROWS <= 256) ? ROWS : 256;
    constexpr int NT    = ROWS / TROWS;
    constexpr int TILE  = UNROLL * ROWS;

    extern __shared__ __align__(128) char smem_raw[];
    double* s_val      = (double*) smem_raw;
    local_int_t* s_col = (local_int_t*) (smem_raw + (size_t) 2 * TILE * sizeof(double));
    uint64_t* bar      = (uint64_t*) (smem_raw + (size_t) 2 * TILE * (sizeof(double) + sizeof(local_int_t)));

    __shared__ int s_entry_base, s_max_row_len;

    const int tid = threadIdx.x;
    const int cta_row_base = blockIdx.x * ROWS + color_str;
    int slice, in_slice_base;
    intdiv32_divmod(cta_row_base, slice_size, slice_size_div, &slice, &in_slice_base);

    if (IsWarpZero() && IsElectedLane())
    {
        ptx::mbarrier_init(&bar[0], 1);
        ptx::mbarrier_init(&bar[1], 1);
        ptx::fence_proxy_async(ptx::space_shared);
        const slice_ptr_t so = slice_offsets[slice];
        int eb, mrl;
        intdiv32_div((int32_t) so, slice_size_div, &eb);
        intdiv32_div((int32_t) (slice_offsets[slice + 1] - so), slice_size_div, &mrl);
        s_entry_base = eb;
        s_max_row_len = mrl;
    }
    __syncthreads();
    const int entry_base = s_entry_base;
    const int max_row_len = s_max_row_len;

    auto load = [&](int b, int kb)
    {
        if (IsWarpZero() && IsElectedLane())
        {
            if (kb < max_row_len)
            {
#pragma unroll
                for (int t = 0; t < NT; ++t)
                {
                    const int32_t coord[2] = {in_slice_base + t * TROWS, entry_base + kb};
                    double* vd = s_val + (size_t) b * TILE + (size_t) t * UNROLL * TROWS;
                    local_int_t* cd = s_col + (size_t) b * TILE + (size_t) t * UNROLL * TROWS;
                    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, vd, &val_map, coord, &bar[b]);
                    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, cd, &col_map, coord, &bar[b]);
                }
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[b],
                    static_cast<uint32_t>((size_t) TILE * (sizeof(double) + sizeof(local_int_t))));
            }
            else
            {
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[b], 0);
            }
        }
    };

    const int r0 = tid * RPT;
    const int mt = r0 / TROWS;
    const int rb = r0 - mt * TROWS;
    const size_t toff = (size_t) mt * UNROLL * TROWS;
    auto consume = [&](int b, int kb, double (&sum)[RPT])
    {
        const double* v = s_val + (size_t) b * TILE + toff;
        const local_int_t* c = s_col + (size_t) b * TILE + toff;
#pragma unroll
        for (int e = 0; e < UNROLL; ++e)
        {
            if (kb + e < max_row_len)
            {
                const double* ve = v + (size_t) e * TROWS;
                const local_int_t* ce = c + (size_t) e * TROWS;
                int cols[RPT];
                double bv[RPT];
#pragma unroll
                for (int m = 0; m < RPT; ++m)
                    cols[m] = ce[rb + m];
#pragma unroll
                for (int m = 0; m < RPT; ++m)
                    bv[m] = (cols[m] >= 0) ? x[cols[m]] : 0.0;
#pragma unroll
                for (int m = 0; m < RPT; ++m)
                    sum[m] += ve[rb + m] * bv[m];
            }
        }
    };

    double y_m[RPT], diag_m[RPT], sum[RPT];
#pragma unroll
    for (int m = 0; m < RPT; ++m)
        sum[m] = 0.0;
    LoadVecD<RPT>(&rhs[cta_row_base + tid * RPT], y_m);
    LoadVecD<RPT>(&diag[cta_row_base + tid * RPT], diag_m);

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

    double xo[RPT];
#pragma unroll
    for (int m = 0; m < RPT; ++m)
        xo[m] = (alpha * y_m[m] - sum[m]) / diag_m[m];
    StoreVecD<RPT>(&x[cta_row_base + tid * RPT], xo);
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

bool MakeMap(CUtensorMap* map, void* addr, CUtensorMapDataType dtype, size_t elem, uint64_t total_entries, int box_rows,
    int UNROLL, int slice_size)
{
    const uint64_t gdim[2] = {(uint64_t) slice_size, total_entries};
    const uint64_t gstr[1] = {(uint64_t) slice_size * elem};
    const uint32_t bdim[2] = {(uint32_t) box_rows, (uint32_t) UNROLL};
    const uint32_t estr[2] = {1, 1};
    const CUresult r = cuTensorMapEncodeTiled(map, dtype, 2, addr, gdim, gstr, bdim, estr,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (r != CUDA_SUCCESS)
    {
        const char* msg = nullptr;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "[spsv-tma-2d] cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "?");
        return false;
    }
    return true;
}

template <int BLKDIM, int UNROLL, int RPT>
bool LaunchColorLoop2d(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* slice_offsets,
    local_int_t* columns, double* values, slice_ptr_t last_nnz, cudaStream_t stream)
{
    constexpr int ROWS  = BLKDIM * RPT;
    constexpr int TROWS = (ROWS <= 256) ? ROWS : 256;
    constexpr int TILE  = UNROLL * ROWS;
    constexpr size_t smem = (size_t) 2 * TILE * (sizeof(double) + sizeof(local_int_t)) + 2 * sizeof(uint64_t);

    const local_int_t rows = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;
    if (A.totalColors <= 0 || rows % A.totalColors != 0)
        return false;
    const local_int_t color_size = rows / A.totalColors;
    if (ROWS % TROWS != 0 || ROWS > slice_size || slice_size % ROWS != 0 || color_size % ROWS != 0
        || (int) smem > MaxOptinSmem())
        return false;
    if ((last_nnz % slice_size) != 0)
        return false;

    CUtensorMap val_map, col_map;
    const uint64_t total_entries = (uint64_t) (last_nnz / slice_size);
    if (!MakeMap(&val_map, (void*) values, CU_TENSOR_MAP_DATA_TYPE_FLOAT64, sizeof(double), total_entries, TROWS, UNROLL,
            slice_size))
        return false;
    if (!MakeMap(&col_map, (void*) columns, CU_TENSOR_MAP_DATA_TYPE_INT32, sizeof(local_int_t), total_entries, TROWS,
            UNROLL, slice_size))
        return false;

    auto kernel = spsv_sell_tma2d<BLKDIM, UNROLL, RPT>;
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr_set = true;
    }

    const intdiv32_t sdiv = intdiv32_gen(slice_size);
    const local_int_t grid = color_size / ROWS;
    const double alpha = 1.0;
    if (forward)
    {
        for (int color = 0; color < A.totalColors; ++color)
        {
            const int cs = color * color_size;
            const int ce = (int) std::min<local_int_t>((color + 1) * (local_int_t) color_size, rows);
            kernel<<<grid, BLKDIM, smem, stream>>>(
                cs, ce, xv, rv, A.diagonal, slice_offsets, alpha, slice_size, sdiv, val_map, col_map);
        }
    }
    else
    {
        for (int color = A.totalColors - 1; color >= 0; --color)
        {
            const int cs = color * color_size;
            const int ce = (int) std::min<local_int_t>((color + 1) * (local_int_t) color_size, rows);
            kernel<<<grid, BLKDIM, smem, stream>>>(
                cs, ce, xv, rv, A.diagonal, slice_offsets, alpha, slice_size, sdiv, val_map, col_map);
        }
    }
    return true;
}

}

bool SpsvTma2dSellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, slice_ptr_t last_nnz, cudaStream_t stream, int blk, int unroll,
    int rpt)
{
#define TRY(B, U, R)                                                                                                   \
    if (blk == (B) && unroll == (U) && rpt == (R))                                                                     \
        return LaunchColorLoop2d<B, U, R>(                                                                             \
            forward, A, rv, xv, sell_block_offset, sell_columns, sell_values, last_nnz, stream);
#define ROW(B, U) TRY(B, U, 1) TRY(B, U, 2) TRY(B, U, 4)
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

#else // INDEX_64

bool SpsvTma2dSellCfg(int, const SparseMatrix&, double*, double*, slice_ptr_t*, local_int_t*, double*, slice_ptr_t,
    cudaStream_t, int, int, int)
{
    return false;
}

#endif // !INDEX_64
#endif
