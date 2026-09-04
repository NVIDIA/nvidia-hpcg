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
 @file spsv-tma.cu

 TMA SymGS triangular solve: LDG's per-colour launch, with the matrix stream
 staged through shared memory by the bulk asynchronous copy engine rather than
 pulled into registers by the consuming threads.

 One CTA owns a contiguous block of ROWS = BLKDIM * RPT rows of one colour, and
 the block is required to sit inside a single slice, so the k-th stored entry of
 every row in the block is one contiguous run of ROWS elements in the
 Sliced-ELL arrays. That is exactly the shape cp.async.bulk moves, so a single
 elected lane issues two bulk copies per k -- columns and values -- and the
 other warps do no address arithmetic and hold no load in flight.

 Two shared buffers are double-buffered against each other over an mbarrier
 pair: while one is being consumed the other is filling. The issuing lane
 arrives on the barrier with the transaction count it just committed, and the
 whole CTA waits on the parity flip, so the copies are ordered by the barrier
 rather than by a __syncthreads on the loading warp.

 rhs and diag are read once into registers before the k loop, because the CTA
 crosses that loop's __syncthreads calls and a row's own two scalars do not
 change across it.

 Columns are read from shared memory as 32-bit values, so this family takes
 32-bit columns (idx32_t), which is what the index-mode dispatch already
 enforces. Instantiated for both slice-offset widths, like the rest of the
 explicit path.

 The bulk copy and its mbarrier arrival are compute capability 9.0 and later
 instructions. This tree builds one binary for sm_80, sm_90 and sm_100, so the
 device code below is compiled only in the 9.0+ passes and the launcher refuses
 the family outright on an older device.
 */

#ifdef USE_CUDA
#ifdef EXPLICIT_KERNELS

#include "CudaKernels.hpp"
#include "IndexMode.hpp"
#include "SparseMatrix.hpp"
#include "intdiv.hh"

#include <algorithm>
#include <cstdint>
#include <cuda/ptx>
#include <cuda_runtime.h>

// cp.async.bulk (SASS UBLKCP) and the mbarrier transaction-count instructions
// it arrives on exist only from compute capability 9.0. libcu++'s cuda::ptx
// wrappers do not degrade below that -- they resolve to a call to an undefined
// __cuda_ptx_*_is_not_supported_before_SM_90__ symbol, which ptxas reports as a
// fatal unresolved extern rather than a warning. The tree these kernels come
// from builds only for sm_90 and up and so needed no guard at all; this one
// must also produce an sm_80 image.
//
// The guard elides the device bodies rather than the kernel declarations, so
// every architecture in the fatbinary still carries a real entry point and
// nothing is left unresolved. The sm_80 entry point is never launched: the
// launcher checks the running device's compute capability and refuses, which
// the caller reports through its existing "no explicit kernel" path.
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 900)
#define HPCG_TMA_BULK 1
#else
#define HPCG_TMA_BULK 0
#endif

namespace
{

namespace ptx = cuda::ptx;

#if HPCG_TMA_BULK

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

#endif // HPCG_TMA_BULK

template <class OffsetT, int BLKDIM, int UNROLL, int RPT>
__global__ __launch_bounds__(BLKDIM) void spsv_sell_tma_color(int color_str, int color_end, double* __restrict__ x,
    const double* __restrict__ rhs, const OffsetT* __restrict__ slice_offsets, const idx32_t* __restrict__ col_idx,
    const double* __restrict__ values, const double* __restrict__ diag, double alpha, int slice_size,
    intdiv32_t slice_size_div)
{
#if HPCG_TMA_BULK
    constexpr int ROWS = BLKDIM * RPT;

    extern __shared__ __align__(16) char smem_raw[];
    idx32_t* s_col = reinterpret_cast<idx32_t*>(smem_raw);
    double* s_val = reinterpret_cast<double*>(smem_raw + (size_t) 2 * UNROLL * ROWS * sizeof(idx32_t));
    uint64_t* bar
        = reinterpret_cast<uint64_t*>(smem_raw + (size_t) 2 * UNROLL * ROWS * (sizeof(idx32_t) + sizeof(double)));

    auto scol = [&](int buf, int e) -> idx32_t* { return s_col + ((size_t) buf * UNROLL + e) * ROWS; };
    auto sval = [&](int buf, int e) -> double* { return s_val + ((size_t) buf * UNROLL + e) * ROWS; };

    const int tid = threadIdx.x;
    const int cta_row_base = blockIdx.x * ROWS + color_str;

    int slice, in_slice_base;
    intdiv32_divmod(cta_row_base, slice_size, slice_size_div, &slice, &in_slice_base);
    // Flat element offsets into the column/value arrays exceed 2^31 for large
    // local problems, so widen before they enter the pointer arithmetic even
    // when OffsetT itself is 32-bit.
    // 64-bit only when the slice offsets are; see FlatOffsetT. The source tree
    // reads 64-bit offsets unconditionally and so computes this in size_t, but
    // this instantiation may be reading 32-bit ones, and widening those costs a
    // sign-extension and the registers to hold the result.
    const FlatOffsetT<OffsetT> block_base
        = (FlatOffsetT<OffsetT>) slice_offsets[slice] + (FlatOffsetT<OffsetT>) in_slice_base;
    // Per-slice nnz is bounded by slice_size * HPCG_MAX_ROW_LEN and so fits in
    // int32. Narrowing the difference before dividing keeps this in the 32-bit
    // reciprocal above instead of the 64-bit form a wider OffsetT would force.
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
                    // The transfer sizes describe the column and value arrays,
                    // not the slice-offset array, so they stay 4 and 8 bytes per
                    // element whatever OffsetT is: slice_offsets is read here as
                    // two scalar loads above and never moved by the copy engine.
                    ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, scol(buf, e), col_idx + off,
                        static_cast<uint32_t>(ROWS * sizeof(idx32_t)), &bar[buf]);
                    ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, sval(buf, e), values + off,
                        static_cast<uint32_t>(ROWS * sizeof(double)), &bar[buf]);
                    ++valid;
                }
            }
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[buf],
                static_cast<uint32_t>(valid * ROWS * (sizeof(idx32_t) + sizeof(double))));
        }
    };

    auto consume = [&](int buf, int kb, double (&sum)[RPT])
    {
#pragma unroll
        for (int e = 0; e < UNROLL; ++e)
        {
            if (kb + e < max_row_len)
            {
                const idx32_t* c = scol(buf, e);
                const double* v = sval(buf, e);
                idx32_t cols[RPT];
                double b[RPT];
#pragma unroll
                for (int m = 0; m < RPT; ++m)
                    cols[m] = c[tid + m * BLKDIM];
#pragma unroll
                for (int m = 0; m < RPT; ++m)
                    b[m] = (cols[m] >= 0) ? x[cols[m]] : 0.0;
#pragma unroll
                for (int m = 0; m < RPT; ++m)
                    sum[m] += v[tid + m * BLKDIM] * b[m];
            }
        }
    };

    double rhs_m[RPT], diag_m[RPT], sum[RPT];
#pragma unroll
    for (int m = 0; m < RPT; ++m)
    {
        const int row = cta_row_base + tid + m * BLKDIM;
        const bool live = row < color_end;
        rhs_m[m] = live ? rhs[row] : 0.0;
        diag_m[m] = live ? diag[row] : 1.0;
        sum[m] = 0.0;
    }

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
    for (int m = 0; m < RPT; ++m)
    {
        const int row = cta_row_base + tid + m * BLKDIM;
        if (row < color_end)
            x[row] = (alpha * rhs_m[m] - sum[m]) / diag_m[m];
    }
#endif // HPCG_TMA_BULK
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

// The device bodies above exist only in the 9.0+ passes, so on an older device
// the kernel would launch, do nothing and leave x untouched -- a wrong answer
// rather than a failure. Refusing here turns that into the caller's existing
// "no explicit kernel for this configuration" report.
bool DeviceHasBulkCopy()
{
    static bool v = [] {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 9;
    }();
    return v;
}

template <class OffsetT, int BLKDIM, int UNROLL, int RPT>
bool LaunchSpsvTma(bool forward, const SparseMatrix& A, const double* rv, double* xv, const OffsetT* slice_offsets,
    const idx32_t* columns, const double* values, cudaStream_t stream)
{
    constexpr int ROWS = BLKDIM * RPT;
    const local_int_t rows = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;

    if (!DeviceHasBulkCopy())
        return false;

    if (A.totalColors <= 0 || rows % A.totalColors != 0)
        return false;
    const local_int_t color_size = rows / A.totalColors;

    // The CTA's whole row block has to lie inside one slice: the kernel resolves
    // slice and in-slice base once, from its first row, and then treats each k
    // as one contiguous run of ROWS elements. These conditions are also what
    // makes every bulk copy legally aligned. cp.async.bulk needs 16 bytes on
    // both ends and a size that is a multiple of 16; ROWS is at least 32, so
    // ROWS * 4 and ROWS * 8 are multiples of 128, and because slice_offsets
    // entries are whole multiples of slice_size and in_slice_base is a multiple
    // of ROWS, every source offset is a multiple of ROWS elements too.
    if (ROWS > slice_size || slice_size % ROWS != 0 || color_size % ROWS != 0)
        return false;

    constexpr size_t smem = (size_t) 2 * UNROLL * ROWS * (sizeof(idx32_t) + sizeof(double)) + 2 * sizeof(uint64_t);
    if ((int) smem > MaxOptinSmem())
        return false;

    auto kernel = spsv_sell_tma_color<OffsetT, BLKDIM, UNROLL, RPT>;
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr_set = true;
    }

    const local_int_t grid = color_size / ROWS;
    const double alpha = 1.0;
    const intdiv32_t sdiv = intdiv32_gen(slice_size);

    if (forward)
    {
        for (int color = 0; color < A.totalColors; ++color)
        {
            const int cs = color * color_size;
            const int ce = (int) std::min<local_int_t>((color + 1) * (local_int_t) color_size, rows);
            kernel<<<grid, BLKDIM, smem, stream>>>(
                cs, ce, xv, rv, slice_offsets, columns, values, A.diagonal, alpha, slice_size, sdiv);
        }
    }
    else
    {
        for (int color = A.totalColors - 1; color >= 0; --color)
        {
            const int cs = color * color_size;
            const int ce = (int) std::min<local_int_t>((color + 1) * (local_int_t) color_size, rows);
            kernel<<<grid, BLKDIM, smem, stream>>>(
                cs, ce, xv, rv, slice_offsets, columns, values, A.diagonal, alpha, slice_size, sdiv);
        }
    }
    return true;
}

} // namespace

// rpt is rows per thread, the same knob SV_W carries for the register families;
// here it also sets the bulk transfer length, since a CTA stages BLKDIM * rpt
// elements per k. The solve reaches one row block further than the SpMV does --
// BLKDIM up to 512 and rpt up to 16 -- because a colour is a fraction of the
// rows and the block still has to divide it.
template <class OffsetT>
bool SpsvTmaSellCfg(bool forward, const SparseMatrix& A, const double* rv, double* xv, const OffsetT* slice_offsets,
    const idx32_t* columns, const double* values, cudaStream_t stream, int blk, int unroll, int rpt)
{
#define HPCG_SV_TMA_CASE(B, U, R)                                                                                      \
    if (blk == (B) && unroll == (U) && rpt == (R))                                                                     \
        return LaunchSpsvTma<OffsetT, B, U, R>(forward, A, rv, xv, slice_offsets, columns, values, stream);
#define HPCG_SV_TMA_ROW(B, U)                                                                                          \
    HPCG_SV_TMA_CASE(B, U, 1)                                                                                          \
    HPCG_SV_TMA_CASE(B, U, 2) HPCG_SV_TMA_CASE(B, U, 4) HPCG_SV_TMA_CASE(B, U, 8) HPCG_SV_TMA_CASE(B, U, 16)
#define HPCG_SV_TMA_BLK(B)                                                                                             \
    HPCG_SV_TMA_ROW(B, 1)                                                                                              \
    HPCG_SV_TMA_ROW(B, 2)                                                                                              \
    HPCG_SV_TMA_ROW(B, 3) HPCG_SV_TMA_ROW(B, 4) HPCG_SV_TMA_ROW(B, 6) HPCG_SV_TMA_ROW(B, 8)
    HPCG_SV_TMA_BLK(32)
    HPCG_SV_TMA_BLK(64)
    HPCG_SV_TMA_BLK(128)
    HPCG_SV_TMA_BLK(256)
    HPCG_SV_TMA_BLK(512)
#undef HPCG_SV_TMA_BLK
#undef HPCG_SV_TMA_ROW
#undef HPCG_SV_TMA_CASE

    return false;
}

// Both slice-offset widths, 32-bit columns: --mi 0 and --mi 1 respectively.
template bool SpsvTmaSellCfg<idx32_t>(bool, const SparseMatrix&, const double*, double*, const idx32_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int);
template bool SpsvTmaSellCfg<idx64_t>(bool, const SparseMatrix&, const double*, double*, const idx64_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int);

#endif // EXPLICIT_KERNELS
#endif // USE_CUDA
