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
 @file mv-tma.cu

 TMA SpMV: the matrix stream is staged through shared memory by the bulk
 asynchronous copy engine rather than pulled into registers by the consuming
 threads.

 One CTA owns a contiguous block of ROWS = BLKDIM * RPT rows, and the block is
 required to sit inside a single slice, so the k-th stored entry of every row
 in the block is one contiguous run of ROWS elements in the Sliced-ELL arrays.
 That is exactly the shape cp.async.bulk moves, so a single elected lane issues
 two bulk copies per k -- columns and values -- and the other warps do no
 address arithmetic and hold no load in flight.

 Two shared buffers are double-buffered against each other over an mbarrier
 pair: while one is being consumed the other is filling. The issuing lane
 arrives on the barrier with the transaction count it just committed, and the
 whole CTA waits on the parity flip, so the copies are ordered by the barrier
 rather than by a __syncthreads on the loading warp.

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
__global__ __launch_bounds__(BLKDIM) void mv_sell_tma(int m, double alpha, double beta, double* __restrict__ y,
    const double* __restrict__ x, const OffsetT* __restrict__ slice_offsets, const idx32_t* __restrict__ col_idx,
    const double* __restrict__ values, int slice_size, intdiv32_t slice_size_div)
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
    const int cta_row_base = blockIdx.x * ROWS;

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
// the kernel would launch, do nothing and leave y untouched -- a wrong answer
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
bool LaunchMvTma(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const OffsetT* slice_offsets, const idx32_t* columns, const double* values, cudaStream_t stream)
{
    constexpr int ROWS = BLKDIM * RPT;
    const local_int_t m = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;
    if (!DeviceHasBulkCopy())
        return false;
    // The CTA's whole row block has to lie inside one slice: the kernel resolves
    // slice and in-slice base once, from its first row, and then treats each k
    // as one contiguous run of ROWS elements. These three conditions are also
    // what makes every bulk copy legally aligned. cp.async.bulk needs 16 bytes
    // on both ends and a size that is a multiple of 16; ROWS is at least 32, so
    // ROWS * 4 and ROWS * 8 are multiples of 128, and because slice_offsets
    // entries are whole multiples of slice_size and in_slice_base is a multiple
    // of ROWS, every source offset is a multiple of ROWS elements too.
    if (ROWS > slice_size || slice_size % ROWS != 0 || m % ROWS != 0)
        return false;

    constexpr size_t smem = (size_t) 2 * UNROLL * ROWS * (sizeof(idx32_t) + sizeof(double)) + 2 * sizeof(uint64_t);
    if ((int) smem > MaxOptinSmem())
        return false;

    auto kernel = mv_sell_tma<OffsetT, BLKDIM, UNROLL, RPT>;
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

} // namespace

// rpt is rows per thread, the same knob MV_W carries for the register families;
// here it also sets the bulk transfer length, since a CTA stages BLKDIM * rpt
// elements per k. There is no partition knob and no cache-policy knob: the grid
// is one CTA per row block by construction, and the matrix stream never passes
// through the threads' own load path.
template <class OffsetT>
bool MvTmaSellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const OffsetT* slice_offsets, const idx32_t* columns, const double* values, cudaStream_t stream, int blk,
    int unroll, int rpt)
{
#define HPCG_MV_TMA_CASE(B, U, R)                                                                                      \
    if (blk == (B) && unroll == (U) && rpt == (R))                                                                     \
        return LaunchMvTma<OffsetT, B, U, R>(A, alpha, beta, x, y, slice_offsets, columns, values, stream);
#define HPCG_MV_TMA_ROW(B, U)                                                                                          \
    HPCG_MV_TMA_CASE(B, U, 1) HPCG_MV_TMA_CASE(B, U, 2) HPCG_MV_TMA_CASE(B, U, 4) HPCG_MV_TMA_CASE(B, U, 8)
#define HPCG_MV_TMA_BLK(B)                                                                                             \
    HPCG_MV_TMA_ROW(B, 1)                                                                                              \
    HPCG_MV_TMA_ROW(B, 2)                                                                                              \
    HPCG_MV_TMA_ROW(B, 3) HPCG_MV_TMA_ROW(B, 4) HPCG_MV_TMA_ROW(B, 6) HPCG_MV_TMA_ROW(B, 8)
    HPCG_MV_TMA_BLK(32)
    HPCG_MV_TMA_BLK(64)
    HPCG_MV_TMA_BLK(128)
    HPCG_MV_TMA_BLK(256)
#undef HPCG_MV_TMA_BLK
#undef HPCG_MV_TMA_ROW
#undef HPCG_MV_TMA_CASE
    return false;
}

// Both slice-offset widths, 32-bit columns: --mi 0 and --mi 1 respectively. The
// kernel body is width-agnostic -- the only reads of slice_offsets are the two
// scalar loads that resolve the CTA's row block, and both are widened to size_t
// before they reach the pointer arithmetic -- so this is an instantiation, not a
// variant.
template bool MvTmaSellCfg<idx32_t>(const SparseMatrix&, double, double, const double*, double*, const idx32_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int);
template bool MvTmaSellCfg<idx64_t>(const SparseMatrix&, double, double, const double*, double*, const idx64_t*,
    const idx32_t*, const double*, cudaStream_t, int, int, int);

#endif // EXPLICIT_KERNELS
#endif // USE_CUDA
