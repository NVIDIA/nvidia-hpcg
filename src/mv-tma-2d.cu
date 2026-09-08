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
 @file mv-tma-2d.cu

 TMA2D SpMV: the TMA family's staging through shared memory, driven by a tensor
 map instead of by an address the kernel computes.

 The Sliced-ELL columns and values of one slice are a dense
 slice_size-by-max_row_len tile, so the whole operator is a two-dimensional
 tensor whose fast axis is the row within a slice and whose slow axis is the
 stored-entry index. Describing it once on the host, as a CUtensorMap, replaces
 the per-k address arithmetic of the 1-D family with a pair of tile coordinates:
 the CTA's row offset inside its slice, and the entry index its slice starts at.
 The copy engine then moves a whole UNROLL-deep box per instruction rather than
 one k at a time, and out-of-range reads past the last stored entry are
 zero-filled by the hardware instead of being skipped by the issuing lane.

 The box's fast axis is capped at 256 rows, which is the tensor-map limit, so a
 row block wider than that is covered by NT boxes side by side; TROWS and NT
 carry that split.

 One CTA owns a contiguous block of ROWS = BLKDIM * RPT rows and the block is
 required to sit inside a single slice, so its coordinates resolve once, in one
 elected lane, and are broadcast through shared memory. Two shared buffers are
 double-buffered against each other over an mbarrier pair, as in the 1-D family.

 Unlike the 1-D family, a thread owns RPT *consecutive* rows rather than RPT
 rows a block size apart, which is what lets the y read-back and the y store be
 single vector accesses.

 Columns are read as 32-bit values and the column tensor map is encoded as
 INT32, so this family takes 32-bit columns (idx32_t), which is what the
 index-mode dispatch already enforces. Instantiated for both slice-offset
 widths, like the rest of the explicit path.

 The bulk tensor copy and its mbarrier arrival are compute capability 9.0 and
 later instructions. This tree builds one binary for sm_80, sm_90 and sm_100, so
 the device code below is compiled only in the 9.0+ passes and the launcher
 refuses the family outright on an older device.
 */

#ifdef USE_CUDA
#ifdef EXPLICIT_KERNELS

#include "CudaKernels.hpp"
#include "IndexMode.hpp"
#include "SparseMatrix.hpp"
#include "intdiv.hh"

#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda/ptx>
#include <cuda_runtime.h>

// cp.async.bulk.tensor (SASS UTMALDG) and the mbarrier transaction-count
// instructions it arrives on exist only from compute capability 9.0. libcu++'s
// cuda::ptx wrappers do not degrade below that -- they resolve to a call to an
// undefined __cuda_ptx_*_is_not_supported_before_SM_90__ symbol, which ptxas
// reports as a fatal unresolved extern rather than a warning. The tree these
// kernels come from builds only for sm_90 and up and so needed no guard at all;
// this one must also produce an sm_80 image.
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

// double4 has no CUDA type: the built-in vector types stop at two doubles, so
// the 32-byte access a thread's four consecutive rows allow has to be spelled
// out. The alignment is what makes it one instruction rather than four.
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

template <class OffsetT, int BLKDIM, int UNROLL, int RPT>
__global__ __launch_bounds__(BLKDIM) void mv_sell_tma2d(int m, double alpha, double beta, double* __restrict__ y,
    const double* __restrict__ x, const OffsetT* __restrict__ slice_offsets, int slice_size,
    intdiv32_t slice_size_div, const __grid_constant__ CUtensorMap val_map, const __grid_constant__ CUtensorMap col_map)
{
#if HPCG_TMA_BULK
    constexpr int ROWS = BLKDIM * RPT;
    constexpr int TROWS = (ROWS <= 256) ? ROWS : 256;
    constexpr int NT = ROWS / TROWS;
    constexpr int TILE = UNROLL * ROWS;

    extern __shared__ __align__(128) char smem_raw[];
    double* s_val = (double*) smem_raw;
    idx32_t* s_col = (idx32_t*) (smem_raw + (size_t) 2 * TILE * sizeof(double));
    uint64_t* bar = (uint64_t*) (smem_raw + (size_t) 2 * TILE * (sizeof(double) + sizeof(idx32_t)));

    __shared__ int s_entry_base, s_max_row_len;

    const int tid = threadIdx.x;
    const int cta_row_base = blockIdx.x * ROWS;
    int slice, in_slice_base;
    intdiv32_divmod(cta_row_base, slice_size, slice_size_div, &slice, &in_slice_base);

    if (IsWarpZero() && IsElectedLane())
    {
        ptx::mbarrier_init(&bar[0], 1);
        ptx::mbarrier_init(&bar[1], 1);
        ptx::fence_proxy_async(ptx::space_shared);
        const OffsetT so = slice_offsets[slice];
        // A slice offset is bounded by the operator's padded nonzero count, and
        // --mi 1 exists precisely for problems where that count does not fit in
        // int32 -- 512^3 reaches about 3.6e9 -- so narrowing it to feed the
        // 32-bit reciprocal would wrap and name a different part of the matrix,
        // silently and with the right-looking magnitude. A true 64-bit divide is
        // affordable exactly here and nowhere else in this kernel: one elected
        // lane runs it once per CTA, before the k loop, so it is off the
        // per-thread critical path entirely and its latency is paid while the
        // rest of the CTA is still at the __syncthreads below.
        int eb;
        if constexpr (sizeof(OffsetT) == 8)
            eb = (int) (so / (FlatOffsetT<OffsetT>) slice_size);
        else
            intdiv32_div((int32_t) so, slice_size_div, &eb);
        // The difference, unlike the offset itself, is one slice's nnz, bounded
        // by slice_size * HPCG_MAX_ROW_LEN whatever the problem size, so it is an
        // int32 on both paths and keeps the magic reciprocal.
        int mrl;
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
                    idx32_t* cd = s_col + (size_t) b * TILE + (size_t) t * UNROLL * TROWS;
                    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, vd, &val_map, coord, &bar[b]);
                    ptx::cp_async_bulk_tensor(ptx::space_shared, ptx::space_global, cd, &col_map, coord, &bar[b]);
                }
                // The whole box arrives whether or not its trailing k columns
                // are past the slice's last stored entry -- the tensor map
                // zero-fills those rather than dropping them -- so the expected
                // transaction count is the full tile and does not depend on
                // max_row_len the way the 1-D family's per-k issue does.
                ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, &bar[b],
                    static_cast<uint32_t>((size_t) TILE * (sizeof(double) + sizeof(idx32_t))));
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
        const idx32_t* c = s_col + (size_t) b * TILE + toff;
#pragma unroll
        for (int e = 0; e < UNROLL; ++e)
        {
            if (kb + e < max_row_len)
            {
                const double* ve = v + (size_t) e * TROWS;
                const idx32_t* ce = c + (size_t) e * TROWS;
                idx32_t cols[RPT];
                double bv[RPT];
#pragma unroll
                for (int mm = 0; mm < RPT; ++mm)
                    cols[mm] = ce[rb + mm];
#pragma unroll
                for (int mm = 0; mm < RPT; ++mm)
                    bv[mm] = (cols[mm] >= 0) ? x[cols[mm]] : 0.0;
#pragma unroll
                for (int mm = 0; mm < RPT; ++mm)
                    sum[mm] += ve[rb + mm] * bv[mm];
            }
        }
    };

    const int row_base = cta_row_base + tid * RPT;
    const bool has_beta = (beta != 0.0);
    double y_m[RPT], sum[RPT];
#pragma unroll
    for (int mm = 0; mm < RPT; ++mm)
        sum[mm] = 0.0;
    // Read y before the k loop rather than at the store: the CTA crosses the
    // loop's __syncthreads calls and a row's own y does not change across them,
    // so the read costs nothing extra here and does not sit between the last
    // multiply and the store.
    if (has_beta)
        LoadVecD<RPT>(&y[row_base], y_m);

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

    // No row guard: the launcher requires m % ROWS == 0, so every thread's RPT
    // rows are live and the store is one vector access.
    double yo[RPT];
#pragma unroll
    for (int mm = 0; mm < RPT; ++mm)
        yo[mm] = has_beta ? (beta * y_m[mm] + alpha * sum[mm]) : (alpha * sum[mm]);
    StoreVecD<RPT>(&y[row_base], yo);
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

/*
  Describe one of the two Sliced-ELL arrays as a slice_size-by-total_entries
  tensor tiled into box_rows-by-UNROLL boxes.

  An encode that the driver rejects is reported and refused rather than
  ignored: the kernel takes the map by value, so a map that was never written
  would be launched as uninitialised bytes.
*/
bool MakeMap(CUtensorMap* map, void* addr, CUtensorMapDataType dtype, size_t elem, uint64_t total_entries,
    int box_rows, int UNROLL, int slice_size)
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
        fprintf(stderr, "[mv-tma-2d] cuTensorMapEncodeTiled failed: %s\n", msg ? msg : "?");
        return false;
    }
    return true;
}

template <class OffsetT, int BLKDIM, int UNROLL, int RPT>
bool LaunchMvTma2d(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const OffsetT* slice_offsets, const idx32_t* columns, const double* values, slice_ptr_t last_nnz,
    cudaStream_t stream)
{
    constexpr int ROWS = BLKDIM * RPT;
    constexpr int TROWS = (ROWS <= 256) ? ROWS : 256;
    constexpr int TILE = UNROLL * ROWS;
    constexpr size_t smem = (size_t) 2 * TILE * (sizeof(double) + sizeof(idx32_t)) + 2 * sizeof(uint64_t);

    const local_int_t m = A.localNumberOfRows;
    const int slice_size = (int) A.slice_size;
    if (!DeviceHasBulkCopy())
        return false;
    // The CTA's whole row block has to lie inside one slice: the kernel resolves
    // its two tile coordinates once, from its first row, and every box it then
    // fetches is measured from them. Together with a padded nonzero count that
    // is a whole number of slices, these are also what makes the tensor
    // description legal -- the map's global stride is slice_size elements, so a
    // trailing partial slice would put rows of one slice inside another's tile,
    // and a box origin that is not a multiple of the box extent would straddle
    // two tiles.
    if (ROWS % TROWS != 0 || ROWS > slice_size || slice_size % ROWS != 0 || m % ROWS != 0
        || (int) smem > MaxOptinSmem() || (last_nnz % slice_size) != 0)
        return false;

    CUtensorMap val_map, col_map;
    const uint64_t total_entries = (uint64_t) (last_nnz / slice_size);
    // The entry index reaches the kernel as a TMA tile coordinate, which is
    // int32. --mi 1 raises the padded nonzero count past that, so the bound has
    // to be checked rather than assumed; it is the count divided by slice_size
    // here, which keeps every problem this benchmark runs comfortably inside the
    // range and refuses only an operator whose tiles genuinely cannot be
    // addressed.
    if (total_entries > (uint64_t) INT32_MAX)
        return false;
    if (!MakeMap(&val_map, const_cast<double*>(values), CU_TENSOR_MAP_DATA_TYPE_FLOAT64, sizeof(double), total_entries,
            TROWS, UNROLL, slice_size))
        return false;
    // INT32 with a stride derived from sizeof(idx32_t): the data type and the
    // stride agree by construction, not by luck. This tree's explicit kernels
    // take 32-bit columns only, and ExplicitIndexModeUsable in CudaKernels.cu
    // refuses --mi 2 (IndexMode::I64_I64) before any of them is reached. That
    // guarantee has to come from the dispatch, because a mismatch here does not
    // fail: the encode succeeds and the copies then read the wrong halves of the
    // wrong elements.
    if (!MakeMap(&col_map, const_cast<idx32_t*>(columns), CU_TENSOR_MAP_DATA_TYPE_INT32, sizeof(idx32_t),
            total_entries, TROWS, UNROLL, slice_size))
        return false;

    auto kernel = mv_sell_tma2d<OffsetT, BLKDIM, UNROLL, RPT>;
    static bool attr_set = false;
    if (!attr_set)
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr_set = true;
    }

    const intdiv32_t sdiv = intdiv32_gen(slice_size);
    const local_int_t grid = m / ROWS;
    kernel<<<grid, BLKDIM, smem, stream>>>(
        (int) m, alpha, beta, y, x, slice_offsets, slice_size, sdiv, val_map, col_map);
    return true;
}

} // namespace

// rpt is rows per thread, the same knob MV_W carries for the register families;
// here it also sets the box extent along the slow axis of the tensor, since a
// CTA stages BLKDIM * rpt rows per k. It stops at 4 rather than the 1-D
// family's 8 because a thread's rows are consecutive and are loaded and stored
// as one vector access, and 4 doubles is the widest one the hardware has.
//
// last_nnz is the operator's padded nonzero count, which the 1-D family does not
// need: the tensor map's global dimensions describe the whole array, so its
// extent has to be named on the host rather than derived per CTA.
template <class OffsetT>
bool MvTma2dSellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const OffsetT* slice_offsets, const idx32_t* columns, const double* values, slice_ptr_t last_nnz,
    cudaStream_t stream, int blk, int unroll, int rpt)
{
#define HPCG_MV_TMA2D_CASE(B, U, R)                                                                                    \
    if (blk == (B) && unroll == (U) && rpt == (R))                                                                     \
        return LaunchMvTma2d<OffsetT, B, U, R>(                                                                        \
            A, alpha, beta, x, y, slice_offsets, columns, values, last_nnz, stream);
#define HPCG_MV_TMA2D_ROW(B, U) HPCG_MV_TMA2D_CASE(B, U, 1) HPCG_MV_TMA2D_CASE(B, U, 2) HPCG_MV_TMA2D_CASE(B, U, 4)
#define HPCG_MV_TMA2D_BLK(B)                                                                                           \
    HPCG_MV_TMA2D_ROW(B, 1)                                                                                            \
    HPCG_MV_TMA2D_ROW(B, 2)                                                                                            \
    HPCG_MV_TMA2D_ROW(B, 3) HPCG_MV_TMA2D_ROW(B, 4) HPCG_MV_TMA2D_ROW(B, 6) HPCG_MV_TMA2D_ROW(B, 8)
    HPCG_MV_TMA2D_BLK(32)
    HPCG_MV_TMA2D_BLK(64)
    HPCG_MV_TMA2D_BLK(128)
    HPCG_MV_TMA2D_BLK(256)
#undef HPCG_MV_TMA2D_BLK
#undef HPCG_MV_TMA2D_ROW
#undef HPCG_MV_TMA2D_CASE
    return false;
}

// Both slice-offset widths, 32-bit columns: --mi 0 and --mi 1 respectively. The
// only reads of slice_offsets are the two scalar loads the elected lane makes to
// resolve the CTA's tile coordinates, so this is an instantiation rather than a
// variant -- the tensor maps themselves are width-agnostic, since the arrays
// they describe are not the offset array.
template bool MvTma2dSellCfg<idx32_t>(const SparseMatrix&, double, double, const double*, double*, const idx32_t*,
    const idx32_t*, const double*, slice_ptr_t, cudaStream_t, int, int, int);
template bool MvTma2dSellCfg<idx64_t>(const SparseMatrix&, double, double, const double*, double*, const idx64_t*,
    const idx32_t*, const double*, slice_ptr_t, cudaStream_t, int, int, int);

#endif // EXPLICIT_KERNELS
#endif // USE_CUDA
