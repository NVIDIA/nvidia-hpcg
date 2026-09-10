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

#pragma once
#ifdef USE_CUDA
#include <cstdio>

#include "SparseMatrix.hpp"

// Kernel configuration structure
struct KernelConfig {
    int SV_UNROLL;
    int MV_UNROLL;
    int SV_BLOCK_SIZE;
    int MV_BLOCK_SIZE;
    int VECTOR_WIDTH;
    bool USE_TMA_MV;
    bool USE_TMA_SV;
};

// Global configuration instance
extern KernelConfig g_config;

// Initialize configuration from environment variables
void InitKernelConfig();

enum SvKernelKind
{
    SV_KIND_LDG = 0,
    SV_KIND_TMA = 1,
    SV_KIND_LDGV2 = 2,
    SV_KIND_TMA2D = 3,
    SV_KIND_LDGV3 = 4,
    // The explicit-HPCG / cuSPARSE TMA lineage, distinct from SV_KIND_TMA2D:
    // one row per thread, static shared buffers, 8-way rows for MV. It needs
    // the colour stride (SV) or the row partition (MV) to be a whole number of
    // slices, which the coarsest grid is not, so it is n/a at the last level.
    SV_KIND_TMAEX = 5,
};
// Number of kernel families above; autotune sizes its per-family arrays with it,
// and both kname tables there must have this many entries.
//
// CudaKernels.cu repeats this enum rather than including this header, because it
// defines KernelConfig, g_config and DIR of its own. Adding a family here means
// adding it there as well.
constexpr int kNumSvKernelKinds = 6;

// For LDG_V2 and LDG3 the rpt slot carries W, the rows each thread owns. LDG3
// packs two further choices into the same integer. Both describe how W is
// fetched rather than naming a different kernel, so they belong on this axis
// rather than in families of their own, and HPCG_FORCE_KIND=4 searches all of
// them:
//
//   sign   negative asks for the widest load the hardware offers -- int4
//          columns and, on Blackwell and later, 256-bit values -- where the
//          positive form uses 128-bit accesses.
//
//   +100   asks for ordinary cached loads in place of the streaming __ldcs the
//          family was built with. Streaming is only free when there is no reuse
//          to discard. At the fine levels the matrix is tens of gigabytes and
//          streaming is right; at the coarse levels it is tens of megabytes,
//          small enough to stay resident and be re-read every iteration, and
//          marking it evict-first throws away exactly that reuse. Which side of
//          the line a level falls on is a measurement, not a guess, so it is
//          swept.
//
//   thousands
//          the MV row-partition count, as one plus its base-2 logarithm, so 1
//          means PART=1, 2 means 2, 4 means 8 and 6 means 32. Zero means the
//          historical PART=8, which keeps every rpt written before this field
//          existed decoding to what it used to mean.
//
//          PART is gridDim.x. Because blockIdx.x moves fastest, PART controls
//          how far apart in the row space the concurrently scheduled blocks
//          sit. LDG fixed it at 8 to spread the column stream over the memory
//          system, which pays while the matrix is far larger than cache and
//          costs at the coarse levels, where LDG_V2's plain 1D walk -- PART=1
//          here -- was measurably faster. It is a launch shape, not a code
//          path, so it costs one kernel argument rather than more
//          instantiations.
//
// So 4 is W=4 narrow streaming, -4 is W=4 wide streaming, 104 is W=4 narrow
// cached, and -104 is W=4 wide cached, each at the default PART=8; 1104 is that
// last one at PART=1. LDG_V2 and the TMA families only ever put a plain W or
// rpt here, which decodes unchanged.
inline int SvRptWidth(int rpt) { return (rpt < 0 ? -rpt : rpt) % 100; }
inline bool SvRptWide(int rpt) { return rpt < 0; }
inline bool SvRptCached(int rpt) { return ((rpt < 0 ? -rpt : rpt) / 100) % 10 != 0; }
inline int SvRptPart(int rpt)
{
    const int f = ((rpt < 0 ? -rpt : rpt) / 1000) % 10;
    return f == 0 ? 8 : (1 << (f - 1));
}
// Suffix for the autotune banner: "" streaming narrow, "w" wide, "c" cached,
// with "pN" appended when the partition count is not the default 8.
inline const char* SvRptTag(int rpt)
{
    const bool w = SvRptWide(rpt);
    const bool c = SvRptCached(rpt);
    const char* base = w ? (c ? "wc" : "w") : (c ? "c" : "");
    const int p = SvRptPart(rpt);
    if (p == 8)
        return base;
    static thread_local char buf[16];
    snprintf(buf, sizeof buf, "%sp%d", base, p);
    return buf;
}
void SetSvChoice(int level, int kind, int blk, int unroll, int rpt);
// sweep: 0 (default) one full cycle of both, preserving prior behaviour;
// 1 forward only; 2 backward only. Forward runs over L, backward over U --
// different submatrices, so isolating one answers whether a config's
// advantage is symmetric between them or belongs to one triangle.
float TimeSvConfig(const SparseMatrix& A, double* rv, double* xv, int kind, int blk, int unroll, int rpt, int iters,
    int sweep = 0);
// dir: 0 the full matrix A, as ComputeSPMV multiplies it; 1 the L submatrix and
// 2 the U submatrix, as ComputeSYMGS multiplies them, with the alpha and beta
// each of those call sites actually passes. The selected MV config comes from
// dir 0 alone, so 1 and 2 exist to measure what that choice costs.
float TimeMvConfigDir(const SparseMatrix& A, double* x, double* y, int kind, int blk, int unroll, int rpt, int parts,
    int iters, int dir);
// `parts` splits the MV row space into that many block-partitions (LDG_V2 only;
// 1 = the original flat 1-D grid). Ignored by the other families.
void SetMvChoice(int level, int kind, int blk, int unroll, int rpt, int parts);
float TimeMvConfig(
    const SparseMatrix& A, double* x, double* y, int kind, int blk, int unroll, int rpt, int parts, int iters);
void AutotuneSymGS(const SparseMatrix& A);

///////// L2 Memory Compression Allocation Support Routines //
cudaError_t setProp(CUmemAllocationProp* prop);
cudaError_t cudaMallocCompressible(void** adr, size_t size);
cudaError_t cudaFreeCompressible(void* ptr, size_t size);

///////// Allocate CUDA Memory for data structures //
slice_ptr_t EstimateLUmem(local_int_t n, local_int_t padded_n, local_int_t level, int slice_size);
size_t EstimateGpuOptMem(const SparseMatrix& A_in);
void AllocateMemCuda(SparseMatrix& A_in);
void AllocateMemOptCuda(SparseMatrix& A_in);

///////// Deallocate CUDA Memory for data structures //
void DeleteMatrixGpu(SparseMatrix& A);

///////// Genrerate Problem //
void GenerateProblemCuda(SparseMatrix& A, Vector* b, Vector* x, Vector* xexact);

// Halo Exchange
void SetupHaloCuda(SparseMatrix& A, local_int_t sendbufld, local_int_t* sendlen, local_int_t* sendbuff,
    local_int_t* tot_to_send, int* nneighs, int* neighs_h, local_int_t* sendlen_h, local_int_t** elem_to_send_d);
void ExtToLocMapCuda(
    local_int_t localNumberOfRows, local_int_t str, local_int_t end, local_int_t* extToLocMap, local_int_t* eltsToRecv);
void ExtTolocCuda(local_int_t localNumberOfRows, int neighborId, slice_ptr_t ext_nnz, local_int_t* csr_ext_columns,
    double* csr_ext_values, slice_ptr_t* ext2csr_offsets, local_int_t* extToLocMap, local_int_t* csrColumns);
void PackSendBufferCuda(const SparseMatrix& A, Vector& x, bool cpu_data, cudaStream_t stream1);
void ExchangeHaloCuda(const SparseMatrix& A, Vector& x, cudaStream_t stream1, int use_ibarrier = 0);

// Optimize Problem
void SetVectorAscCuda(local_int_t* arr, local_int_t n);
void ColorMatrixCuda(double* A_vals, local_int_t* A_col, local_int_t* nnzPerRow, local_int_t rows, local_int_t* color,
    int* num_colors, int* count_colors, int max_colors, local_int_t* ref2opt, local_int_t* opt2ref, int rank, int nx,
    int* rowhash);
void PermElemToSendCuda(local_int_t totalToBeSent, local_int_t* elementsToSend, local_int_t* perm);
// The SELL slice-offset and column device arrays are width-agnostic (void*); the
// concrete int32/int64 element type is chosen at runtime from `mode` inside the
// wrapper via dispatchIndexMode (see IndexMode.hpp).
#ifndef EXPLICIT_KERNELS
void EllPermColumnsValuesCuda(local_int_t localNumberOfRows, local_int_t* nnzPerRow, local_int_t* csrColumns,
    double* csrValues, slice_ptr_t* permOffsets, void* permColumns, double* permValues, local_int_t* opt2ref,
    local_int_t* ref2opt, slice_ptr_t* diagonalIdx, slice_ptr_t* permLOffsets, slice_ptr_t* permUOffsets, bool diag,
    local_int_t slice_size, IndexMode mode);
void EllMaxRowLenPerBlockCuda(local_int_t nrow, int sliceSize, slice_ptr_t* sellLPermOffsets,
    slice_ptr_t* sellUPermOffsets, void* sellLSliceMrl, void* sellUSliceMrl, IndexMode mode);
void PrefixsumCuda(local_int_t localNumberOfRows, void* arr, IndexMode mode);
// 64-bit sum of a slice_ptr_t device array (exact per-matrix nnz for cuSPARSE).
slice_ptr_t SumSlicePtrCuda(const slice_ptr_t* arr, local_int_t n);
void MultiplyBySliceSizeCUDA(local_int_t nrow, int slice_size, void* arr, IndexMode mode);
void CreateAMatrixSliceOffsetsCuda(local_int_t nrow, local_int_t slice_size, void* arr, IndexMode mode);
void CreateSellLUColumnsValuesCuda(const local_int_t n, int sliceSize, void* columns, double* values,
    void* sellLSliceOffset, void* sellLColumns, double* sellLValues, void* sellUSliceOffset,
    void* sellUColumns, double* sellUValues, int level, IndexMode mode);
// Reads slice-offset element `index` from a width-agnostic offset array, widened to 64-bit.
long long ReadSellOffsetCuda(const void* arr, size_t index, IndexMode mode);
#else
// Explicit kernels: fixed-width device arrays, no IndexMode dispatch (see SparseMatrix.hpp).
void EllPermColumnsValuesCuda(local_int_t localNumberOfRows, local_int_t* nnzPerRow, local_int_t* csrColumns,
    double* csrValues, slice_ptr_t* permOffsets, local_int_t* permColumns, double* permValues, local_int_t* opt2ref,
    local_int_t* ref2opt, slice_ptr_t* diagonalIdx, slice_ptr_t* permLOffsets, slice_ptr_t* permUOffsets, bool diag,
    local_int_t slice_size);
void EllMaxRowLenPerBlockCuda(local_int_t nrow, int sliceSize, slice_ptr_t* sellLPermOffsets,
    slice_ptr_t* sellUPermOffsets, slice_ptr_t* sellLSliceMrl, slice_ptr_t* sellUSliceMrl);
void PrefixsumCuda(local_int_t localNumberOfRows, slice_ptr_t* arr);
void MultiplyBySliceSizeCUDA(local_int_t nrow, int slice_size, slice_ptr_t* arr);
void CreateAMatrixSliceOffsetsCuda(local_int_t nrow, local_int_t slice_size, slice_ptr_t* arr);
void CreateSellLUColumnsValuesCuda(const local_int_t n, int sliceSize, local_int_t* columns, double* values,
    slice_ptr_t* sellLSliceOffset, local_int_t* sellLColumns, double* sellLValues, slice_ptr_t* sellUSliceOffset,
    local_int_t* sellUColumns, double* sellUValues, int level);
#endif
void PermVectorCuda(local_int_t* perm, Vector& x, local_int_t length);
void F2cPermCuda(local_int_t nrow_c, local_int_t* f2c, local_int_t* f2cPerm, local_int_t* permF, local_int_t* ipermC);

// Test CG
void ReplaceMatrixDiagonalCuda(SparseMatrix& A, Vector& diagonal);
void CopyMatrixDiagonalCuda(SparseMatrix& A, Vector& diagonal);

// CG Support Kernels
// 1. MG
void ComputeRestrictionCuda(const SparseMatrix& A, const Vector& r);
void ComputeProlongationCuda(const SparseMatrix& A, Vector& x);

// 2. WAXPBY
void ComputeWAXPBYCuda(
    const local_int_t n, const double alpha, const Vector& x, const double beta, const Vector& y, Vector& w);

// 3.SYMGS
void SpmvDiagCuda(local_int_t n, double* x, double* d);
void AxpbyCuda(local_int_t n, double* x, double* y, double* z);
void SpFmaCuda(local_int_t n, double* x, double* y, double* z);

// 4.External Matrix SpMV + Scatter
void ExtSpMVCuda(SparseMatrix& A, double alpha, double* x, double* y);

// Transfer Problem to CPU
size_t CopyDataToHostCuda(SparseMatrix& A, Vector* b, Vector* x, Vector* xexact);
enum DIR{Forward = 0, Backward = 1, General = 2};
void sv_sell(DIR d, const SparseMatrix & A, double *rv, double *xv);
void mv_sell(DIR d, const SparseMatrix & A, double alpha, double beta, double *x, double *y);
#endif