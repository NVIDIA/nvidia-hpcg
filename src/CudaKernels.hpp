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
#include "SparseMatrix.hpp"

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

#ifdef EXPLICIT_KERNELS
///////// Explicit Sliced-ELL SpMV / SpSV //
// Which triangle of the permuted Sliced-ELL operator a call addresses: the full
// matrix, the strict lower part, or the strict upper part.
enum DIR
{
    Forward = 0,
    Backward = 1,
    General = 2
};

/*
  The project-wide Sliced-ELL kernel family numbering, selected per operator by
  HPCG_EXPLICIT_MV_KIND / HPCG_EXPLICIT_SV_KIND. The numbering is fixed so that
  a family keeps the same number as it lands; this build implements LDG and
  LDG_V2, and refuses the rest rather than serving a substitute.
*/
enum SellKernelKind
{
    SELL_KIND_LDG = 0,
    SELL_KIND_TMA = 1,
    SELL_KIND_LDGV2 = 2,
    SELL_KIND_TMA2D = 3,
    SELL_KIND_LDGV3 = 4,
    SELL_KIND_TMAEX = 5
};

// Family and launch shape of the explicit kernels. Overridable per run through
// the matching environment variables; InitKernelConfig() installs the defaults,
// which depend on the family because the families are not instantiated over the
// same grid of block sizes and unroll depths.
struct KernelConfig
{
    int SV_KIND;
    int MV_KIND;
    int SV_UNROLL;
    int MV_UNROLL;
    int SV_BLOCK_SIZE;
    int MV_BLOCK_SIZE;
    // Rows each thread owns. LDG_V2 only; the scalar LDG family is one row per
    // thread by construction and ignores it.
    int SV_W;
    int MV_W;
    // Number of equal row partitions the LDG_V2 SpMV grid is split into, one per
    // blockIdx.x. 1 is the flat 1D grid. LDG_V2 SpMV only.
    int MV_PARTS;
};

extern KernelConfig g_config;

void InitKernelConfig();

/*
  Whether this SpMV/SpSV should run on the explicit kernels rather than
  cuSPARSE.

  False unless HPCG_EXPLICIT_MV / HPCG_EXPLICIT_SV is set to a nonzero value,
  so a default run of an EXPLICIT_KERNELS build takes exactly the same code
  path as a build without it. Also false, with a one-time warning, when the
  matrix uses an index mode the explicit kernels are not instantiated for, or
  when the requested family is not one this build contains.

  Takes the whole matrix rather than a flag so that the per-level selection the
  autotuner will drive lands here without touching the call sites.
*/
bool UseExplicitSpMV(const SparseMatrix& A);
bool UseExplicitSpSV(const SparseMatrix& A);

void mv_sell(DIR d, const SparseMatrix& A, double alpha, double beta, double* x, double* y);
void sv_sell(DIR d, const SparseMatrix& A, double* rv, double* xv);

/*
  LDG_V2 launchers, defined in mv-ldg-v2.cu / spsv-ldg-v2.cu and instantiated
  there for the 32-bit-offset case alone. Both return false when the requested
  block size / unroll / W triple, or the launch shape it implies for this
  matrix, is not one the family can serve; nothing is launched in that case.
*/
template <class OffsetT>
bool MvLdgV2SellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y,
    const OffsetT* slice_offsets, const idx32_t* columns, const double* values, cudaStream_t stream, int blk,
    int unroll, int w, int parts);

template <class OffsetT>
bool SpsvLdgV2SellCfg(bool forward, const SparseMatrix& A, const double* rv, double* xv, const OffsetT* slice_offsets,
    const idx32_t* columns, const double* values, cudaStream_t stream, int blk, int unroll, int w);
#endif
#endif