
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
 @file SparseMatrix.hpp

 HPCG data structures for the sparse matrix
 */

#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#ifdef USE_CUDA
#include <cuda.h>
#ifndef EXPLICIT_KERNELS
#include <cusparse.h>
#endif
#endif

#ifdef USE_GRACE
#include <nvpl_sparse.h>
#endif

#include "Cuda.hpp"
#include "Geometry.hpp"
#include "IndexMode.hpp"
#include "MGData.hpp"
#include "Vector.hpp"
#include <cassert>
#include <vector>

extern bool Use_Hpcg_Mem_Reduction;

#ifndef HPCG_NO_MPI
extern p2p_comm_mode_t P2P_Mode;
#endif

#if __cplusplus < 201103L
// for C++03
#include <map>
typedef std::map<global_int_t, local_int_t> GlobalToLocalMap;
#else
// for C++11 or greater
#include <unordered_map>
using GlobalToLocalMap = std::unordered_map<global_int_t, local_int_t>;
#endif

#ifdef USE_CUDA
#ifndef EXPLICIT_KERNELS
struct CUSPARSE_STRUCT
{
    cusparseDnVecDescr_t vecX;
    cusparseDnVecDescr_t vecY;
    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matL;
    cusparseSpMatDescr_t matU;

    // CUSPARSE SpSV
    cusparseSpSVDescr_t spsvDescrL, spsvDescrU;
};
#endif // !EXPLICIT_KERNELS

/*!
  Width-agnostic device storage for the GPU Sliced-ELL operator arrays.

  The concrete element width (int32 vs int64) of the slice-offset and column
  arrays is chosen at run time from SparseMatrix::index_mode (see IndexMode.hpp).
  Values arrays remain double and stay in the typed SparseMatrix fields. Raw
  casts of these pointers are confined to CudaKernels.cu and OptimizeProblem.cpp.
 */
struct SellDeviceArrays
{
    void* aSliceOffsets = nullptr; // matA slice offsets (offset width)
    void* lSliceOffsets = nullptr; // matL slice offsets (offset width)
    void* uSliceOffsets = nullptr; // matU slice offsets (offset width)
    void* aColumns = nullptr;      // matA permuted columns (column width)
    void* lColumns = nullptr;      // matL permuted columns (column width)
    void* uColumns = nullptr;      // matU permuted columns (column width)

    // True when lColumns/uColumns own their allocation (int64 columns) rather
    // than aliasing gpuAux.columns (int32 columns). Governs cudaFree in cleanup.
    bool ownsLuColumns = false;
};

struct GPU_AUX_STRUCT
{
    // Uncolored row related info
    local_int_t* nnzPerRow;
    local_int_t* columns;
    double* values;
    slice_ptr_t* csrAPermOffsets;
    slice_ptr_t* csrLPermOffsets;
    slice_ptr_t* csrUPermOffsets;
    slice_ptr_t* diagonalIdx;

    // Sliced EllPACK Aux
    slice_ptr_t* sellADiagonalIdx;

    // Auxiliary data
    local_int_t* f2c;

    local_int_t* color;
    int* colorCountCpu;

    // MULTI-GPU Aux data
    local_int_t* map;
    slice_ptr_t* ext2csrOffsets;
    local_int_t* elementsToSend;
    global_int_t* localToGlobalMap;
    local_int_t compressNumberOfRows;
    double* sendBuffer;
};
#endif

#ifdef USE_GRACE
struct NVPL_SPARSE_STRUCT
{
    nvpl_sparse_dn_vec_descr_t vecX;
    nvpl_sparse_dn_vec_descr_t vecY;

    nvpl_sparse_sp_mat_descr_t matL;
    nvpl_sparse_sp_mat_descr_t matU;
    nvpl_sparse_sp_mat_descr_t matA;

    nvpl_sparse_spsv_descr_t spsvDescrL, spsvDescrU;
    nvpl_sparse_spmv_descr_t spmvADescr, spmvLDescr, spmvUDescr;
};

struct CPU_AUX_STRUCT
{
    // Auxiliary data
    //  Coloring info as number of colors and where each color starts
    //  Also keep information on how many consecutive rows share the same color
    //  This assumes matrix reordering (rows with same color are packed)
    local_int_t* color;
    local_int_t* firstRowOfColor;
    local_int_t* nRowsWithColor;
    local_int_t* tempIndex;
};
#endif

struct SparseMatrix_STRUCT
{
    rank_type_t rankType;
    int level;
    char* title;                                //!< name of the sparse matrix
    Geometry* geom;                             //!< geometry associated with this matrix
    global_int_t totalNumberOfRows;             //!< total number of matrix rows across all processes
    global_int_t totalNumberOfNonzeros;         //!< total number of matrix nonzeros across all processes
    local_int_t localNumberOfRows;              //!< number of rows local to this process
    local_int_t localNumberOfColumns;           //!< number of columns local to this process
    slice_ptr_t localNumberOfNonzeros; //!< nnz local to this process (64-bit; can exceed 2^31)
    local_int_t* nonzerosInRow;                 //!< The number of nonzeros in a row will always be 27 or fewer
    global_int_t** mtxIndG;                     //!< matrix indices as global values
    local_int_t** mtxIndL;                      //!< matrix indices as local values
    double** matrixValues;                      //!< values of matrix entries
    double** matrixDiagonal;                    //!< values of matrix diagonal entries
    GlobalToLocalMap globalToLocalMap;          //!< global-to-local mapping
    std::vector<global_int_t> localToGlobalMap; //!< local-to-global mapping
    mutable bool isDotProductOptimized;
    mutable bool isSpmvOptimized;
    mutable bool isMgOptimized;
    mutable bool isWaxpbyOptimized;

    mutable MGData* mgData; // Pointer to the coarse level data for this fine matrix
    void* optimizationData; // pointer that can be used to store implementation-specific data

    local_int_t totalToBeSent; //!< total number of entries to be sent
    local_int_t slice_size;
    slice_ptr_t sellALocalNumberOfNonzeros;
    slice_ptr_t sellLLocalNumberOfNonzeros;
    slice_ptr_t sellULocalNumberOfNonzeros;

#ifndef HPCG_NO_MPI
    local_int_t numberOfExternalValues; //!< number of entries that are external to this process
    int numberOfSendNeighbors;          //!< number of neighboring processes that will be send local data
    local_int_t* elementsToSend;        //!< elements to send to neighboring processes
    int* neighbors;                     //!< neighboring processes
    int* neighborsPhysical;
    local_int_t* receiveLength; //!< lenghts of messages received from neighboring processes
    local_int_t* sendLength;    //!< lenghts of messages sent to neighboring processes
    double* sendBuffer;         //!< send buffer for non-blocking sends
    slice_ptr_t extNnz; //!< external (halo) nnz; 64-bit like localNumberOfNonzeros
#endif

    // Optmization Data common between CPU and GPU
    // Coloring permutations
    local_int_t totalColors;
    local_int_t* ref2opt;
    local_int_t* opt2ref;
    local_int_t* f2cPerm;

    // Sliced EllPACK
#ifndef EXPLICIT_KERNELS
    // aarch64 host slice offsets and columns are width-agnostic (void*), sized by
    // --mi like GPU sellDev: offsets are int32 for --mi 0 and int64 for --mi 1/2;
    // columns are int32 for --mi 0/1 and int64 for --mi 2. Typed access is
    // confined to dispatchIndexMode() call sites in CpuKernels. The GPU
    // (cuSPARSE) path never reads these directly either -- it goes through
    // sellDev below. These fields are CPU/NVPL-only in a mixed-indexing build.
    void *sellASliceMrl, *sellLSliceMrl, *sellUSliceMrl;
    void *sellAPermColumns, *sellLPermColumns, *sellUPermColumns;
#else
    // Explicit GPU kernels (LDG/LDG2/LDG3/TMA) read these directly as fixed-
    // width device pointers -- they predate sellDev and never go through
    // cuSPARSE, so there is no IndexMode to dispatch on. slice_ptr_t is
    // unconditionally 64-bit (see Geometry.hpp); columns stay local_int_t.
    slice_ptr_t *sellASliceMrl, *sellLSliceMrl, *sellUSliceMrl;
    local_int_t *sellAPermColumns, *sellLPermColumns, *sellUPermColumns;
#endif
    double *sellAPermValues, *sellLPermValues, *sellUPermValues;
    double* diagonal;

    char* bufferSvL = nullptr;
    char* bufferSvU = nullptr;
    char* bufferMvA = nullptr;
    char* bufferMvL = nullptr;
    char* bufferMvU = nullptr;

    // MULTI-GPU data
    slice_ptr_t* csrExtOffsets;
    local_int_t* csrExtColumns;
    double* csrExtValues;
    double* tempBuffer;

    // When MPI_All2allv is used for P2P communication
    int* scounts;
    int* rcounts;
    int* sdispls;
    int* rdispls;

    // Runtime Sliced-ELL index widths (--mi). Used by GPU (cuSPARSE) and
    // aarch64 CPU (NVPL Sparse). Coarse levels inherit this in AllocateMem*.
    IndexMode index_mode = IndexMode::I32_I32;

#ifdef USE_CUDA
#ifndef EXPLICIT_KERNELS
    CUSPARSE_STRUCT cusparseOpt;
#endif
    GPU_AUX_STRUCT gpuAux;
    SellDeviceArrays sellDev; // Width-agnostic SELL offset/column device arrays
#endif

#ifdef USE_GRACE
    NVPL_SPARSE_STRUCT nvplSparseOpt;
    CPU_AUX_STRUCT cpuAux;
#endif

    mutable struct SparseMatrix_STRUCT* Ac; // Coarse grid matrix
};

typedef struct SparseMatrix_STRUCT SparseMatrix;

/*!
  Initializes the known system matrix data structure members to 0.

  @param[in] A the known system matrix
 */
inline void InitializeSparseMatrix(SparseMatrix& A, Geometry* geom)
{
    A.title = 0;
    A.geom = geom;
    A.totalNumberOfRows = 0;
    A.totalNumberOfNonzeros = 0;
    A.localNumberOfRows = 0;
    A.localNumberOfColumns = 0;
    A.localNumberOfNonzeros = 0;
    A.nonzerosInRow = 0;
    A.mtxIndG = 0;
    A.mtxIndL = 0;
    A.matrixValues = 0;
    A.matrixDiagonal = 0;

    // Optimization is ON by default. The code that switches it OFF is in the
    // functions that are meant to be optimized.
    A.isDotProductOptimized = true;
    A.isSpmvOptimized = true;
    A.isMgOptimized = true;
    A.isWaxpbyOptimized = true;

    A.totalToBeSent = 0;

#ifndef HPCG_NO_MPI
    A.numberOfExternalValues = 0;
    A.numberOfSendNeighbors = 0;
    A.totalToBeSent = 0;
    A.elementsToSend = 0;
    A.neighbors = 0;
    A.neighborsPhysical = 0;
    A.receiveLength = 0;
    A.sendLength = 0;
    A.sendBuffer = 0;
    A.extNnz = 0;
#endif
    A.mgData = 0; // Fine-to-coarse grid transfer initially not defined.

    return;
}

/*!
  Copy values from matrix diagonal into user-provided vector.

  @param[in] A the known system matrix.
  @param[inout] diagonal  Vector of diagonal values (must be allocated before call to this function).
 */
inline void CopyMatrixDiagonal(SparseMatrix& A, Vector& diagonal)
{
    double** curDiagA = A.matrixDiagonal;
    double* dv = diagonal.values;
    assert(A.localNumberOfRows == diagonal.localLength);
    for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
        dv[i] = *(curDiagA[i]);
    return;
}

/*!
  Replace specified matrix diagonal value.

  @param[inout] A The system matrix.
  @param[in] diagonal  Vector of diagonal values that will replace existing matrix diagonal values.
 */
inline void ReplaceMatrixDiagonal(SparseMatrix& A, Vector& diagonal)
{
    double** curDiagA = A.matrixDiagonal;
    double* dv = diagonal.values;
    assert(A.localNumberOfRows == diagonal.localLength);
    for (local_int_t i = 0; i < A.localNumberOfRows; ++i)
        *(curDiagA[i]) = dv[i];
    return;
}
#endif // SPARSEMATRIX_HPP
