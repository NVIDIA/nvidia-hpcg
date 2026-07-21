
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
 @file GenerateProblem.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

#include "mytimer.hpp"

#include "GenerateProblem.hpp"
#include "GenerateProblem_ref.hpp"
#ifdef USE_CUDA
#include "Cuda.hpp"
#include "CudaKernels.hpp"
#endif

#ifdef USE_GRACE
#include "CpuKernels.hpp"
#endif

/*!
  Routine to generate a sparse matrix, right hand side, initial guess, and exact solution.

  @param[in]  A        The generated system matrix
  @param[inout] b      The newly allocated and generated right hand side vector (if b!=0 on entry)
  @param[inout] x      The newly allocated solution vector with entries set to 0.0 (if x!=0 on entry)
  @param[inout] xexact The newly allocated solution vector with entries set to the exact solution (if the xexact!=0
  non-zero on entry)

  @see GenerateGeometry
*/
#ifdef USE_CUDA
void GenerateProblem_Gpu(SparseMatrix& A, Vector* b, Vector* x, Vector* xexact)
{
    global_int_t nx = A.geom->nx;
    global_int_t ny = A.geom->ny;
    global_int_t nz = A.geom->nz;
    global_int_t gnx = A.geom->gnx;
    global_int_t gny = A.geom->gny;
    global_int_t gnz = A.geom->gnz;
    global_int_t gix0 = A.geom->gix0;
    global_int_t giy0 = A.geom->giy0;
    global_int_t giz0 = A.geom->giz0;

    local_int_t localNumberOfRows = nx * ny * nz;
    local_int_t numberOfNonzerosPerRow = 27;
    global_int_t totalNumberOfRows = gnx * gny * gnz;

    if (b != 0)
        InitializeVector(*b, localNumberOfRows, GPU);
    if (x != 0)
        InitializeVector(*x, localNumberOfRows, GPU);
    if (xexact != 0)
        InitializeVector(*xexact, localNumberOfRows, GPU);

    GenerateProblemCuda(A, b, x, xexact);

    const slice_ptr_t localNumberOfNonzeros = A.localNumberOfNonzeros;
    global_int_t totalNumberOfNonzeros = 27LL * ((gnx - 2LL) * (gny - 2LL) * (gnz - 2LL))
        + 18LL
            * (2LL * ((gnx - 2LL) * (gny - 2LL)) + 2LL * ((gnx - 2LL) * (gnz - 2LL))
                + 2LL * ((gny - 2LL) * (gnz - 2LL)))
        + 12LL * (4LL * (gnx - 2LL) + 4LL * (gny - 2LL) + 4LL * (gnz - 2LL)) + 8LL * 8LL;

    A.title = 0;
    A.totalNumberOfRows = totalNumberOfRows;
    A.totalNumberOfNonzeros = totalNumberOfNonzeros;
    A.localNumberOfRows = localNumberOfRows;
    A.localNumberOfColumns = localNumberOfRows;
    A.localNumberOfNonzeros = localNumberOfNonzeros;

    return;
}
#endif

#ifdef USE_GRACE
// Neighbor rank to sequential ID and vice versa
extern int *rankToId_h, *idToRank_h;
// GenerateProblem_Cpu is called 4 times for each level
// Sometimes we need to perform actions based on the level (global across the applications)
int global_steps = 0;
void GenerateProblem_Cpu(SparseMatrix& A, Vector* b, Vector* x, Vector* xexact)
{
    // Make local copies of geometry information.  Use global_int_t since the RHS products in the calculations
    // below may result in global range values.
    global_int_t nx = A.geom->nx;
    global_int_t ny = A.geom->ny;
    global_int_t nz = A.geom->nz;
    global_int_t gnx = A.geom->gnx;
    global_int_t gny = A.geom->gny;
    global_int_t gnz = A.geom->gnz;
    global_int_t gix0 = A.geom->gix0;
    global_int_t giy0 = A.geom->giy0;
    global_int_t giz0 = A.geom->giz0;

    local_int_t localNumberOfRows = nx * ny * nz; // This is the size of our subblock
    // If this assert fails, it most likely means that the local_int_t is set to int and should be set to long long
    assert(localNumberOfRows
        > 0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)
    local_int_t numberOfNonzerosPerRow
        = 27; // We are approximating a 27-point finite element/volume/difference 3D stencil

    global_int_t totalNumberOfRows = gnx * gny * gnz; // Total number of grid points in mesh
    // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
    assert(totalNumberOfRows
        > 0); // Throw an exception of the number of rows is less than zero (can happen if int overflow)

    // Allocate arrays that are of length localNumberOfRows
    if (global_steps == 0)
    {
        rankToId_h = new int[A.geom->size + 1];
        idToRank_h = new int[27];
        global_steps++;
    }
    local_int_t* nonzerosInRow = new local_int_t[localNumberOfRows];
    global_int_t** mtxIndG = new global_int_t*[localNumberOfRows];
    local_int_t** mtxIndL = new local_int_t*[localNumberOfRows];
    double** matrixValues = new double*[localNumberOfRows];
    double** matrixDiagonal = new double*[localNumberOfRows];

    if (b != 0)
        InitializeVector(*b, localNumberOfRows, CPU);
    if (x != 0)
        InitializeVector(*x, localNumberOfRows, CPU);
    if (xexact != 0)
        InitializeVector(*xexact, localNumberOfRows, CPU);
    double* bv = 0;
    double* xv = 0;
    double* xexactv = 0;
    if (b != 0)
        bv = b->values; // Only compute exact solution if requested
    if (x != 0)
        xv = x->values; // Only compute exact solution if requested
    if (xexact != 0)
        xexactv = xexact->values; // Only compute exact solution if requested
    A.localToGlobalMap.resize(localNumberOfRows);

    // Use a parallel loop to do initial assignment:
    // distributes the physical placement of arrays of pointers across the memory system
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
    for (local_int_t i = 0; i < localNumberOfRows; ++i)
    {
        matrixValues[i] = 0;
        matrixDiagonal[i] = 0;
        mtxIndG[i] = 0;
        mtxIndL[i] = 0;
    }

    if (global_steps == 1)
    {
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
        for (local_int_t i = 0; i < A.geom->size + 1; i++)
        {
            rankToId_h[i] = 0;
        }
        global_steps++;
    }

    // Now allocate the arrays pointed to
    // Use size_t: localNumberOfRows * 27 overflows int32 for large local problems (e.g. 512^3 x 320).
    const size_t rowNnzBudget = (size_t) localNumberOfRows * (size_t) numberOfNonzerosPerRow;
    mtxIndL[0] = new local_int_t[rowNnzBudget];
    matrixValues[0] = new double[rowNnzBudget];
    mtxIndG[0] = new global_int_t[rowNnzBudget];

    slice_ptr_t localNumberOfNonzeros = 0;
    slice_ptr_t ext_nnz = 0;
#ifndef HPCG_NO_OPENMP
#pragma omp parallel for reduction(+ : localNumberOfNonzeros) reduction(+ : ext_nnz)
#endif
    for (local_int_t i = 0; i < localNumberOfRows; i++)
    {
        mtxIndL[i] = mtxIndL[0] + (size_t) i * numberOfNonzerosPerRow;
        matrixValues[i] = matrixValues[0] + (size_t) i * numberOfNonzerosPerRow;
        mtxIndG[i] = mtxIndG[0] + (size_t) i * numberOfNonzerosPerRow;

        const local_int_t iz = (i / (nx * ny));
        const local_int_t iy = (i - iz * nx * ny) / nx;
        const local_int_t ix = i - (iz * ny + iy) * nx;
        const global_int_t gix = ix + gix0;
        const global_int_t giy = iy + giy0;
        const global_int_t giz = iz + giz0;

        local_int_t currentLocalRow = i;
        global_int_t currentGlobalRow = gix + giy * gnx + giz * gnx * gny;

        A.localToGlobalMap[currentLocalRow] = currentGlobalRow;

        // Fast path: points strictly interior to this rank's subdomain have all 27
        // neighbors valid and local, so we emit their global column indices and values
        // directly and skip the per-neighbor owner-rank test. Produces results identical
        // to the slow path (every row for a single rank, ~all rows otherwise).
        if (IsInteriorRowCpu(ix, iy, iz, nx, ny, nz))
        {
            global_int_t* idxG = mtxIndG[currentLocalRow];
            double* vals = matrixValues[currentLocalRow];
            for (int k = 0; k < 27; k++)
            {
                idxG[k] = (gix + tid2indCpu[k][0]) + (giy + tid2indCpu[k][1]) * gnx
                    + (giz + tid2indCpu[k][2]) * gnx * gny;
                vals[k] = (k == 13) ? 26.0 : -1.0;
            }
            matrixDiagonal[currentLocalRow] = vals + 13;
            nonzerosInRow[currentLocalRow] = 27;
            localNumberOfNonzeros += 27;
            if (b != 0)
                bv[currentLocalRow] = 0.0; // 26.0 - (27 - 1)
            if (x != 0)
                xv[currentLocalRow] = 0.0;
            if (xexact != 0)
                xexactv[currentLocalRow] = 1.0;
            continue;
        }

        // Boundary rows: classify each neighbor systematically (valid / owner rank).
        char numberOfNonzerosInRow = 0;
        double* currentValuePointer = matrixValues[currentLocalRow];
        global_int_t* currentIndexPointerG = mtxIndG[currentLocalRow];
        double* diagonalPointer = nullptr;
        for (int k = 0; k < 27; k++)
        {
            const StencilNeighborCpu nb = ClassifyStencilNeighborCpu(*A.geom, ix, iy, iz, k);
            if (!nb.valid)
                continue;

            *currentIndexPointerG++ = nb.globalCol;
            if (k == 13)
            {
                *currentValuePointer = 26.0;
                diagonalPointer = currentValuePointer;
            }
            else
            {
                *currentValuePointer = -1.0;
            }

            if (!nb.isLocal)
            {
                if (global_steps == 2)
                {
                    // Deliberate benign race: several threads may mark the same ownerRank from
                    // the parallel boundary-row loop, but every writer stores the identical
                    // constant 1 to a naturally-aligned int, so the store never tears and the
                    // result is deterministically 1 regardless of interleaving. The array is
                    // read only after this loop's implicit barrier (PrefixsumCpu below), so
                    // there is no concurrent reader. No atomic is needed.
                    rankToId_h[nb.ownerRank + 1] = 1; // sequential Id assigned later via prefix sum
                }
                ext_nnz++;
            }

            currentValuePointer++;
            numberOfNonzerosInRow++;
        }

        matrixDiagonal[currentLocalRow] = diagonalPointer;
        nonzerosInRow[currentLocalRow] = numberOfNonzerosInRow;
        localNumberOfNonzeros += numberOfNonzerosInRow;
        if (b != 0)
            bv[currentLocalRow] = 26.0 - ((double) (numberOfNonzerosInRow - 1));
        if (x != 0)
            xv[currentLocalRow] = 0.0;
        if (xexact != 0)
            xexactv[currentLocalRow] = 1.0;
    }

    // Prefixsum to RakToId
    // Map physical neighbor ranks to sequential IDs
    //  less memory consumption
    if (global_steps == 2)
    {
        PrefixsumCpu(rankToId_h + 1, A.geom->size);
        int counter = 1;
        for (int i = 1; i < A.geom->size + 1; i++)
        {
            if (rankToId_h[i] == counter)
            {
                idToRank_h[counter - 1] = i - 1;
                counter++;
            }
        }
        global_steps++;
    }

#ifdef HPCG_DETAILED_DEBUG
    HPCG_fout << "Process " << A.geom->rank << " of " << A.geom->size << " has " << localNumberOfRows << " rows."
              << endl
              << "Process " << A.geom->rank << " of " << A.geom->size << " has " << localNumberOfNonzeros
              << " nonzeros." << endl;
#endif

    global_int_t totalNumberOfNonzeros = 27LL * ((gnx - 2LL) * (gny - 2LL) * (gnz - 2LL))
        + 18LL
            * (2LL * ((gnx - 2LL) * (gny - 2LL)) + 2LL * ((gnx - 2LL) * (gnz - 2LL))
                + 2LL * ((gny - 2LL) * (gnz - 2LL)))
        + 12LL * (4LL * (gnx - 2LL) + 4LL * (gny - 2LL) + 4LL * (gnz - 2LL)) + 8LL * 8LL;

    // If this assert fails, it most likely means that the global_int_t is set to int and should be set to long long
    // This assert is usually the first to fail as problem size increases beyond the 32-bit integer range.
    assert(totalNumberOfNonzeros
        > 0); // Throw an exception of the number of nonzeros is less than zero (can happen if int overflow)

    A.title = 0;
    A.totalNumberOfRows = totalNumberOfRows;
    A.totalNumberOfNonzeros = totalNumberOfNonzeros;
    A.localNumberOfRows = localNumberOfRows;
    A.localNumberOfColumns = localNumberOfRows;
    A.localNumberOfNonzeros = localNumberOfNonzeros;
    A.nonzerosInRow = nonzerosInRow;
    A.mtxIndG = mtxIndG;
    A.mtxIndL = mtxIndL;
    A.matrixValues = matrixValues;
    A.matrixDiagonal = matrixDiagonal;
    A.extNnz = ext_nnz;

    return;
}
#endif // USE_GRACE

void GenerateProblem(SparseMatrix& A, Vector* b, Vector* x, Vector* xexact)
{
    if (A.rankType == GPU)
    {
#ifdef USE_CUDA
        GenerateProblem_Gpu(A, b, x, xexact);
#endif
    }
    else
    {
#ifdef USE_GRACE
        GenerateProblem_Cpu(A, b, x, xexact);
#endif
    }
}