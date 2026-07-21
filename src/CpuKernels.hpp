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

#ifndef CPUKERNELS_HPP
#define CPUKERNELS_HPP

#ifdef USE_GRACE

#include <nvpl_sparse.h>
extern nvpl_sparse_handle_t nvpl_sparse_handle;

#include "SparseMatrix.hpp"
#include "Vector.hpp"
#include <algorithm>
#include <random>
#include <vector>

#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

#ifndef HPCG_NO_MPI
#include <mpi.h>
#endif

#ifndef HPCG_NO_OPENMP
#include <omp.h>
#endif

///////// Deallocate CPU Memory for data structures //
void DeleteMatrixCpu(SparseMatrix& A);

///////// Find the size of CPU reference allocated memory //
size_t EstimateCpuRefMem(SparseMatrix& A);

/*
    Translation of a 3D point in all directions
    27 possibilities
*/
constexpr int tid2indCpu[32][4] = {{-1, -1, -1, 0}, {0, -1, -1, 0}, {1, -1, -1, 0}, {-1, 0, -1, 0}, {0, 0, -1, 0},
    {1, 0, -1, 0}, {-1, 1, -1, 0}, {0, 1, -1, 0}, {1, 1, -1, 0}, {-1, -1, 0, 0}, {0, -1, 0, 0}, {1, -1, 0, 0},
    {-1, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {-1, 1, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}, {-1, -1, 1, 0}, {0, -1, 1, 0},
    {1, -1, 1, 0}, {-1, 0, 1, 0}, {0, 0, 1, 0}, {1, 0, 1, 0}, {-1, 1, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 0}, {0, 0, 0, 0},
    {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

/*
    Systematic classification of a 27-point stencil neighbor on the CPU path.

    GenerateProblem (global column indices), SetupHalo (local column indices), and the
    halo external-column pass all need to answer the same questions about each neighbor
    of a local grid point: is it inside the global domain, which rank owns it, and what
    is its column index in this rank's numbering (if local) and in the owner's numbering
    (for the halo lookup). These helpers centralize that logic so the three call sites
    stay consistent, and isolate the hybrid GPU/CPU uneven-partition (different_dim)
    handling to one place.
*/
struct StencilNeighborCpu
{
    bool valid;             //!< neighbor lies inside the global domain
    bool isLocal;           //!< neighbor is owned by this rank
    int ownerRank;          //!< logical rank that owns the neighbor
    global_int_t globalCol; //!< global column index of the neighbor
    local_int_t localCol;   //!< column index in THIS rank's local numbering (valid iff isLocal)
    local_int_t ownerCol;   //!< column index in the OWNER rank's local numbering (for halo lookup)
};

//! True when a local point is strictly interior to this rank's subdomain, i.e. all 27
//! of its stencil neighbors are owned by this rank (no halo, no owner-rank test needed).
static inline bool IsInteriorRowCpu(
    local_int_t ix, local_int_t iy, local_int_t iz, local_int_t nx, local_int_t ny, local_int_t nz)
{
    return ix >= 1 && ix < nx - 1 && iy >= 1 && iy < ny - 1 && iz >= 1 && iz < nz - 1;
}

//! Column index of a neighbor (given by its global coordinates) in THIS rank's numbering.
static inline local_int_t LocalColCpu(
    const Geometry& g, global_int_t cgix, global_int_t cgiy, global_int_t cgiz)
{
    return (local_int_t) ((cgiz - g.giz0) * g.ny * g.nx + (cgiy - g.giy0) * g.nx + (cgix - g.gix0));
}

//! Classify neighbor k (offset tid2indCpu[k]) of local point (ix,iy,iz).
static inline StencilNeighborCpu ClassifyStencilNeighborCpu(
    const Geometry& g, local_int_t ix, local_int_t iy, local_int_t iz, int k)
{
    StencilNeighborCpu n;
    const global_int_t cgix = ix + g.gix0 + tid2indCpu[k][0];
    const global_int_t cgiy = iy + g.giy0 + tid2indCpu[k][1];
    const global_int_t cgiz = iz + g.giz0 + tid2indCpu[k][2];

    n.valid = (cgiz > -1 && cgiz < g.gnz && cgiy > -1 && cgiy < g.gny && cgix > -1 && cgix < g.gnx);
    if (!n.valid)
        return n;

    n.globalCol = cgix + cgiy * g.gnx + cgiz * g.gnx * g.gny;

    // Owner rank coordinates and owner-local coordinates. Default: uniform decomposition.
    int ipx = cgix / g.nx;
    int ipy = cgiy / g.ny;
    int ipz = cgiz / g.nz;
    local_int_t zi = cgiz % g.nz;
    local_int_t yi = cgiy % g.ny;
    local_int_t xi = cgix % g.nx;
    global_int_t new_nx = g.nx;
    global_int_t new_ny = g.ny;

    // For hybrid GPU/CPU runs one dimension may be partitioned unevenly; along that
    // dimension the owner rank and owner-local coordinate must be derived from this
    // rank's position rather than a uniform divide.
    if (g.different_dim == Z)
    {
        long long int local = cgiz - g.giz0;
        if (local >= 0 && local < g.nz) { ipz = g.ipz; zi = local; }
        else if (local < 0) { ipz = g.ipz - 1; zi = g.previous_neighbor_dim - 1; }
        else { ipz = g.ipz + 1; zi = 0; }
    }
    else if (g.different_dim == Y)
    {
        long long int local = cgiy - g.giy0;
        if (local >= 0 && local < g.ny) { ipy = g.ipy; yi = local; }
        else if (local < 0) { ipy = g.ipy - 1; yi = g.previous_neighbor_dim - 1; new_ny = g.previous_neighbor_dim; }
        else { ipy = g.ipy + 1; yi = 0; new_ny = g.next_neighbor_dim; }
    }
    else if (g.different_dim == X)
    {
        long long int local = cgix - g.gix0;
        if (local >= 0 && local < g.nx) { ipx = g.ipx; xi = local; }
        else if (local < 0) { ipx = g.ipx - 1; xi = g.previous_neighbor_dim - 1; new_nx = g.previous_neighbor_dim; }
        else { ipx = g.ipx + 1; xi = 0; new_nx = g.next_neighbor_dim; }
    }

    n.ownerRank = ipx + ipy * g.npx + ipz * g.npy * g.npx;
    n.isLocal = (n.ownerRank == g.logical_rank);
    n.ownerCol = (local_int_t) (zi * new_ny * new_nx + yi * new_nx + xi);
    n.localCol = n.isLocal ? LocalColCpu(g, cgix, cgiy, cgiz) : n.ownerCol;
    return n;
}

// Generate Problem
// Inclusive Prefix Sum (int for neighbor-rank compression; slice_ptr_t for CSR offsets)
void PrefixsumCpu(int* x, int N);
void PrefixsumCpu(slice_ptr_t* x, int N);

// Optimize Problem
void AllocateMemCpu(SparseMatrix& A_in);
size_t EstimateCpuOptMem(const SparseMatrix& A_in);
void ColorMatrixCpu(SparseMatrix& A, int* num_colors);
void CreateSellPermCpu(SparseMatrix& A);
void F2cPermCpu(local_int_t nrow_c, local_int_t* f2c, local_int_t* f2c_perm, local_int_t* perm_f, local_int_t* iperm_c);

// Permute a vector using coloring buffer
void PermVectorCpu(local_int_t* perm, Vector& x, local_int_t length);

// Test CG
void ReplaceMatrixDiagonalCpu(SparseMatrix& A, Vector diagonal);

// CG Support Kernels
// Dot-product Per single rank
void ComputeDotProductCpu(const local_int_t n, const Vector& x, const Vector& y, double& result, bool& isOptimized);

// WAXPBY
int ComputeWAXPBYCpu(const local_int_t n, const double alpha, const Vector& x, const double beta, const Vector& y,
    Vector& w, bool& isOptimized);
// SYMGS
void SpmvDiagCpu(local_int_t n, const double* x, double* y, double* z);
void AxpbyCpu(local_int_t n, double* x, double* y, double* z);
void SpFmaCpu(local_int_t n, const double* x, double* y, double* z);

// External Matrix SpMV + Scatter
void ExtSpMVCpu(const SparseMatrix& A, const local_int_t n, const double alpha, const double* x, double* y);

#endif // USE_GRACE
#endif // CPUKERNELS_HPP