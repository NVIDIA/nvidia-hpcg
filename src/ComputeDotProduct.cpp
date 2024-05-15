
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "mytimer.hpp"
#include <mpi.h>
#endif
#include "ComputeDotProduct.hpp"
#include "ComputeDotProduct_ref.hpp"
#ifdef USE_CUDA
#include "Cuda.hpp"
#define CHECK_CUBLAS(x)                                                                                                \
    do                                                                                                                 \
    {                                                                                                                  \
        cublasStatus_t cublasStatus = (x);                                                                             \
        if (cublasStatus != CUBLAS_STATUS_SUCCESS)                                                                     \
        {                                                                                                              \
            fprintf(stderr, "CUBLAS: %s = %d at (%s:%d)\n", #x, cublasStatus, __FILE__, __LINE__);                     \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)
#endif

#ifdef USE_GRACE
#include "CpuKernels.hpp"
#endif

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized);
  otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/

int ComputeDotProduct(const local_int_t n, const Vector& x, const Vector& y, double& result, double& time_allreduce,
    bool& isOptimized, rank_type_t rt)
{

    double local_result = 0.0;
    if (rt == GPU)
    {
#ifdef USE_CUDA
        cublasStatus_t t = cublasDdot(cublashandle, n, x.values_d, 1, y.values_d, 1, &local_result);
#endif
    }
    else
    {
#ifdef USE_GRACE
        // Consider replacing with NVPL BLAS dot product
        ComputeDotProductCpu(n, x, y, local_result, isOptimized);
#endif
    }

#ifndef HPCG_NO_MPI
    // Use MPI's reduce function to collect all partial sums
    double t0 = mytimer();
    double global_result = 0.0;
    MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    result = global_result;
    t0 = mytimer() - t0;
    time_allreduce += t0;
#else
    time_allreduce += 0.0;
    result = local_result;
#endif

    return 0;
}