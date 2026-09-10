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

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#ifndef EXPLICIT_KERNELS
#include <cusparse.h>
#endif

// Thrust for coloring
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/cub.cuh>

#include "Cuda.hpp"
#include "SparseMatrix.hpp"
#include "mytimer.hpp"

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#include "Geometry.hpp"
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
extern p2p_comm_mode_t P2P_Mode; // Initialized in src/init.cpp
#ifdef USE_NCCL
extern ncclComm_t Nccl_Comm; // Initialized in src/init.cpp
#endif
#endif


// Runtime configuration parameters (can be overridden by environment variables)
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
KernelConfig g_config;

// Function to initialize configuration from environment variables
void InitKernelConfig() {
    // Set default values
    g_config.SV_UNROLL = 8;
    g_config.MV_UNROLL = 8;
    g_config.SV_BLOCK_SIZE = 64;
    g_config.MV_BLOCK_SIZE = 256;
    g_config.VECTOR_WIDTH = 2;
    g_config.USE_TMA_MV = false;
    g_config.USE_TMA_SV = false;
    
    // Read from environment variables
    const char* env_sv_unroll = std::getenv("SV_UNROLL");
    if (env_sv_unroll) {
        g_config.SV_UNROLL = std::atoi(env_sv_unroll);
    }
    
    const char* env_mv_unroll = std::getenv("MV_UNROLL");
    if (env_mv_unroll) {
        g_config.MV_UNROLL = std::atoi(env_mv_unroll);
    }
    
    const char* env_sv_block_size = std::getenv("SV_BLOCK_SIZE");
    if (env_sv_block_size) {
        g_config.SV_BLOCK_SIZE = std::atoi(env_sv_block_size);
    }
    
    const char* env_mv_block_size = std::getenv("MV_BLOCK_SIZE");
    if (env_mv_block_size) {
        g_config.MV_BLOCK_SIZE = std::atoi(env_mv_block_size);
    }
    
    const char* env_vector_width = std::getenv("VECTOR_WIDTH");
    if (env_vector_width) {
        g_config.VECTOR_WIDTH = std::atoi(env_vector_width);
    }
    
    const char* env_use_tma_mv = std::getenv("USE_TMA_MV");
    if (env_use_tma_mv) {
        g_config.USE_TMA_MV = (std::atoi(env_use_tma_mv) != 0);
    }
    
    const char* env_use_tma_sv = std::getenv("USE_TMA_SV");
    if (env_use_tma_sv) {
        g_config.USE_TMA_SV = (std::atoi(env_use_tma_sv) != 0);
    }

#ifdef INDEX_64
    // TMA column tensor maps are hard-wired to CU_TENSOR_MAP_DATA_TYPE_INT32 and
    // the device-side shared-memory buffers are declared as int[] / int*. With
    // INDEX_64, column indices are int64, so TMA would silently truncate/corrupt
    // them. Disable TMA under INDEX_64 and surface a clear message.
    if (g_config.USE_TMA_MV || g_config.USE_TMA_SV) {
        fprintf(stderr,
            "WARNING: USE_TMA_MV/USE_TMA_SV are not supported with INDEX_64 "
            "(TMA column path is INT32-only). Disabling TMA.\n");
        g_config.USE_TMA_MV = false;
        g_config.USE_TMA_SV = false;
    }
#endif
}

// Note: Legacy defines removed - use g_config for runtime configuration

// Always include TMA headers since TMA usage is now runtime-configurable
#include <cuda/barrier>
#include <cuda/type_traits>
#include <cuda/ptx>
namespace ptx = cuda::ptx;


// Support Atomic Add
__device__ int atomic_add(int* ptr, int val)
{
    return atomicAdd(ptr, val);
}

__device__ long atomic_add(long long* ptr, long long val)
{
    return atomicAdd(reinterpret_cast<long long unsigned*>(ptr), static_cast<long long unsigned>(val));
}

///////// L2 Memory Compression Allocation Support Routines //
cudaError_t setProp(CUmemAllocationProp* prop)
{
    CUdevice currentDevice;
    if (cuCtxGetDevice(&currentDevice) != CUDA_SUCCESS)
    {
        printf("CUDA context not initialized?");
        return cudaErrorInvalidValue;
    }

    int compressionAvailable = 0;
    if (cuDeviceGetAttribute(&compressionAvailable, CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, currentDevice)
        != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    memset(prop, 0, sizeof(CUmemAllocationProp));
    prop->type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop->location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop->location.id = currentDevice;
    if (compressionAvailable)
        prop->allocFlags.compressionType = CU_MEM_ALLOCATION_COMP_GENERIC;
    return cudaSuccess;
}

cudaError_t cudaMallocCompressible(void** adr, size_t size)
{
    CUmemAllocationProp prop = {};
    cudaError_t err = setProp(&prop);
    if (err != cudaSuccess)
        return err;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;
    CUdeviceptr dptr;
    if (cuMemAddressReserve(&dptr, size, 0, 0, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemGenericAllocationHandle allocationHandle;
    if (cuMemCreate(&allocationHandle, size, &prop, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    if (cuMemMap(dptr, size, 0, allocationHandle, 0) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    if (cuMemRelease(allocationHandle) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    CUmemAccessDesc accessDescriptor;
    accessDescriptor.location.id = prop.location.id;
    accessDescriptor.location.type = prop.location.type;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    if (cuMemSetAccess(dptr, size, &accessDescriptor, 1) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;

    *adr = (void*) dptr;
    return cudaSuccess;
}

cudaError_t cudaFreeCompressible(void* ptr, size_t size)
{
    CUmemAllocationProp prop = {};
    cudaError_t err = setProp(&prop);
    if (err != cudaSuccess)
        return err;

    size_t granularity = 0;
    if (cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS)
        return cudaErrorMemoryAllocation;
    size = ((size - 1) / granularity + 1) * granularity;

    if (ptr == nullptr)
        return cudaSuccess;
    if (cuMemUnmap((CUdeviceptr) ptr, size) != CUDA_SUCCESS
        || cuMemAddressFree((CUdeviceptr) ptr, size) != CUDA_SUCCESS)
        return cudaErrorInvalidValue;
    return cudaSuccess;
}

///////// Allocate CUDA Memory data structures //
/*
    This functions estimates the neede memory for SELL L and U matrices
    Helps reduce memory footprint
*/
slice_ptr_t EstimateLUmem(local_int_t n, local_int_t padded_n, local_int_t level, int slice_size) {
    bool power_two = (n & (n - 1)) == 0;
    float divisor = n <= slice_size * 6 ? 1.0 : (power_two? 1.85 : 1.60);
    // Compute the padded storage estimate in 64-bit: padded_n * HPCG_MAX_ROW_LEN
    // (and the estimate itself) exceeds 2^31 for large local problems
    // (e.g. 1024x512x512 -> ~3.9e9), so both the intermediate and the result
    // must be 64-bit or they overflow.
    slice_ptr_t estimated_size = (slice_ptr_t) ((slice_ptr_t) padded_n * HPCG_MAX_ROW_LEN * 1.0f / divisor);
    slice_ptr_t v288x512x512[] = {1057190464, 132276512, 16615072, 2074384};
    slice_ptr_t v296x512x512[] = {1095636608, 136618560, 16967616, 2883872};
    slice_ptr_t* v = n == 288 * 512 * 512 ? v288x512x512
        : n == 296 * 512 * 512            ? v296x512x512
        : nullptr;
    if (v != nullptr)
    {
        if (level == 0)
            estimated_size = v[0];
        else if (level == 1)
            estimated_size = v[1];
        else if (level == 2)
            estimated_size = v[2];
        else if (level == 3)
            estimated_size = v[3];
    }

    // Round up so the estimate is a safe *alignment* as well as a safe size.
    //
    // AllocateMemOptCuda packs L and U into one columns allocation:
    //     sellLPermColumns = gpuAux.columns;
    //     sellUPermColumns = gpuAux.columns + estimated_size;
    // so estimated_size is an element displacement applied to an otherwise
    // cudaMalloc-aligned base. The LDG_V2/LDG3 families gather columns with
    // wide vector loads (v2/v4/v8.u32, i.e. 8/16/32 bytes), which the hardware
    // requires to be naturally aligned. An unaligned displacement therefore
    // misaligns *every* U column load in those kernels, while L (base pointer,
    // aligned) and the separately-cudaMalloc'd value arrays are unaffected.
    //
    // This is not an overrun -- compute-sanitizer reports "Invalid __global__
    // read of size 16 bytes ... is misaligned and is inside the nearest
    // allocation". It is also invisible to the scalar LDG/explicit kernels,
    // whose 4-byte loads only need 4-byte alignment, which is why only the
    // wide-load families ever tripped on it.
    //
    // The raw estimate is padded_n * HPCG_MAX_ROW_LEN / divisor computed in
    // float, so its low bits are essentially arbitrary: at 256^3 level 1 it
    // came out 30607082 == 2 (mod 4), putting U's base 8 bytes off a 16-byte
    // boundary -- W=2 (8-byte) loads still worked, W=4 (16-byte) faulted. The
    // divisor value thus decided alignment by luck, which is why divisor=1.0
    // "fixed" the crash (padded_n * 27 is a multiple of 32) and why the
    // hard-coded sizes above never tripped (all are multiples of 16).
    //
    // Contract: the estimate is a multiple of 8 elements, so the displaced U
    // base is 256-bit (32-byte) aligned for int32 columns (8 * 4 == 32) and
    // 64-byte aligned for int64 columns. That is the widest access the kernels
    // issue (ld.global.v8.u32 / v4.f64), so every column/value load is
    // naturally aligned. This applies to both the explicit path and the
    // cuSPARSE path, which places uColumns at the same displacement.
    // The value arrays get their own cudaMalloc and are aligned already.
    constexpr slice_ptr_t kWideLoadAlign = 8;
    estimated_size = (estimated_size + kWideLoadAlign - 1) / kWideLoadAlign * kWideLoadAlign;

    return estimated_size;
}
/*
    This function allocates GPU device memory needed to setup the probem
    Supports L2 compression for gpuAux.values
*/
void AllocateMemCuda(SparseMatrix& A_in)
{
    SparseMatrix* A = &A_in;
    global_int_t nx = A->geom->nx;
    global_int_t ny = A->geom->ny;
    global_int_t nz = A->geom->nz;

    local_int_t numberOfMgLevels = 4;
    local_int_t slice_size = A->slice_size;
    IndexMode index_mode = A->index_mode; // Propagate the runtime index mode to every MG level.
    CHECK_CUDART(cudaMalloc((void**) &(ranktoId), sizeof(local_int_t) * (A->geom->size + 1)));

    for (int level = 0; level < numberOfMgLevels; ++level)
    {
        A->level = level;
        A->slice_size = slice_size;
        A->index_mode = index_mode;
        local_int_t localNumberOfRows = nx * ny * nz;

        size_t num_blocks = (localNumberOfRows + slice_size - 1) / slice_size;
        size_t paddedRowLen = num_blocks * slice_size;

        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.nnzPerRow), sizeof(local_int_t) * (localNumberOfRows + 1)));
        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.csrLPermOffsets), sizeof(slice_ptr_t) * (localNumberOfRows + 1)));
        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.csrUPermOffsets), sizeof(slice_ptr_t) * (localNumberOfRows + 1)));
        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.map), sizeof(local_int_t) * (localNumberOfRows + 1)));
        CHECK_CUDART(cudaMalloc((void**) &(A->csrExtOffsets), sizeof(slice_ptr_t) * (localNumberOfRows + 1)));
        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.diagonalIdx), sizeof(slice_ptr_t) * localNumberOfRows));
        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.localToGlobalMap), sizeof(global_int_t) * localNumberOfRows));
        CHECK_CUDART(cudaMalloc(&(A->ref2opt), localNumberOfRows * sizeof(local_int_t)));
        CHECK_CUDART(cudaMalloc(&(A->opt2ref), localNumberOfRows * sizeof(local_int_t)));
        CHECK_CUDART(cudaMalloc(&(A->f2cPerm), localNumberOfRows * sizeof(local_int_t)));
        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.f2c), localNumberOfRows * sizeof(local_int_t)));
        CHECK_CUDART(cudaMalloc((void**) &(A->diagonal), sizeof(double) * localNumberOfRows));
        CHECK_CUDART(cudaMalloc((void**) &(A->tempBuffer), sizeof(double) * localNumberOfRows));

        /*
            Size 512x512x288 is the largest int32 local problem size with
                lowest convergence rate. Hard-coded size is used to avoid
                allocate memory in the middle of the Optimization phase.
            Size 512x512x296 is the largest possible int32 local problem
             size.
            Hard-coding is used only when Use_Hpcg_Mem_Reduction is set
             to true in src/main. Other wise we use an estimation div-
             isor for L and U matrices.
        */

        /*Memory Estimation for lower and upper parts*/
        slice_ptr_t estimated_size = EstimateLUmem(localNumberOfRows, (local_int_t) paddedRowLen, level, slice_size);

        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.columns), sizeof(local_int_t) * estimated_size * 2));

        if (!Use_Compression)
            CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.values),
                sizeof(double) * (paddedRowLen * HPCG_MAX_ROW_LEN + slice_size * HPCG_MAX_ROW_LEN)));
        else
            CHECK_CUDART(cudaMallocCompressible((void**) &(A->gpuAux.values),
                sizeof(double) * (paddedRowLen * HPCG_MAX_ROW_LEN + slice_size * HPCG_MAX_ROW_LEN)));

        nx /= 2;
        ny /= 2;
        nz /= 2;
        A->Ac = new SparseMatrix;
        if (level == numberOfMgLevels - 1)
        {
            A->Ac = 0;
        }
        else
        {
            A = A->Ac;
        }
    }
}

// Bytes allocated by one AllocateMemOptCuda multigrid level. Keep in sync with
// AllocateMemOptCuda below (sellAPermValues and sell{L,U}PermColumns alias
// gpuAux buffers from AllocateMemCuda and are not counted here).
static void AccumulateGpuOptMemOptCudaLevel(size_t& opt_mem, local_int_t localNumberOfRows, int slice_size, int level)
{
    local_int_t num_blocks = (localNumberOfRows + slice_size - 1) / slice_size;
    local_int_t paddedRowLen = num_blocks * slice_size;
    slice_ptr_t estimated_size = EstimateLUmem(localNumberOfRows, paddedRowLen, level, slice_size);

    opt_mem += sizeof(local_int_t) * ((size_t) paddedRowLen * HPCG_MAX_ROW_LEN + (size_t) slice_size * HPCG_MAX_ROW_LEN);
    if (Use_Hpcg_Mem_Reduction)
        opt_mem += sizeof(double) * estimated_size;
    else
        opt_mem += 2 * sizeof(double) * estimated_size;

    opt_mem += 3 * sizeof(local_int_t) * (num_blocks + 1);
    opt_mem += localNumberOfRows * sizeof(local_int_t);
    opt_mem += 64 * sizeof(int);

    if (Use_Hpcg_Mem_Reduction && (localNumberOfRows % 8 == 0))
        opt_mem += 2048 + (8 * sizeof(local_int_t) * size_t(localNumberOfRows));
}

/*
    Estimate GPU device memory retained after AllocateMemOptCuda.
*/
size_t EstimateGpuOptMem(const SparseMatrix& A_in)
{
    const SparseMatrix* A = &A_in;
    global_int_t nx = A->geom->nx;
    global_int_t ny = A->geom->ny;
    global_int_t nz = A->geom->nz;

    local_int_t numberOfMgLevels = 4;
    size_t opt_mem = 0;
    for (int level = 0; level < numberOfMgLevels; ++level)
    {
        local_int_t localNumberOfRows = nx * ny * nz;
        AccumulateGpuOptMemOptCudaLevel(opt_mem, localNumberOfRows, A->slice_size, level);

        nx /= 2;
        ny /= 2;
        nz /= 2;
        if (level != numberOfMgLevels - 1)
            A = A->Ac;
    }
    return opt_mem;
}

/*
    This function allocates GPU device memory needed to optimize the problem
    It creates sliced ellpack data structures for the general, lower, and upper
    matrices. It also allocates cusparse spsv
    buffer im memoty reduction mode
    Supports L2 compression for sell_perm_[l,u]_values
*/
void AllocateMemOptCuda(SparseMatrix& A_in)
{
    SparseMatrix* A = &A_in;
    global_int_t nx = A->geom->nx;
    global_int_t ny = A->geom->ny;
    global_int_t nz = A->geom->nz;

    local_int_t numberOfMgLevels = 4;
    for (int level = 0; level < numberOfMgLevels; ++level)
    {
        local_int_t localNumberOfRows = nx * ny * nz;
        int slice_size = A->slice_size;

#ifndef EXPLICIT_KERNELS
        local_int_t num_blocks = (localNumberOfRows + slice_size - 1) / slice_size;
        local_int_t paddedRowLen = num_blocks * slice_size;

        // Index-mode-selected element widths for the SELL offset/column arrays.
        const IndexMode mode = A->index_mode;
        const size_t colBytes = columnIndexBytes(mode);
        const size_t offBytes = offsetIndexBytes(mode);
        const size_t aColElems
            = (size_t) paddedRowLen * HPCG_MAX_ROW_LEN + (size_t) slice_size * HPCG_MAX_ROW_LEN;

        // A operator columns (column-index width), 64-bit size math to avoid overflow.
        CHECK_CUDART(cudaMalloc(&(A->sellDev.aColumns), colBytes * aColElems));
        A->sellAPermValues = A->gpuAux.values; // Use the same space as values

        /*Memory Estimation for lower and upper parts*/
        slice_ptr_t estimated_size = EstimateLUmem(localNumberOfRows, (local_int_t) paddedRowLen, level, slice_size);

        // L/U columns: when columns are 32-bit we reuse gpuAux.columns (int32, unused
        // after SELL creation); when columns are 64-bit that int32 buffer cannot be
        // reused, so we own dedicated buffers instead.
        if (columnsAre64(mode))
        {
            CHECK_CUDART(cudaMalloc(&(A->sellDev.lColumns), colBytes * (size_t) estimated_size));
            CHECK_CUDART(cudaMalloc(&(A->sellDev.uColumns), colBytes * (size_t) estimated_size));
            A->sellDev.ownsLuColumns = true;
        }
        else
        {
            A->sellDev.lColumns = A->gpuAux.columns;
            A->sellDev.uColumns = A->gpuAux.columns + estimated_size;
            A->sellDev.ownsLuColumns = false;
        }
        if (!Use_Compression)
        {
            if (Use_Hpcg_Mem_Reduction)
            {
                CHECK_CUDART(cudaMalloc((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));
        
                // Both matrices have the same values, -1
                A->sellLPermValues = A->sellUPermValues;
            }
            else
            {
                CHECK_CUDART(cudaMalloc((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));
                CHECK_CUDART(cudaMalloc((void**) &(A->sellLPermValues), sizeof(double) * estimated_size));
            }
        }
        else
        {
            if (Use_Hpcg_Mem_Reduction)
            {
                CHECK_CUDART(cudaMalloc((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));

                // Both matrices have the same values, -1
                A->sellLPermValues = A->sellUPermValues;
            }
            else
            {
                CHECK_CUDART(cudaMallocCompressible((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));
                CHECK_CUDART(cudaMallocCompressible((void**) &(A->sellLPermValues), sizeof(double) * estimated_size));
            }
        }

        const size_t numSliceEntries = (size_t) (paddedRowLen / slice_size + 1);
        CHECK_CUDART(cudaMalloc(&(A->sellDev.lSliceOffsets), offBytes * numSliceEntries));
        CHECK_CUDART(cudaMalloc(&(A->sellDev.uSliceOffsets), offBytes * numSliceEntries));
        CHECK_CUDART(cudaMalloc(&(A->sellDev.aSliceOffsets), offBytes * numSliceEntries));

        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.color), localNumberOfRows * sizeof(local_int_t)));
        CHECK_CUDART(cudaMemset(A->gpuAux.color, -1, localNumberOfRows * sizeof(local_int_t)));
        A->gpuAux.colorCountCpu = new int[64];
        for (int i = 0; i < 64; i++)
        {
            A->gpuAux.colorCountCpu[i] = 0;
        }

        // SpSV related memory optimization
        // HPCG estimated buffer size. The cuSPARSE Sliced-ELL SpSV scratch buffer
        // scales with the index element width, so use the mode's offset width
        // (offBytes) rather than sizeof(local_int_t); for the default int32 mode
        // offBytes == sizeof(local_int_t) so legacy sizing is preserved.
        if (Use_Hpcg_Mem_Reduction && (localNumberOfRows % 8 == 0))
        {
            size_t buffer_size_sv_l = 2048 + (8 * offBytes * size_t(localNumberOfRows));
            CHECK_CUDART(cudaMalloc(&A->bufferSvL, buffer_size_sv_l));
            // Same buffer since we they both share the same diagional
            A->bufferSvU = A->bufferSvL;
        }

        nx /= 2;
        ny /= 2;
        nz /= 2;
#else
        size_t num_blocks = (localNumberOfRows + slice_size - 1) / slice_size;
        size_t paddedRowLen = num_blocks * slice_size;

        // Okay We need to find the memory needed
        CHECK_CUDART(cudaMalloc((void**) &(A->sellAPermColumns),
            sizeof(local_int_t) * (paddedRowLen * HPCG_MAX_ROW_LEN + slice_size * HPCG_MAX_ROW_LEN)));
        A->sellAPermValues = A->gpuAux.values; // Use the same space as values

        /*Memory Estimation for lower and upper parts*/
        slice_ptr_t estimated_size = EstimateLUmem(localNumberOfRows, (local_int_t) paddedRowLen, level, slice_size);

        // Reuse columns arrays, not used after we create SELL
        A->sellLPermColumns = A->gpuAux.columns;
        A->sellUPermColumns = A->gpuAux.columns + estimated_size;
        if (!Use_Compression)
        {
            if (Use_Hpcg_Mem_Reduction)
            {
                CHECK_CUDART(cudaMalloc((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));
        
                // Both matrices have the same values, -1
                A->sellLPermValues = A->sellUPermValues;
            }
            else
            {
                CHECK_CUDART(cudaMalloc((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));
                CHECK_CUDART(cudaMalloc((void**) &(A->sellLPermValues), sizeof(double) * estimated_size));
            }
        }
        else
        {
            if (Use_Hpcg_Mem_Reduction)
            {
                CHECK_CUDART(cudaMalloc((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));

                // Both matrices have the same values, -1
                A->sellLPermValues = A->sellUPermValues;
            }
            else
            {
                CHECK_CUDART(cudaMallocCompressible((void**) &(A->sellUPermValues), sizeof(double) * estimated_size));
                CHECK_CUDART(cudaMallocCompressible((void**) &(A->sellLPermValues), sizeof(double) * estimated_size));
            }
        }

        CHECK_CUDART(cudaMalloc((void**) &(A->sellLSliceMrl), sizeof(slice_ptr_t) * (paddedRowLen / slice_size + 1)));
        CHECK_CUDART(cudaMalloc((void**) &(A->sellUSliceMrl), sizeof(slice_ptr_t) * (paddedRowLen / slice_size + 1)));
        CHECK_CUDART(cudaMalloc((void**) &(A->sellASliceMrl), sizeof(slice_ptr_t) * (paddedRowLen / slice_size + 1)));

        CHECK_CUDART(cudaMalloc((void**) &(A->gpuAux.color), localNumberOfRows * sizeof(local_int_t)));
        CHECK_CUDART(cudaMemset(A->gpuAux.color, -1, localNumberOfRows * sizeof(local_int_t)));
        A->gpuAux.colorCountCpu = new int[64];
        for (int i = 0; i < 64; i++)
        {
            A->gpuAux.colorCountCpu[i] = 0;
        }

#ifndef EXPLICIT_KERNELS
        // SpSV related memory optimization
        // HPCG estimated buffer size
        if (Use_Hpcg_Mem_Reduction && (localNumberOfRows % 8 == 0))
        {
            size_t buffer_size_sv_l = 2048 + (8 * sizeof(local_int_t) * size_t(localNumberOfRows));
            CHECK_CUDART(cudaMalloc(&A->bufferSvL, buffer_size_sv_l));
            // Same buffer since we they both share the same diagional
            A->bufferSvU = A->bufferSvL;
        }
#endif

        nx /= 2;
        ny /= 2;
        nz /= 2;
#endif
        if (level == numberOfMgLevels - 1)
        {
        }
        else
        {
            A = A->Ac;
        }
    }
}

/*
    This function deallocates GPU device memory
*/
void DeleteMatrixGpu(SparseMatrix& A)
{
    local_int_t numberOfMgLevels = 4;
    SparseMatrix* AA = &A;
    for (int level = 0; level < numberOfMgLevels; ++level)
    {
#ifndef HPCG_CONTIGUOUS_ARRAYS
        for (local_int_t i = 0; i < AA->localNumberOfRows; ++i)
        {
            delete[] AA->matrixValues[i];
            delete[] AA->mtxIndL[i];
        }
#else
        delete[] AA->matrixValues[0];
        delete[] AA->mtxIndL[0];
#endif
        if (AA->title)
            delete[] AA->title;
        if (AA->nonzerosInRow)
            delete[] AA->nonzerosInRow;

        if (AA->mtxIndL)
            delete[] AA->mtxIndL;
        if (AA->matrixValues)
            delete[] AA->matrixValues;
        if (AA->matrixDiagonal)
            delete[] AA->matrixDiagonal;

#ifndef HPCG_NO_MPI
        if (AA->elementsToSend)
            delete[] AA->elementsToSend;
        if (AA->neighbors)
            delete[] AA->neighbors;

        if (AA->receiveLength)
            delete[] AA->receiveLength;
        if (AA->sendLength)
            delete[] AA->sendLength;
        if (AA->sendBuffer)
            CHECK_CUDART(cudaFreeHost(AA->sendBuffer));
#endif

        if (AA->geom != 0)
        {
            DeleteGeometry(*AA->geom);
            delete AA->geom;
            AA->geom = 0;
        }
        if (AA->mgData != 0)
        {
            DeleteMGData(*AA->mgData);
            delete AA->mgData;
            AA->mgData = 0;
        } // Delete MG data

        // CUDA Free
        int slice_size = AA->slice_size;
        size_t num_blocks = (AA->localNumberOfRows + slice_size - 1) / slice_size;
        size_t paddedRowLen = num_blocks * slice_size;

        CHECK_CUDART(cudaFree(AA->gpuAux.nnzPerRow));
        CHECK_CUDART(cudaFree(AA->gpuAux.csrLPermOffsets));
        CHECK_CUDART(cudaFree(AA->gpuAux.csrUPermOffsets));
        CHECK_CUDART(cudaFree(AA->gpuAux.map));
        CHECK_CUDART(cudaFree(AA->gpuAux.diagonalIdx));
        CHECK_CUDART(cudaFree(AA->gpuAux.localToGlobalMap));
        CHECK_CUDART(cudaFree(AA->ref2opt));
        CHECK_CUDART(cudaFree(AA->opt2ref));
        CHECK_CUDART(cudaFree(AA->f2cPerm));
        CHECK_CUDART(cudaFree(AA->gpuAux.f2c));
        CHECK_CUDART(cudaFree(AA->diagonal));
        CHECK_CUDART(cudaFree(AA->tempBuffer));
        CHECK_CUDART(cudaFree(AA->gpuAux.columns));

        if (!Use_Compression)
            CHECK_CUDART(cudaFree(AA->gpuAux.values));
        else
            CHECK_CUDART(cudaFreeCompressible(AA->gpuAux.values,
                sizeof(double) * (paddedRowLen * HPCG_MAX_ROW_LEN + slice_size * HPCG_MAX_ROW_LEN)));

        CHECK_CUDART(cudaFree(AA->sellDev.aColumns));
        if (AA->sellDev.ownsLuColumns)
        {
            CHECK_CUDART(cudaFree(AA->sellDev.lColumns));
            CHECK_CUDART(cudaFree(AA->sellDev.uColumns));
        }

        if (!Use_Compression)
        {
            if (Use_Hpcg_Mem_Reduction)
            {
                CHECK_CUDART(cudaFree(AA->sellLPermValues));
            }
            else
            {
                CHECK_CUDART(cudaFree(AA->sellLPermValues));
                CHECK_CUDART(cudaFree(AA->sellUPermValues));
            }
        }
        else
        {
            slice_ptr_t estimated_size = EstimateLUmem(AA->localNumberOfRows, (local_int_t) paddedRowLen, level, slice_size);
            if (Use_Hpcg_Mem_Reduction)
            {
                CHECK_CUDART(cudaFreeCompressible(AA->sellLPermValues, sizeof(double) * estimated_size));
            }
            else
            {
                CHECK_CUDART(cudaFreeCompressible(AA->sellLPermValues, sizeof(double) * estimated_size));
                CHECK_CUDART(cudaFreeCompressible(AA->sellUPermValues, sizeof(double) * estimated_size));
            }
        }

        CHECK_CUDART(cudaFree(AA->sellDev.lSliceOffsets));
        CHECK_CUDART(cudaFree(AA->sellDev.uSliceOffsets));
        CHECK_CUDART(cudaFree(AA->sellDev.aSliceOffsets));

#ifndef EXPLICIT_KERNELS
        if (AA->cusparseOpt.vecX)
            CHECK_CUSPARSE(cusparseDestroyDnVec(AA->cusparseOpt.vecX));
        if (AA->cusparseOpt.vecY)
            CHECK_CUSPARSE(cusparseDestroyDnVec(AA->cusparseOpt.vecY));

        CHECK_CUSPARSE(cusparseDestroySpMat(AA->cusparseOpt.matA));
        CHECK_CUSPARSE(cusparseDestroySpMat(AA->cusparseOpt.matL));
        CHECK_CUSPARSE(cusparseDestroySpMat(AA->cusparseOpt.matU));
#endif

        CHECK_CUDART(cudaFree(AA->csrExtOffsets));
        CHECK_CUDART(cudaFree(AA->csrExtColumns));
        CHECK_CUDART(cudaFree(AA->csrExtValues));

        CHECK_CUDART(cudaFree(AA->gpuAux.color));
        delete[] AA->gpuAux.colorCountCpu;

#ifndef EXPLICIT_KERNELS
        CHECK_CUDART(cudaFree(AA->bufferSvL));
        if (!Use_Hpcg_Mem_Reduction || AA->localNumberOfRows % 8 != 0)
            CHECK_CUDART(cudaFree(AA->bufferSvU));
#endif

#ifndef HPCG_NO_MPI
        if (P2P_Mode == MPI_GPU_All2allv || P2P_Mode == MPI_CPU_All2allv)
        {
            if (A.scounts)
                delete[] AA->scounts;
            if (A.rcounts)
                delete[] AA->rcounts;
            if (A.sdispls)
                delete[] AA->sdispls;
            if (A.rdispls)
                delete[] AA->rdispls;
        }
#endif

        AA = AA->Ac;
    }
}


///////// Genrerate Problem //
#define FULL_MASK 0xffffffff

/*
    Translation of a 3D coordinate in all directions
    27 possible neighbor
*/
__device__ char4 tid2ind[32] = {{-1, -1, -1, 0}, {0, -1, -1, 0}, {1, -1, -1, 0}, {-1, 0, -1, 0}, {0, 0, -1, 0},
    {1, 0, -1, 0}, {-1, 1, -1, 0}, {0, 1, -1, 0}, {1, 1, -1, 0}, {-1, -1, 0, 0}, {0, -1, 0, 0}, {1, -1, 0, 0},
    {-1, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}, {-1, 1, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}, {-1, -1, 1, 0}, {0, -1, 1, 0},
    {1, -1, 1, 0}, {-1, 0, 1, 0}, {0, 0, 1, 0}, {1, 0, 1, 0}, {-1, 1, 1, 0}, {0, 1, 1, 0}, {1, 1, 1, 0}, {0, 0, 0, 0},
    {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

/*
    GPU Kernel
    Sets an array values to minus one
*/
__global__ void __launch_bounds__(128) setMinusOne_kernel(size_t count, double* arr)
{
    const size_t i = (size_t) blockIdx.x * 128 + threadIdx.x;
    if (i < count)
        arr[i] = -1.0;
}

/*
    GPU Device Function
    Shifts a 64 bit value form one thread in a warp to the remaining
    threads. First it divides the 64-bit value to lower and upper
    parts. Then sends the lower and upper parts seperately from src
    to other threads
*/
__device__ __inline__ double shfl64_device(long long int x, int src)
{

    int lo, hi;
    asm volatile("mov.b64 {%0,%1}, %2;" : "=r"(lo), "=r"(hi) : "l"(x));
    lo = __shfl_sync(FULL_MASK, lo, src);
    hi = __shfl_sync(FULL_MASK, hi, src);
    asm volatile("mov.b64 %0, {%1,%2};" : "=l"(x) : "r"(lo), "r"(hi));
    return x;
}

/*
    GPU Kernel
    Prefix sum to find extenal row offset
    Finds the map from compressed external row id
    to original row id
    **Note** The generated external matrix is compressed
    to have rows with external nnz only, empty rows are
    skipped
*/
template <int THREADS_PER_CTA, int GRIDX>
__global__ void __launch_bounds__(THREADS_PER_CTA) compressCsrOffsets_kernel(local_int_t localNumberOfRows,
    slice_ptr_t* csr_offsets, local_int_t* map, slice_ptr_t* tmp_offsets, int* temp, slice_ptr_t* nnz_per_row, int rank)
{

    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x * THREADS_PER_CTA + tidx;
    const int str = (int64_t(bidx + 0) * int64_t(localNumberOfRows + 1)) / (THREADS_PER_CTA * GRIDX);
    const int end = (int64_t(bidx + 1) * int64_t(localNumberOfRows + 1)) / (THREADS_PER_CTA * GRIDX) - 1;
    nnz_per_row[0] = 0;
    int j = 0;
    if (str <= end)
        tmp_offsets[str] = nnz_per_row[str];
    for (local_int_t i = str; i < end; i++)
    {
        if (nnz_per_row[i + 1] > 0)
            j++;
        tmp_offsets[i + 1] = nnz_per_row[i + 1] + tmp_offsets[i];
    }
    if (str <= end && end < localNumberOfRows)
    {
        if (nnz_per_row[end + 1] > 0)
            j++;
    }

    temp[bidx] = j;
    __syncthreads();
    if (tidx == 0)
    {
        __threadfence();
        atomicAdd(temp + THREADS_PER_CTA * GRIDX, 1);
        while (1)
        {
            if (((volatile int*) temp)[THREADS_PER_CTA * GRIDX] >= GRIDX)
                break;
        }
        __threadfence();
    }
    __syncthreads();

    slice_ptr_t tmp = 0;
    int map_str = 0;
    for (int i = 1; i < bidx + 1; i++)
    {
        const int64_t ptr1 = ((int64_t) (i + 0) * (localNumberOfRows + 1)) / (THREADS_PER_CTA * GRIDX);
        const int64_t ptr2 = ((int64_t) (i + 1) * (localNumberOfRows + 1)) / (THREADS_PER_CTA * GRIDX);
        map_str += temp[i - 1];
        if (ptr1 == 0 || ptr1 >= ptr2)
            continue;
        tmp += tmp_offsets[ptr1 - 1];
    }
    __syncthreads();
    if (tidx == 0)
    {
        __threadfence();
        atomicAdd(temp + THREADS_PER_CTA * GRIDX, 1);
        while (1)
        {
            if (((volatile int*) temp)[THREADS_PER_CTA * GRIDX] >= 2 * GRIDX)
                break;
        }
        __threadfence();
    }
    __syncthreads();
    j = 0;
    for (local_int_t i = str; i < end + 1; i++)
    {
        tmp_offsets[i] += tmp;
        if (i < localNumberOfRows)
        {
            if (nnz_per_row[i + 1] > 0)
            {
                csr_offsets[map_str + j] = tmp_offsets[i];
                map[map_str + j] = i;
                j++;
            }
        }
    }
    if (bidx == (THREADS_PER_CTA * GRIDX - 1))
    {
        csr_offsets[map_str + temp[bidx]] = tmp_offsets[localNumberOfRows];
        map[localNumberOfRows] = map_str + temp[bidx];
    }
}

/*
    GPU Kernel
    Generates the HPCG problem:
        Finds internal and external nnz per row
        position of diagonal
        localToGlobalMap
        Sets the neighbors of the current rank to one, this would help
        find the logical order of neighbors (rankToId)
    Sets matrix_values of diagonal to 26.0
*/
__global__ void __launch_bounds__(128) generateProblem_kernel(int rank, int partition_by, local_int_t npx,
    local_int_t npy, local_int_t nx, local_int_t ny, local_int_t nz, local_int_t gnx, local_int_t gny, local_int_t gnz,
    local_int_t gix0, local_int_t giy0, local_int_t giz0, local_int_t* csr_offsets, local_int_t* columns,
    double* values, double* bv, double* xv, double* ev, slice_ptr_t* diagonalIdx, double* diagonal,
    slice_ptr_t* csrExtOffsets, global_int_t* localToGlobalMap, bool update, int* rankToId)
{

    extern __shared__ int shdiag[];
    int* shd = shdiag + (threadIdx.x & (~31));

    const int lrow = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = lrow / 32;
    const int lid = lrow % 32;

    int ntot = nx * ny * nz;

    if (wid * 32 >= ntot)
        return;

    const int iz = lrow / (nx * ny);
    const int iy = (lrow - iz * nx * ny) / nx;
    const int ix = lrow - (iz * ny + iy) * nx;

    const int ipz0 = rank / (npx * npy);
    const int ipy0 = (rank - ipz0 * npx * npy) / npx;
    const int ipx0 = rank - (ipz0 * npy + ipy0) * npx;

    const long long int gix = gix0 + ix;
    const long long int giy = giy0 + iy;
    const long long int giz = giz0 + iz;

    local_int_t nnz = 0;
    local_int_t nnz_ext = 0;
    char4 disp = tid2ind[lid];
    const global_int_t currentGlobalRow = giz * gnx * gny + giy * gnx + gix;
    for (int i = 0; i < min(32, ntot - wid * 32); i++)
    {
        long long int cgix = shfl64_device(gix, i) + disp.x;
        long long int cgiy = shfl64_device(giy, i) + disp.y;
        long long int cgiz = shfl64_device(giz, i) + disp.z;

        int ok
            = cgiz > -1 && cgiz < gnz && cgiy > -1 && cgiy < gny && cgix > -1 && cgix < gnx && lid < HPCG_MAX_ROW_LEN;
        ////////////////////////////////////////////////////////
        int ipz = cgiz / nz;
        int ipy = cgiy / ny;
        int ipx = cgix / nx;

        if (partition_by == 2 /*Z*/)
        {
            long long int local = cgiz - giz0;
            if (local >= 0 && local < nz)
                ipz = ipz0;
            else if (local < 0)
                ipz = ipz0 - 1;
            else if (local >= nz)
                ipz = ipz0 + 1;
        }
        else if (partition_by == 1 /*Y*/)
        {
            long long int local = cgiy - giy0;
            if (local >= 0 && local < ny)
                ipy = ipy0;
            else if (local < 0)
                ipy = ipy0 - 1;
            else if (local >= ny)
                ipy = ipy0 + 1;
        }
        else if (partition_by == 0 /*X*/)
        {
            long long int local = cgix - gix0;
            if (local >= 0 && local < nx)
                ipx = ipx0;
            else if (local < 0)
                ipx = ipx0 - 1;
            else if (local >= nx)
                ipx = ipx0 + 1;
        }
        int col_rank = ipx + ipy * npx + ipz * npy * npx;
        int int_msk = __ballot_sync(FULL_MASK, ok && (rank == col_rank));
        int ext_msk = __ballot_sync(FULL_MASK, ok && (rank != col_rank));

        if (lid == i)
            nnz = __popc(int_msk | ext_msk);
        if (lid == i)
            nnz_ext = __popc(ext_msk);

        int ipos = __popc(int_msk & ((1 << lid) - 1));
        if (ok)
        {
            if (lid == 13)
                shd[i] = ipos;
            if (update && rank != col_rank)
                rankToId[col_rank] = 1;
        }
    }

    __syncthreads();
    if (lrow >= ntot)
        return;

    slice_ptr_t in_id = (slice_ptr_t) shd[lid] + (slice_ptr_t) lrow * HPCG_MAX_ROW_LEN;

    diagonalIdx[lrow] = in_id;
    
    localToGlobalMap[lrow] = currentGlobalRow;
    diagonal[lrow] = 26.0;
    csr_offsets[lrow] = nnz;
    csrExtOffsets[lrow + 1] = nnz_ext;
    atomic_add(&csr_offsets[ntot], nnz);

    
    values[in_id] = 26.0;
    if (bv != NULL)
        bv[lrow] = 26.0 - ((double) (nnz - 1));
    if (xv != NULL)
        xv[lrow] = 0.0;
    if (ev != NULL)
        ev[lrow] = 1.0;
    return;
}

/*
    Transforms neighbors ranks to sequential ids --> reduces memory allocation
    Calls generateProblem_kernel, compressCsrOffsets_kernel
    Allocates and creates the external matrix on GPU
*/
void GenerateProblemCuda(SparseMatrix& A, Vector* b, Vector* x, Vector* xexact)
{
    global_int_t npx = A.geom->npx;
    global_int_t npy = A.geom->npy;
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

    int* temp = NULL;
    size_t temp_storage_bytes = 0;
    if (A.level == 0)
        cub::DeviceScan::InclusiveSum(temp, temp_storage_bytes, ranktoId, ranktoId, A.geom->size);

    size_t max_size = max(sizeof(int) * (128 * 64 + 1), temp_storage_bytes);
    CHECK_CUDART(cudaMalloc(&temp, max_size));

    local_int_t n = nx * ny * nz;
    dim3 block2(128, 1, 1);
    dim3 grid2((n + block2.x - 1) / block2.x, 1, 1);

    double* bv = b != 0 ? b->values_d : NULL;
    double* xv = x != 0 ? x->values_d : NULL;
    double* ev = xexact != 0 ? xexact->values_d : NULL;

    // Generete nnzPerRow
    CHECK_CUDART(cudaMemsetAsync(&(A.gpuAux.nnzPerRow[localNumberOfRows]), 0, sizeof(local_int_t), stream));
    const size_t total_nnz = (size_t) localNumberOfRows * HPCG_MAX_ROW_LEN;
    const size_t grid_nnz = (total_nnz + 128 - 1) / 128;
    setMinusOne_kernel<<<grid_nnz, 128, 0, stream>>>(total_nnz, A.gpuAux.values);
    generateProblem_kernel<<<grid2, block2, block2.x * sizeof(int), stream>>>(A.geom->logical_rank,
        A.geom->different_dim, npx, npy, nx, ny, nz, gnx, gny, gnz, gix0, giy0, giz0, A.gpuAux.nnzPerRow,
        A.gpuAux.columns, A.gpuAux.values, bv, xv, ev, A.gpuAux.diagonalIdx, A.diagonal, A.csrExtOffsets,
        A.gpuAux.localToGlobalMap, A.level == 0, ranktoId);

    CHECK_CUDART(cudaMemcpy(A.gpuAux.csrUPermOffsets, A.csrExtOffsets, sizeof(slice_ptr_t) * (localNumberOfRows + 1),
        cudaMemcpyDeviceToDevice));
    CHECK_CUDART(cudaMemsetAsync(&(temp[64 * 128]), 0, sizeof(int), stream));
    compressCsrOffsets_kernel<128, 64><<<64, 128, 0, stream>>>(localNumberOfRows, A.csrExtOffsets, A.gpuAux.map,
        A.gpuAux.csrLPermOffsets, temp, A.gpuAux.csrUPermOffsets, A.geom->logical_rank);
    CHECK_CUDART(cudaMemcpy(&(A.gpuAux.compressNumberOfRows), &(A.gpuAux.map[localNumberOfRows]), sizeof(local_int_t),
        cudaMemcpyDeviceToHost));

    A.extNnz = 0;
    slice_ptr_t extNnz_tmp = 0;
    CHECK_CUDART(cudaMemcpy(
        &extNnz_tmp, &(A.csrExtOffsets[A.gpuAux.compressNumberOfRows]), sizeof(slice_ptr_t), cudaMemcpyDeviceToHost));
    A.extNnz = extNnz_tmp;
    // Device nnzPerRow scan is still int32; host field is slice_ptr_t for --mi nnz > 2^31.
    // OptimizeProblemCuda derives true nnz from 64-bit CSR offsets when this wraps.
    local_int_t localNumberOfNonzeros32 = 0;
    CHECK_CUDART(cudaMemcpy(&localNumberOfNonzeros32, &(A.gpuAux.nnzPerRow[localNumberOfRows]), sizeof(local_int_t),
        cudaMemcpyDeviceToHost));

    CHECK_CUDART(cudaMalloc((void**) &(A.csrExtColumns), sizeof(local_int_t) * A.extNnz));
    CHECK_CUDART(cudaMalloc((void**) &(A.csrExtValues), sizeof(double) * A.extNnz));
    CHECK_CUDART(cudaMalloc((void**) &(A.gpuAux.ext2csrOffsets), sizeof(slice_ptr_t) * A.extNnz));

    if (A.level == 0)
        cub::DeviceScan::InclusiveSum(temp, temp_storage_bytes, ranktoId, ranktoId, A.geom->size);

    A.localNumberOfNonzeros = (slice_ptr_t) localNumberOfNonzeros32;
    CHECK_CUDART(cudaFree(temp));
}

///////// Setup Halo //
/*
    GPU Kernel
    Generates column indices and stores them in gpuAux.columns
    Finds send buffer indices

    **Note** For external column indices, the -rank - 1 is stored,
    and the real column index is stored in csrExtColumns
    This trick helps find the rank of each external col index
    fast
*/
__global__ void __launch_bounds__(128) setupHalo_kernel(int rank, int partition_by, int pnd /*previous neighbor dim
                                                                                             */
    ,
    int nnd /*next neighbor dim*/, local_int_t npx, local_int_t npy, local_int_t nx, local_int_t ny, local_int_t nz,
    local_int_t gnx, local_int_t gny, local_int_t gnz, local_int_t gix0, local_int_t giy0, local_int_t giz0,
    local_int_t* csr_offsets, local_int_t* columns, double* values, slice_ptr_t* diagonalIdx, double* diagonal,
    slice_ptr_t* csrExtOffsets, local_int_t* csrExtColumns, slice_ptr_t* ext2csrOffsets, global_int_t* localToGlobalMap,
    local_int_t* f2c, local_int_t sendbufld, local_int_t* sendcnt, local_int_t* sendbuf, int* rankToId)
{

    __shared__ int rank_row[128][HPCG_MAX_ROW_LEN];
    const local_int_t WARPSIZE = 32;

    const local_int_t lrow = blockIdx.x * blockDim.x + threadIdx.x;
    const int wid = lrow / WARPSIZE;
    const int lid = lrow % WARPSIZE;
    const local_int_t ntot = nx * ny * nz;
    if (wid * WARPSIZE >= ntot)
        return;

    const int iz = lrow / (nx * ny);
    const int iy = (lrow - iz * nx * ny) / nx;
    const int ix = lrow - (iz * ny + iy) * nx;

    const int ipz0 = rank / (npx * npy);
    const int ipy0 = (rank - ipz0 * npx * npy) / npx;
    const int ipx0 = rank - (ipz0 * npy + ipy0) * npx;

    const long long int gix = gix0 + ix;
    const long long int giy = giy0 + iy;
    const long long int giz = giz0 + iz;

    char4 disp = tid2ind[lid];

    global_int_t offset = (global_int_t) wid * WARPSIZE * HPCG_MAX_ROW_LEN;

    csrExtColumns += csrExtOffsets[wid * WARPSIZE];
    ext2csrOffsets += csrExtOffsets[wid * WARPSIZE];

    const int wx = threadIdx.x / WARPSIZE;
    if (iz % 2 == 0 && iy % 2 == 0 && iz % 2 == 0)
    {
        local_int_t currentCoarseRow = (iz * nx * ny) / 8 + (iy * nx) / 4 + ix / 2;
        f2c[currentCoarseRow] = lrow;
    }

    for (int i = 0; i < HPCG_MAX_ROW_LEN; i++)
        rank_row[threadIdx.x][i] = 0;

    columns += offset;
    for (size_t i = 0; i < min(WARPSIZE, ntot - wid * WARPSIZE); i++)
    {
        long long int cgix = shfl64_device(gix, i) + disp.x;
        long long int cgiy = shfl64_device(giy, i) + disp.y;
        long long int cgiz = shfl64_device(giz, i) + disp.z;

        int ok
            = cgiz > -1 && cgiz < gnz && cgiy > -1 && cgiy < gny && cgix > -1 && cgix < gnx && lid < HPCG_MAX_ROW_LEN;
        ////////////////////////////////////////////////////////
        int ipz = cgiz / nz;
        int ipy = cgiy / ny;
        int ipx = cgix / nx;

        local_int_t zi = (cgiz) % nz;
        local_int_t yi = (cgiy) % ny;
        local_int_t xi = (cgix) % nx;

        global_int_t new_nx = nx;
        global_int_t new_ny = ny;

        if (partition_by == 2 /*Z*/)
        {
            long long int local = cgiz - giz0;
            if (local >= 0 && local < nz)
            {
                ipz = ipz0;
                zi = local;
            }
            else if (local < 0)
            {
                ipz = ipz0 - 1;
                zi = pnd - 1;
            }
            else if (local >= nz)
            {
                ipz = ipz0 + 1;
                zi = 0;
            }
        }
        else if (partition_by == 1 /*Y*/)
        {
            long long int local = cgiy - giy0;
            if (local >= 0 && local < ny)
            {
                ipy = ipy0;
                yi = local;
            }
            else if (local < 0)
            {
                ipy = ipy0 - 1;
                yi = pnd - 1;
                new_ny = pnd;
            }
            else if (local >= ny)
            {
                ipy = ipy0 + 1;
                yi = 0;
                new_ny = nnd;
            }
        }
        else if (partition_by == 0 /*X*/)
        {
            long long int local = cgix - gix0;
            if (local >= 0 && local < nx)
            {
                ipx = ipx0;
                xi = local;
            }
            else if (local < 0)
            {
                ipx = ipx0 - 1;
                xi = pnd - 1;
                new_nx = pnd;
            }
            else if (local >= nx)
            {
                ipx = ipx0 + 1;
                xi = 0;
                new_nx = nnd;
            }
        }
        int col_rank = ipx + ipy * npx + ipz * npy * npx;
        local_int_t lcol = zi * new_ny * new_nx + yi * new_nx + xi;
        ////////////////////////////////////////////////////////
        int int_msk = __ballot_sync(FULL_MASK, ok && (rank == col_rank));
        int ext_msk = __ballot_sync(FULL_MASK, ok && (rank != col_rank));

        int int_nnz = __popc(int_msk);
        int ext_nnz = __popc(ext_msk);

        int ipos = __popc(int_msk & ((1 << lid) - 1));
        int xpos = __popc(ext_msk & ((1 << lid) - 1));

        if (ok)
        {
            if (rank != col_rank)
            {
                columns[xpos + int_nnz] = -col_rank - 1;
                csrExtColumns[xpos] = lcol;
                ext2csrOffsets[xpos] = offset + xpos + int_nnz; // pos;

                // Here we create rankToId to avoid creating space to each rank,
                //  Instead, each neighbor rank has now a sequential Id
                rank_row[wx * WARPSIZE + i][rankToId[col_rank] - 1] = 0x2C1 /*hash/magic number*/;
            }
            else if ((rank == col_rank))
            {
                columns[ipos] = lcol;
            }
        }

        columns += HPCG_MAX_ROW_LEN;
        csrExtColumns += ext_nnz;
        ext2csrOffsets += ext_nnz;
        offset += HPCG_MAX_ROW_LEN;
    }

    __syncthreads();

    const local_int_t one = 1;
    if (lrow < ntot)
    {
        for (int i = 0; i < HPCG_MAX_ROW_LEN; i++)
            if (rank_row[threadIdx.x][i] == 0x2C1)
                sendbuf[i * sendbufld + atomic_add(&sendcnt[i], one)] = lrow;
    }
    return;
}

/*
    GPU Kernel
    This is kernel is called for each neighbor
    Stores the sequential index of each external column id
    We add localNumberOfRows to each index to know it is
    an external column index
*/
__global__ void __launch_bounds__(128) extToLocMap_kernel(
    local_int_t localNumberOfRows, local_int_t str, local_int_t end, local_int_t* extToLocMap, local_int_t* eltsToRecv)
{

    const local_int_t tidx = blockIdx.x * 128 + threadIdx.x;
    const local_int_t i = tidx + str;
    if (i >= end)
        return;

    const local_int_t col = eltsToRecv[i];
    extToLocMap[col] = localNumberOfRows + i;
}

/*
    GPU Kernel
    external column indices are corrected for gpuAux.columns
        the margin of localNumberOfRows is kept to know
        it external (these indices will be removed in
        OptimizeProblem)
    csrExtColumns has the correct external column
        indices

    Sets the extranl values to -1.0
*/
__global__ void __launch_bounds__(128)
    extToloc_kernel(local_int_t localNumberOfRows, int neighborId, slice_ptr_t ext_nnz, local_int_t* csrExtColumns,
        double* csrExtValues, slice_ptr_t* ext2csrOffsets, local_int_t* extToLocMap, local_int_t* columns)
{

    const slice_ptr_t i = (slice_ptr_t) blockIdx.x * 128 + threadIdx.x;
    if (i >= ext_nnz)
        return;

    const global_int_t col = csrExtColumns[i];
    const slice_ptr_t off = ext2csrOffsets[i]; // 64-bit: flat column offset can exceed INT_MAX at 512^3
    const local_int_t rankIdOfColumnEntry = -columns[off] - 1;
    if (neighborId == rankIdOfColumnEntry)
    {
        columns[off] = extToLocMap[col];
        csrExtColumns[i] = extToLocMap[col] - localNumberOfRows;
        csrExtValues[i] = -1.0;
    }
}

/*
    Setups the halo region
    Finds the total to send
    Calls setupHalo_kernel
*/
void SetupHaloCuda(SparseMatrix& A, local_int_t sendbufld, local_int_t* sendlen, local_int_t* sendbuff,
    local_int_t* tot_to_send, int* nneighs, int* neighs_h, local_int_t* sendlen_h, local_int_t** elem_to_send_d)
{
    global_int_t npx = A.geom->npx;
    global_int_t npy = A.geom->npy;
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

    local_int_t n = nx * ny * nz;
    dim3 block2(128, 1, 1);
    dim3 grid2((n + block2.x - 1) / block2.x, 1, 1);

    // USE csrLPermOffsets as temporal array only!
    CHECK_CUDART(cudaMemsetAsync(A.gpuAux.f2c, 0, sizeof(local_int_t) * localNumberOfRows, stream));

    setupHalo_kernel<<<grid2, block2, 0, stream>>>(A.geom->logical_rank, A.geom->different_dim,
        A.geom->previous_neighbor_dim, A.geom->next_neighbor_dim, npx, npy, nx, ny, nz, gnx, gny, gnz, gix0, giy0, giz0,
        A.gpuAux.nnzPerRow, A.gpuAux.columns, A.gpuAux.values, A.gpuAux.diagonalIdx, A.diagonal,
        A.gpuAux.csrLPermOffsets, A.csrExtColumns, A.gpuAux.ext2csrOffsets, A.gpuAux.localToGlobalMap, A.gpuAux.f2c,
        sendbufld, sendlen, sendbuff, ranktoId);

    if (A.level == 0)
    {
        rankToId_h = new int[A.geom->size];
        CHECK_CUDART(cudaMemcpy(rankToId_h, ranktoId, A.geom->size * sizeof(int), cudaMemcpyDeviceToHost));
        idToRank_h = new int[rankToId_h[A.geom->size - 1]];
    }

    nneighs[0] = rankToId_h[A.geom->size - 1];

    int counter = 1;
    for (int i = 0; i < A.geom->size; i++)
    {
        if (rankToId_h[i] == counter)
        {
            idToRank_h[counter - 1] = i;
            counter++;
        }
    }

    CHECK_CUDART(cudaMemcpy(sendlen_h, sendlen, HPCG_MAX_ROW_LEN * sizeof(local_int_t), cudaMemcpyDeviceToHost));
    tot_to_send[0] = 0;
    for (int i = 0; i < HPCG_MAX_ROW_LEN; i++)
    {
        tot_to_send[0] += sendlen_h[i];
    }

    if (tot_to_send[0])
    {
        CHECK_CUDART(cudaMalloc(elem_to_send_d, tot_to_send[0] * sizeof(local_int_t)));
        tot_to_send[0] = 0;
        for (int i = 0; i < nneighs[0]; i++)
        {
            sendlen_h[i] = sendlen_h[i];
            neighs_h[i] = idToRank_h[i];

            // Compreess
            CHECK_CUDART(cudaMemcpy(elem_to_send_d[0] + tot_to_send[0], sendbuff + i * sendbufld,
                sendlen_h[i] * sizeof(local_int_t), cudaMemcpyDeviceToDevice));
            thrust::device_ptr<local_int_t> keys(elem_to_send_d[0] + tot_to_send[0]);
            thrust::sort(keys, keys + sendlen_h[i]);

            tot_to_send[0] += sendlen_h[i];
        }
    }

    return;
}

/*
    Calls extToloc_kernel
*/
void ExtToLocMapCuda(
    local_int_t localNumberOfRows, local_int_t str, local_int_t end, local_int_t* extToLocMap, local_int_t* eltsToRecv)
{
    const int grid = (end - str + 128 - 1) / 128;
    extToLocMap_kernel<<<grid, 128, 0, stream>>>(localNumberOfRows, str, end, extToLocMap, eltsToRecv);
}

/*
    Calls extToLoc_kernel
*/
void ExtTolocCuda(local_int_t localNumberOfRows, int neighborId, slice_ptr_t ext_nnz, local_int_t* csrExtColumns,
    double* csrExtValues, slice_ptr_t* ext2csrOffsets, local_int_t* extToLocMap, local_int_t* columns)
{

    const int grid = (int) ((ext_nnz + 127) / 128);
    extToloc_kernel<<<grid, 128, 0, stream>>>(
        localNumberOfRows, neighborId, ext_nnz, csrExtColumns, csrExtValues, ext2csrOffsets, extToLocMap, columns);
}

#ifndef HPCG_NO_MPI
/*
    GPU Kernel
    Gathers x values to send to neighbors
*/
__global__ void __launch_bounds__(128)
    sendbuf_kernel(local_int_t totalToBeSent, double* sendBuffer, double* xv, local_int_t* elementsToSend)
{

    const local_int_t i = blockIdx.x * 128 + threadIdx.x;
    if (i < totalToBeSent)
    {
        sendBuffer[i] = xv[elementsToSend[i]];
    }
}

/*
    Calls sendbuf_kernel
    Transfers the gathered buffer to CPU, when USE_CPU_MPI is defined
*/
void PackSendBufferCuda(const SparseMatrix& A, Vector& x, bool cpu_data, cudaStream_t stream1)
{
    if (A.totalToBeSent > 0)
    {
        const int grid = (A.totalToBeSent + 128 - 1) / 128;
        sendbuf_kernel<<<grid, 128, 0, stream1>>>(
            A.totalToBeSent, A.gpuAux.sendBuffer, x.values_d, A.gpuAux.elementsToSend);

        if (P2P_Mode == MPI_CPU || P2P_Mode == MPI_CPU_All2allv)
        {
            CHECK_CUDART(cudaMemcpyAsync(
                A.sendBuffer, A.gpuAux.sendBuffer, A.totalToBeSent * sizeof(double), cudaMemcpyDeviceToHost, stream1));
            CHECK_CUDART(cudaEventRecord(copy_done, stream1));
        }
    }
}

/*
    After the scattred x buffer is received, send and recieve from neighbors
    Supports different P2P modes based on the communication method defined by
    --p2p parameter
    [Experimental/Deactivated] A smart trick to improve MPI_Allreduce in DDOT, 
    by calling MPI_Ibarrier once at the last routine call in MG.
*/
void ExchangeHaloCuda(const SparseMatrix& A, Vector& x, cudaStream_t stream1, int use_ibarrier)
{
    local_int_t localNumberOfRows = A.localNumberOfRows;
    int num_neighbors = A.numberOfSendNeighbors;
    local_int_t* receiveLength = A.receiveLength;
    local_int_t* sendLength = A.sendLength;
    int* neighbors = A.neighborsPhysical;

    if (P2P_Mode == MPI_CPU)
    {
        double* const xv = x.values;
        double* sendBuffer = A.sendBuffer;
        double* x_external = (double*) xv + localNumberOfRows;

        int MPI_MY_TAG = 99;
        MPI_Request* request = new MPI_Request[num_neighbors + 1];

        for (int i = 0; i < num_neighbors; i++)
        {
            local_int_t n_recv = receiveLength[i];
            MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request + i);
            x_external += n_recv;
        }

        CHECK_CUDART(cudaEventSynchronize(copy_done));
        for (int i = 0; i < num_neighbors; i++)
        {
            local_int_t n_send = sendLength[i];
            MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
            sendBuffer += n_send;
        }

        MPI_Waitall(num_neighbors, request, MPI_STATUSES_IGNORE);

        //[Experimental] Can improve MPI_Allreduce performance
        #if 0
        if (use_ibarrier == 1)
            MPI_Ibarrier(MPI_COMM_WORLD, request);
        #endif

        CHECK_CUDART(cudaMemcpyAsync(x.values_d + A.localNumberOfRows, x.values + A.localNumberOfRows,
            A.numberOfExternalValues * sizeof(double), cudaMemcpyHostToDevice, copy_stream));
        CHECK_CUDART(cudaEventRecord(copy_done, copy_stream));
        CHECK_CUDART(cudaStreamWaitEvent(0, copy_done, 0));
        delete[] request;
    }
    else if (P2P_Mode == MPI_CUDA_AWARE)
    {
        double* const xv = x.values_d;
        double* sendBuffer = A.gpuAux.sendBuffer;
        double* x_external = (double*) xv + localNumberOfRows;

        int MPI_MY_TAG = 99;
        MPI_Request* request = new MPI_Request[num_neighbors + 1];

        for (int i = 0; i < num_neighbors; i++)
        {
            local_int_t n_recv = receiveLength[i];
            MPI_Irecv(x_external, n_recv, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD, request + i);
            x_external += n_recv;
        }

        CHECK_CUDART(cudaStreamSynchronize(stream1));
        for (int i = 0; i < num_neighbors; i++)
        {
            local_int_t n_send = sendLength[i];
            MPI_Send(sendBuffer, n_send, MPI_DOUBLE, neighbors[i], MPI_MY_TAG, MPI_COMM_WORLD);
            sendBuffer += n_send;
        }

        MPI_Waitall(num_neighbors, request, MPI_STATUSES_IGNORE);

        //[Experimental] Can improve MPI_Allreduce performance 
        #if 0
        if (use_ibarrier == 1)
            MPI_Ibarrier(MPI_COMM_WORLD, request);
        #endif

        delete[] request;
    }
    else if (P2P_Mode == MPI_GPU_All2allv)
    {
        double* const xv = x.values_d;
        double* sendBuffer = A.gpuAux.sendBuffer;
        double* x_external = (double*) xv + localNumberOfRows;
        CHECK_CUDART(cudaStreamSynchronize(stream1));
        MPI_Alltoallv(
            sendBuffer, A.scounts, A.sdispls, MPI_DOUBLE, x_external, A.rcounts, A.rdispls, MPI_DOUBLE, MPI_COMM_WORLD);
    }
    else if (P2P_Mode == MPI_CPU_All2allv)
    {
        double* const xv = x.values;
        double* sendBuffer = A.sendBuffer;
        double* x_external = (double*) xv + localNumberOfRows;
        CHECK_CUDART(cudaEventSynchronize(copy_done));
        MPI_Alltoallv(
            sendBuffer, A.scounts, A.sdispls, MPI_DOUBLE, x_external, A.rcounts, A.rdispls, MPI_DOUBLE, MPI_COMM_WORLD);
        CHECK_CUDART(cudaMemcpyAsync(x.values_d + A.localNumberOfRows, x.values + A.localNumberOfRows,
            A.numberOfExternalValues * sizeof(double), cudaMemcpyHostToDevice, copy_stream));
        CHECK_CUDART(cudaEventRecord(copy_done, copy_stream));
        CHECK_CUDART(cudaStreamWaitEvent(0, copy_done, 0));
    }
    else if (P2P_Mode == NCCL)
    {
#ifdef USE_NCCL
        double* const xv = x.values_d;
        double* sendBuffer = A.gpuAux.sendBuffer;
        double* x_external = (double*) xv + localNumberOfRows;
        CHECK_NCCL(ncclGroupStart());
        for (int d = 0; d < num_neighbors; d++)
        {
            local_int_t n_send = sendLength[d];
            CHECK_NCCL(ncclSend(sendBuffer, n_send, ncclDouble, neighbors[d], Nccl_Comm, stream1));
            sendBuffer += n_send;

            local_int_t n_recv = receiveLength[d];
            CHECK_NCCL(ncclRecv(x_external, n_recv, ncclDouble, neighbors[d], Nccl_Comm, stream1));
            x_external += n_recv;
        }
        CHECK_NCCL(ncclGroupEnd());
#endif
        CHECK_CUDART(cudaStreamSynchronize(stream1));
    }
    return;
}
#endif

//////////////////////// Optimize Problem /////////////////////////////////////
/*
    GPU Kernel
    Fills an array with sequential numbers from 0...n-1
*/
__global__ void setVectorAsc_kernel(local_int_t* arr, local_int_t n)
{
    local_int_t id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= n)
        return;
    arr[id] = id;
}

/*
    GPU Kernel
    Minmax hashing used for graph coloring
    Colring is based on Jones-Plassmann Luby algorithm
*/
__global__ void minmaxHashStep_kernel(const local_int_t* A_cols, const local_int_t* nnz_per_row, local_int_t* color,
    int next_color, int next_color_p1, unsigned int seed, local_int_t n, int* hash_d)
{
    local_int_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n)
        return;
    // skip if previously matched
    if (color[i] != -1)
        return;
    unsigned int i_rand = __brev(i) /* hash function*/;
    // have we been proved to be not min or max
    bool not_min = false;
    bool not_max = false;
    slice_ptr_t row_start = (slice_ptr_t) i * HPCG_MAX_ROW_LEN;
    slice_ptr_t row_end = row_start + nnz_per_row[i];
    for (auto r = row_start; r < row_end; r++)
    {
        auto j = A_cols[r];
        // skip diagonal
        if (j == i || j >= n || j < 0)
            continue;
        auto j_color = color[j];
        // ignore colored neighbors (consider only the graph formed by removing them)
        if (j_color != -1 && j_color != next_color && j_color != next_color_p1)
            continue;
        unsigned int j_rand = __brev(j) /* hash function*/;
        // bail if any neighbor is greater
        if (i_rand <= j_rand)
            not_max = true;
        if (i_rand >= j_rand)
            not_min = true;
        if (not_max && not_min)
            return;
    }
    // we made it here, which means we have no higher/lower uncolored neighbor.  So we are selected.
    if (!not_min)
        color[i] = next_color;
    else if (!not_max) // else b/c we can be either min or max, so just pick one
        color[i] = next_color_p1;
}

/*
    GPU Kernel
    Minmax hashing used for graph coloring
    Colring is based on Jones-Plassmann Luby algorithm
*/
__global__ void testHashStep3_kernel(const local_int_t* A_cols, const local_int_t* nnz_per_row, local_int_t* color,
    int next_color, unsigned int seed, local_int_t n, int check_color)
{
    local_int_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n)
        return;
    if (color[i] != check_color)
        return;
    slice_ptr_t row_start = (slice_ptr_t) i * HPCG_MAX_ROW_LEN;
    slice_ptr_t row_end = row_start + nnz_per_row[i];
    int trial_color;
    int iter;

    for (iter = 0; iter < check_color; iter++)
    {
        trial_color = (iter + (i % seed) + next_color) % seed;
        bool color_used = false;
        for (auto r = row_start; r < row_end; r++)
        {
            auto j = A_cols[r];

            // skip diagonal
            if (j == i || j == -1 || j >= n)
                continue;

            auto j_color = color[j];
            if (j_color == trial_color)
            {
                color_used = true;
                break;
            }
        }
        if (!color_used)
        {
            color[i] = trial_color;
            return;
        }
    }
}

__global__ void inversePerm_kernel(local_int_t* out, local_int_t* opt2ref, local_int_t elements)
{
    local_int_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < elements)
    {
        out[opt2ref[index]] = index;
    }
}

/*
    GPU Kernel
    Use the coloring algorithm perm/ref2opt array to permute the values
        of elements to send array
*/
__global__ void __launch_bounds__(128)
    permElemToSend_kernel(local_int_t totalToBeSent, local_int_t* elementsToSend, local_int_t* ref2opt)
{
    const local_int_t i = blockIdx.x * 128 + threadIdx.x;
    if (i >= totalToBeSent)
        return;
    elementsToSend[i] = ref2opt[elementsToSend[i]];
}

/*
    GPU Kernel
    Creates the internal problem matrix, permutes the rows and columns
        based on the coloring perm array
    Padds -1 from row length to HPCG_MAX_ROW_LEN
    Counts the number of lower and upper elements for each row

    **Note** The internal column indices are assumed to be ascendingly
        ordered. Order is enforced during setupHalo_kernel
*/
template <int BLOCK_SIZE, int GROUP_SIZE, bool DIAG, class ColT>
__global__ void __launch_bounds__(BLOCK_SIZE) ellPermColumnsValues_kernel(local_int_t localNumberOfRows,
    local_int_t* nnzPerRow, local_int_t* columns, double* values, slice_ptr_t* csr_perm_offsets,
    ColT* csr_perm_columns, double* csr_perm_values, local_int_t* opt2ref, local_int_t* ref2opt,
    slice_ptr_t* ell_diagonal_idx, slice_ptr_t* csrLPermOffsets, slice_ptr_t* csrUPermOffsets,
    local_int_t slice_size)
{

    int lx = threadIdx.x % GROUP_SIZE;
    int wx = threadIdx.x / GROUP_SIZE;
    const int RPB = BLOCK_SIZE / GROUP_SIZE;

    __shared__ int counter[RPB];

    const local_int_t row = blockIdx.x * RPB + wx;
    if (row >= localNumberOfRows)
        return;

    if (lx == 0)
    {
        counter[wx] = 0;
        csrLPermOffsets[row] = 0;
        csrUPermOffsets[row] = 0;
    }
    __syncwarp();

    const local_int_t perm_row = opt2ref[row];
    const slice_ptr_t str = (slice_ptr_t) perm_row * HPCG_MAX_ROW_LEN;
    const local_int_t nnz = nnzPerRow[perm_row];
    columns += str;

    // Fused transpose: write directly in column-major sliced-ELL layout instead
    // of row-major (which previously required a separate TransposeCuda pass).
    //   offset(row, i) = (row/slice_size)*slice_size*HPCG_MAX_ROW_LEN
    //                    + (row % slice_size) + i*slice_size
    const local_int_t row_block_id = row / slice_size;
    const local_int_t row_inblock_id = row - row_block_id * slice_size;
    const slice_ptr_t row_start_index
        = (slice_ptr_t) row_block_id * HPCG_MAX_ROW_LEN * slice_size + row_inblock_id;

    local_int_t l_nnz = 0, u_nnz = 0;
#pragma unroll 9
    for (auto i = lx; i < HPCG_MAX_ROW_LEN; i += GROUP_SIZE)
    {
        const slice_ptr_t dst = row_start_index + (slice_ptr_t) i * slice_size;
        local_int_t orig_col = i < nnz ? columns[i] : localNumberOfRows;
        if (orig_col < localNumberOfRows)
        {
            local_int_t col = ref2opt[orig_col];
            csr_perm_columns[dst] = col;
            if (col == row)
            {
                csr_perm_values[dst] = 26.0;
                if (DIAG)
                    ell_diagonal_idx[row] = dst;
            }
            if (col < row)
                l_nnz++;
            if (col > row)
                u_nnz++;
        }
        else
        {
            csr_perm_columns[dst] = -1;
        }
    }

    atomic_add(&(csrLPermOffsets[row]), (slice_ptr_t) l_nnz);
    atomic_add(&(csrUPermOffsets[row]), (slice_ptr_t) u_nnz);
}

/*
    GPU Kernel
    Finds the maximum row length for lower and upper sliced ELLPACK slices
*/
template <class OffsetT>
__global__ void ellMaxRowLenPerBlock_kernel(local_int_t nrow, local_int_t slice_size, slice_ptr_t* csrLPermOffsets,
    slice_ptr_t* csrUPermOffsets, OffsetT* ell_l_per_color_mrl, OffsetT* ell_u_per_color_mrl)
{
    __shared__ local_int_t l_global_mrl, u_global_mrl;

    local_int_t nrows_per_block = slice_size;
    local_int_t block_str = blockIdx.x * nrows_per_block;

    if (threadIdx.x == 0)
    {
        l_global_mrl = 0;
        u_global_mrl = 0;
    }

    __syncthreads();

    local_int_t l_local_mrl = 0;
    local_int_t u_local_mrl = 0;
    for (auto i = threadIdx.x + block_str; i < block_str + nrows_per_block; i += blockDim.x)
    {
        if (i < nrow)
        {
            local_int_t l_rl = (local_int_t) csrLPermOffsets[i];
            local_int_t u_rl = (local_int_t) csrUPermOffsets[i];

            if (l_local_mrl < l_rl)
                l_local_mrl = l_rl;

            if (u_local_mrl < u_rl)
                u_local_mrl = u_rl;
        }
    }

    if (l_local_mrl > l_global_mrl)
        atomicMax(&l_global_mrl, l_local_mrl);

    if (u_local_mrl > u_global_mrl)
        atomicMax(&u_global_mrl, u_local_mrl);

    __syncthreads();

    ell_l_per_color_mrl[blockIdx.x + 1] = l_global_mrl;
    ell_u_per_color_mrl[blockIdx.x + 1] = u_global_mrl;
}

/*
    GPU Kernel
    Multiplies each element in arr with slice_size to create a slice offset
        based on the number of nonzeros
*/
template <class OffsetT>
__global__ void multiplyBySliceSize_kernel(local_int_t nrow, local_int_t slice_size, OffsetT* arr)
{

    const local_int_t i = blockIdx.x * 128 + threadIdx.x;
    if (i >= nrow)
        return;

    arr[i] = arr[i] * (OffsetT) slice_size;
}

/*
    GPU Kernel
    Generates the HPCG general matrix slice offset, based on the slice size
        and 27 nnz per row
*/
template <class OffsetT>
__global__ void createAMatrixSliceOffsets_kernel(local_int_t nrow, local_int_t slice_size, OffsetT* arr)
{
    const local_int_t i = blockIdx.x * 128 + threadIdx.x;
    if (i >= nrow)
        return;

    // Offsets can exceed 2^31 for large problems: accumulate in the offset type.
    arr[i] = (OffsetT) i * slice_size * HPCG_MAX_ROW_LEN;
}

/*
    GPU Kernel
    Fills the lower and upper values with minus one
*/
template<int THREADS_PER_CTA, int ELEMENTS_PER_THREAD>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    setLUValues_kernel(slice_ptr_t nnz, double* __restrict__ l_values, double* __restrict__ u_values)
{
    const slice_ptr_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const slice_ptr_t stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread with unrolling
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i)
    {
        slice_ptr_t idx = gid + i * stride;
        if (idx < nnz)
        {
            l_values[idx] = -1.0;
            u_values[idx] = -1.0;
        }
    }
}

/*
    GPU Kernel
    Creates the lower and upper matrices in sliced ELLPACK fromat
    Pads -1 for each row when its length is less tahn its max row
        length per slice
*/
template <class OffsetT, class ColT>
__global__ void createSellLUColumnsValues_kernel(const local_int_t n, const local_int_t slice_size,
    const ColT* __restrict ell_columns, double* __restrict ell_values, const OffsetT* __restrict ell_l_slice_offset,
    ColT* __restrict ell_l_columns, double* __restrict ell_l_values, const OffsetT* __restrict ell_u_slice_offset,
    ColT* __restrict ell_u_columns, double* __restrict ell_u_values)
{

    constexpr int MaxRowLen = HPCG_MAX_ROW_LEN;
    local_int_t row_original_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (row_original_id >= n)
        return;

    local_int_t row_inblock_id = row_original_id % slice_size;
    local_int_t row_block_id = row_original_id / slice_size;

    // Array indices are driven by the (possibly 64-bit) slice offsets, so use OffsetT.
    OffsetT row_start_index = (OffsetT) row_block_id * MaxRowLen * slice_size + row_inblock_id;
    OffsetT row_end_index = row_start_index + (OffsetT) MaxRowLen * slice_size;

    OffsetT l_row_start = ell_l_slice_offset[row_block_id] + row_inblock_id;
    OffsetT u_row_start = ell_u_slice_offset[row_block_id] + row_inblock_id;

    OffsetT l_len = (ell_l_slice_offset[row_block_id + 1] - ell_l_slice_offset[row_block_id]) / slice_size;
    OffsetT u_len = (ell_u_slice_offset[row_block_id + 1] - ell_u_slice_offset[row_block_id]) / slice_size;

    OffsetT l_row_end = l_row_start + l_len * slice_size;
    OffsetT u_row_end = u_row_start + u_len * slice_size;

#pragma unroll MaxRowLen
    for (OffsetT i = row_start_index; i < row_end_index; i += slice_size)
    {
        ColT col = __ldcs(&ell_columns[i]);
        double val = __ldcs(&ell_values[i]);
        if (col != -1 && col < row_original_id)
        {
            ell_l_columns[l_row_start] = col;
            l_row_start += slice_size;
        }
        else if (col != -1 && col > row_original_id)
        {
            ell_u_columns[u_row_start] = col;
            u_row_start += slice_size;
        }
    }

    // Padd lower
    for (OffsetT i = l_row_start; i < l_row_end; i += slice_size)
    {
        ell_l_columns[i] = -1;
    }

    // Padd upper
    for (OffsetT i = u_row_start; i < u_row_end; i += slice_size)
    {
        ell_u_columns[i] = -1;
    }
}

/*
    GPU Kernel
    Permutes array/vector elements using the coloring perm array
*/
__global__ void __launch_bounds__(128) permVector_kernel(local_int_t n, double* tmp, double* x, local_int_t* perm)
{

    const local_int_t row = blockIdx.x * 128 + threadIdx.x;
    if (row < n)
    {
        tmp[row] = x[perm[row]];
    }
}

/*
    GPU Kernel
    Permutes space for injection operator
*/
__global__ void __launch_bounds__(128) f2cPerm_kernel(
    local_int_t nrow_c, local_int_t* f2c, local_int_t* f2cPerm, local_int_t* perm_f, local_int_t* iperm_c)
{

    local_int_t i = blockIdx.x * 128 + threadIdx.x;
    if (i < nrow_c)
    {
        f2cPerm[i] = perm_f[f2c[iperm_c[i]]];
    }
}

/*
    Calls setVectorAsc_kernel
    Assigns sequential indices to an array from
        0...n-1
*/
void SetVectorAscCuda(local_int_t* arr, local_int_t n)
{
    int block = 256;
    local_int_t grid = (n + block - 1) / block;
    setVectorAsc_kernel<<<grid, block, 0, stream>>>(arr, n);
}

/*
    Colors the matrix using Jones-Plassmann Luby algorithm
*/
void ColorMatrixCuda(double* A_vals, local_int_t* A_col, local_int_t* nnzPerRow, local_int_t rows, local_int_t* color,
    int* num_colors, int* count_colors, int max_colors, local_int_t* ref2opt, local_int_t* opt2ref, int rank, int nx,
    int* rowhash)
{
    int perm_colors[8];
    perm_colors[0] = 7;
    perm_colors[1] = 0;
    perm_colors[2] = 5;
    perm_colors[3] = 6;
    perm_colors[4] = 2;
    perm_colors[5] = 1;
    perm_colors[6] = 3;
    perm_colors[7] = 4;

    thrust::device_ptr<local_int_t> dp_color(color);
    thrust::device_ptr<local_int_t> dp_perm(opt2ref);
    dim3 block(128, 1, 1);
    dim3 grid((rows + block.x - 1) / block.x, 1, 1);
    int next_color = 0;
    int seed = 0;
    int done = 0;
    int colored = 0;
    int step = 0;
    while (!done && step < max_colors / 2)
    {
        done = 1;
        if (next_color < 7)
        {
            minmaxHashStep_kernel<<<grid, block, 0, stream>>>(
                A_col, nnzPerRow, color, perm_colors[next_color], perm_colors[next_color + 1], seed, rows, rowhash);
            count_colors[perm_colors[next_color]] = thrust::count(dp_color, dp_color + rows, perm_colors[next_color]);
            count_colors[perm_colors[next_color + 1]]
                = thrust::count(dp_color, dp_color + rows, perm_colors[next_color + 1]);
            colored += count_colors[perm_colors[next_color]] + count_colors[perm_colors[next_color + 1]];
        }
        else
        {
            minmaxHashStep_kernel<<<grid, block, 0, stream>>>(
                A_col, nnzPerRow, color, next_color, next_color + 1, seed, rows, rowhash);
            count_colors[next_color] = thrust::count(dp_color, dp_color + rows, next_color);
            count_colors[next_color + 1] = thrust::count(dp_color, dp_color + rows, next_color + 1);
            colored += count_colors[next_color] + count_colors[next_color + 1];
        }

        if (colored < rows)
            done = 0;
        step++;
        next_color += 2;
    }
    int check_color;
    int color_target = 1;
    int recolor_times = 10;
    int maxx = thrust::reduce(dp_color, dp_color + rows, -1, thrust::maximum<int>());

    int max_used_color = maxx;
    if (maxx > 15)
    {
        for (auto target_color_count = maxx - 1; target_color_count > 13; target_color_count--)
        {
            int it_count = 0;
            while (it_count < recolor_times && maxx > target_color_count)
            {
                for (check_color = maxx; check_color >= color_target; check_color--)
                {
                    testHashStep3_kernel<<<grid, block, 0, stream>>>(
                        A_col, nnzPerRow, color, it_count, 15, rows, check_color);
                }
                maxx = thrust::reduce(dp_color, dp_color + rows, -1, thrust::maximum<local_int_t>());
                count_colors[maxx] = thrust::count(dp_color, dp_color + rows, maxx);
                if (rank == 0)
                    printf("%d d_max_color = %d (%d elements)\n", it_count, maxx, count_colors[maxx]);
                it_count++;
            }
        }

        for (auto i = 0; i < max_colors; i++)
            count_colors[i] = 0;

        for (check_color = 0; check_color < next_color; check_color++)
        {
            count_colors[check_color] = thrust::count(dp_color, dp_color + rows, check_color);
        }
        max_used_color = 0;
        for (auto i = 0; i < max_colors; i++)
            if (count_colors[i] > 0)
                max_used_color = i;
    }

    *num_colors = max_used_color + 1;
    thrust::sort_by_key(dp_color, dp_color + rows, dp_perm);
    inversePerm_kernel<<<grid, block, 0, stream>>>(ref2opt, opt2ref, rows);
}

/*
    Permutes elements to send buffer
*/
void PermElemToSendCuda(local_int_t totalToBeSent, local_int_t* elementsToSend, local_int_t* ref2opt)
{
    if (totalToBeSent > 0)
    {
        const local_int_t grid = (totalToBeSent + 128 - 1) / 128;
        permElemToSend_kernel<<<grid, 128, 0, stream>>>(totalToBeSent, elementsToSend, ref2opt);
    }
}

/*
    Creates the internal permuted matrix in (Sliced-)ELLPACK format
*/
#ifndef EXPLICIT_KERNELS
void EllPermColumnsValuesCuda(local_int_t localNumberOfRows, local_int_t* nnzPerRow, local_int_t* columns,
    double* values, slice_ptr_t* csr_perm_offsets, void* csr_perm_columns, double* csr_perm_values,
    local_int_t* opt2ref, local_int_t* ref2opt, slice_ptr_t* diagonalIdx, slice_ptr_t* csrLPermOffsets,
    slice_ptr_t* csrUPermOffsets, bool find_diag, local_int_t slice_size, IndexMode mode)
{
    // The build kernel now emits the column-major sliced-ELL layout directly, so
    // the value buffer must be initialized to -1 over the full padded slice range
    // (the kernel only overwrites the diagonal; off-diagonal/pad values stay -1).
    const local_int_t num_slices = (localNumberOfRows + slice_size - 1) / slice_size;
    const local_int_t paddedRowLen = num_slices * slice_size;
    const size_t nnz_out = (size_t) paddedRowLen * HPCG_MAX_ROW_LEN;

    const size_t grid_nnz = (nnz_out + 128 - 1) / 128;
    setMinusOne_kernel<<<grid_nnz, 128, 0, stream>>>(nnz_out, csr_perm_values);

    // For a partial final slice, the padded rows beyond localNumberOfRows are not
    // visited by the kernel; pre-fill that slice's column indices with -1 so the
    // pad entries are skipped downstream. (No-op for slice-size-divisible sizes.)
    if (paddedRowLen > localNumberOfRows)
    {
        const size_t colBytes = columnIndexBytes(mode);
        const size_t last_slice_base = (size_t) (num_slices - 1) * slice_size * HPCG_MAX_ROW_LEN;
        const size_t last_slice_bytes = (size_t) slice_size * HPCG_MAX_ROW_LEN * colBytes;
        CHECK_CUDART(cudaMemsetAsync(
            byteOffset(csr_perm_columns, last_slice_base, colBytes), 0xFF, last_slice_bytes, stream));
    }

    const int BLOCK_SIZE = 128;
    const int GROUP_SIZE = 8; // Number of threads per row

    const int WORKERS = BLOCK_SIZE / GROUP_SIZE;
    const local_int_t grid = (localNumberOfRows + WORKERS - 1) / WORKERS;

    // Only the column-index element type varies with the index mode here.
    dispatchIndexMode(mode,
        [&](auto /*offTag*/, auto colTag)
        {
            using ColT = decltype(colTag);
            ColT* cols = static_cast<ColT*>(csr_perm_columns);
            if (find_diag)
                ellPermColumnsValues_kernel<BLOCK_SIZE, GROUP_SIZE, true, ColT><<<grid, BLOCK_SIZE, 0, stream>>>(
                    localNumberOfRows, nnzPerRow, columns, values, csr_perm_offsets, cols, csr_perm_values, opt2ref,
                    ref2opt, diagonalIdx, csrLPermOffsets, csrUPermOffsets, slice_size);
            else
                ellPermColumnsValues_kernel<BLOCK_SIZE, GROUP_SIZE, false, ColT><<<grid, BLOCK_SIZE, 0, stream>>>(
                    localNumberOfRows, nnzPerRow, columns, values, csr_perm_offsets, cols, csr_perm_values, opt2ref,
                    ref2opt, diagonalIdx, csrLPermOffsets, csrUPermOffsets, slice_size);
        });
}

/*
    Finds the max lower and upper row length for each slice
*/
void EllMaxRowLenPerBlockCuda(local_int_t nrow, int slice_size, slice_ptr_t* ell_perm_l_offsets,
    slice_ptr_t* ell_perm_u_offsets, void* sellLSliceMrl, void* ell_u_block_mrl, IndexMode mode)
{
    int blockSize = 512;
    local_int_t gridSize = (nrow + slice_size - 1) / slice_size;
    dispatchIndexMode(mode,
        [&](auto offTag, auto /*colTag*/)
        {
            using OffsetT = decltype(offTag);
            ellMaxRowLenPerBlock_kernel<OffsetT><<<gridSize, blockSize, 0, stream>>>(nrow, slice_size,
                ell_perm_l_offsets, ell_perm_u_offsets, static_cast<OffsetT*>(sellLSliceMrl),
                static_cast<OffsetT*>(ell_u_block_mrl));
        });
}

/*
    Finds prefix sum using CUB
*/
void PrefixsumCuda(local_int_t localNumberOfRows, void* arr, IndexMode mode)
{
    dispatchIndexMode(mode,
        [&](auto offTag, auto /*colTag*/)
        {
            using OffsetT = decltype(offTag);
            OffsetT* a = static_cast<OffsetT*>(arr);
            void* d_temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            CHECK_CUDART(cudaMemsetAsync(a, 0, sizeof(OffsetT), stream));
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, a + 1, a + 1, localNumberOfRows);
            CHECK_CUDART(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, a + 1, a + 1, localNumberOfRows);
            CHECK_CUDART(cudaFree(d_temp_storage));
        });
}

/*
    64-bit sum of a slice_ptr_t device array.

    Used to derive exact per-matrix nonzero counts for the cuSPARSE Sliced-ELL
    descriptors without relying on the int32 localNumberOfNonzeros, which wraps
    for local problems with more than 2^31 nonzeros (e.g. 512^3).
*/
slice_ptr_t SumSlicePtrCuda(const slice_ptr_t* arr, local_int_t n)
{
    slice_ptr_t* d_out = nullptr;
    CHECK_CUDART(cudaMalloc(&d_out, sizeof(slice_ptr_t)));
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(d_temp, temp_bytes, arr, d_out, n, stream);
    CHECK_CUDART(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceReduce::Sum(d_temp, temp_bytes, arr, d_out, n, stream);
    slice_ptr_t h = 0;
    CHECK_CUDART(cudaMemcpyAsync(&h, d_out, sizeof(slice_ptr_t), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDART(cudaStreamSynchronize(stream));
    CHECK_CUDART(cudaFree(d_temp));
    CHECK_CUDART(cudaFree(d_out));
    return h;
}

/*
    Multiplies the slice offset based on max row length by
        the slice size to make based on number of nnz
*/
void MultiplyBySliceSizeCUDA(local_int_t nrow, int slice_size, void* arr, IndexMode mode)
{
    const local_int_t grid = (nrow + 128 - 1) / 128;
    dispatchIndexMode(mode,
        [&](auto offTag, auto /*colTag*/)
        {
            using OffsetT = decltype(offTag);
            multiplyBySliceSize_kernel<OffsetT><<<grid, 128, 0, stream>>>(
                nrow, slice_size, static_cast<OffsetT*>(arr));
        });
}

/*
    Creates a slice offset for the general matrix that has exactly
*/
void CreateAMatrixSliceOffsetsCuda(local_int_t nrow, local_int_t slice_size, void* arr, IndexMode mode)
{
    const local_int_t grid = (nrow + 128 - 1) / 128;
    dispatchIndexMode(mode,
        [&](auto offTag, auto /*colTag*/)
        {
            using OffsetT = decltype(offTag);
            createAMatrixSliceOffsets_kernel<OffsetT><<<grid, 128, 0, stream>>>(
                nrow, slice_size, static_cast<OffsetT*>(arr));
        });
}

/*
    Creates the lower and upper matrices in sliced ELLPACK format
*/
void CreateSellLUColumnsValuesCuda(const local_int_t n, const int slice_size, void* ell_columns,
    double* ell_values, void* ell_l_slice_offset, void* ell_l_columns, double* ell_l_values,
    void* ell_u_slice_offset, void* ell_u_columns, double* ell_u_values, int level, IndexMode mode)
{
    local_int_t num_blocks = (n + slice_size - 1) / slice_size;
    local_int_t paddedRowLen = num_blocks * slice_size;

    /*Memory Estimation for lower and upper parts*/
    slice_ptr_t estimated_size = EstimateLUmem(n, (local_int_t) paddedRowLen, level, slice_size);

    const int BlockSize = 128;
    const int ELEMENTS_PER_THREAD = 8;
    const int ELEMENTS_PER_CTA = BlockSize * ELEMENTS_PER_THREAD;
    const slice_ptr_t grid_nnz = (estimated_size + ELEMENTS_PER_CTA - 1) / ELEMENTS_PER_CTA;
    local_int_t grid = (n + BlockSize - 1) / BlockSize;
    setLUValues_kernel<BlockSize, ELEMENTS_PER_THREAD><<<grid_nnz, BlockSize, 0, stream>>>(
        estimated_size, ell_u_values, ell_l_values);
    dispatchIndexMode(mode,
        [&](auto offTag, auto colTag)
        {
            using OffsetT = decltype(offTag);
            using ColT = decltype(colTag);
            createSellLUColumnsValues_kernel<OffsetT, ColT><<<grid, BlockSize, 0, stream>>>(n, slice_size,
                static_cast<const ColT*>(ell_columns), ell_values, static_cast<const OffsetT*>(ell_l_slice_offset),
                static_cast<ColT*>(ell_l_columns), ell_l_values, static_cast<const OffsetT*>(ell_u_slice_offset),
                static_cast<ColT*>(ell_u_columns), ell_u_values);
        });
}
#else
// Explicit kernels: fixed-width device arrays, no IndexMode dispatch (see SparseMatrix.hpp).
void EllPermColumnsValuesCuda(local_int_t localNumberOfRows, local_int_t* nnzPerRow, local_int_t* columns,
    double* values, slice_ptr_t* csr_perm_offsets, local_int_t* csr_perm_columns, double* csr_perm_values,
    local_int_t* opt2ref, local_int_t* ref2opt, slice_ptr_t* diagonalIdx, slice_ptr_t* csrLPermOffsets,
    slice_ptr_t* csrUPermOffsets, bool find_diag, local_int_t slice_size)
{
    // The build kernel now emits the column-major sliced-ELL layout directly, so
    // the value buffer must be initialized to -1 over the full padded slice range
    // (the kernel only overwrites the diagonal; off-diagonal/pad values stay -1).
    const local_int_t num_slices = (localNumberOfRows + slice_size - 1) / slice_size;
    const local_int_t paddedRowLen = num_slices * slice_size;
    const slice_ptr_t nnz_out = (slice_ptr_t) paddedRowLen * HPCG_MAX_ROW_LEN;

    const slice_ptr_t grid_nnz = (nnz_out + 128 - 1) / 128;
    setMinusOne_kernel<<<grid_nnz, 128, 0, stream>>>(nnz_out, csr_perm_values);

    // For a partial final slice, the padded rows beyond localNumberOfRows are not
    // visited by the kernel; pre-fill that slice's column indices with -1 so the
    // pad entries are skipped downstream. (No-op for slice-size-divisible sizes.)
    if (paddedRowLen > localNumberOfRows)
    {
        const size_t last_slice_base = (size_t) (num_slices - 1) * slice_size * HPCG_MAX_ROW_LEN;
        const size_t last_slice_bytes = (size_t) slice_size * HPCG_MAX_ROW_LEN * sizeof(local_int_t);
        CHECK_CUDART(cudaMemsetAsync(csr_perm_columns + last_slice_base, 0xFF, last_slice_bytes, stream));
    }

    const int BLOCK_SIZE = 128;
    const int GROUP_SIZE = 8; // Number of threads per row

    const int WORKERS = BLOCK_SIZE / GROUP_SIZE;
    const local_int_t grid = (localNumberOfRows + WORKERS - 1) / WORKERS;

    if (find_diag)
        ellPermColumnsValues_kernel<BLOCK_SIZE, GROUP_SIZE, true, local_int_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            localNumberOfRows, nnzPerRow, columns, values, csr_perm_offsets, csr_perm_columns, csr_perm_values,
            opt2ref, ref2opt, diagonalIdx, csrLPermOffsets, csrUPermOffsets, slice_size);
    else
        ellPermColumnsValues_kernel<BLOCK_SIZE, GROUP_SIZE, false, local_int_t><<<grid, BLOCK_SIZE, 0, stream>>>(
            localNumberOfRows, nnzPerRow, columns, values, csr_perm_offsets, csr_perm_columns, csr_perm_values,
            opt2ref, ref2opt, diagonalIdx, csrLPermOffsets, csrUPermOffsets, slice_size);
}

/*
    Finds the max lower and upper row length for each slice
*/
void EllMaxRowLenPerBlockCuda(local_int_t nrow, int slice_size, slice_ptr_t* ell_perm_l_offsets,
    slice_ptr_t* ell_perm_u_offsets, slice_ptr_t* sellLSliceMrl, slice_ptr_t* ell_u_block_mrl)
{
    int blockSize = 512;
    local_int_t gridSize = (nrow + slice_size - 1) / slice_size;
    ellMaxRowLenPerBlock_kernel<slice_ptr_t><<<gridSize, blockSize, 0, stream>>>(
        nrow, slice_size, ell_perm_l_offsets, ell_perm_u_offsets, sellLSliceMrl, ell_u_block_mrl);
}

/*
    Finds prefix sum using CUB
*/
void PrefixsumCuda(local_int_t localNumberOfRows, slice_ptr_t* arr)
{
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CHECK_CUDART(cudaMemsetAsync(arr, 0, sizeof(slice_ptr_t), stream));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, arr + 1, arr + 1, localNumberOfRows);
    CHECK_CUDART(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, arr + 1, arr + 1, localNumberOfRows);
    CHECK_CUDART(cudaFree(d_temp_storage));
}

/*
    Multiplies the slice offset based on max row length by
        the slice size to make based on number of nnz
*/
void MultiplyBySliceSizeCUDA(local_int_t nrow, int slice_size, slice_ptr_t* arr)
{
    const local_int_t grid = (nrow + 128 - 1) / 128;
    multiplyBySliceSize_kernel<slice_ptr_t><<<grid, 128, 0, stream>>>(nrow, slice_size, arr);
}

/*
    Creates a slice offset for the general matrix that has exactly
*/
void CreateAMatrixSliceOffsetsCuda(local_int_t nrow, local_int_t slice_size, slice_ptr_t* arr)
{
    const local_int_t grid = (nrow + 128 - 1) / 128;
    createAMatrixSliceOffsets_kernel<slice_ptr_t><<<grid, 128, 0, stream>>>(nrow, slice_size, arr);
}

/*
    Creates the lower and upper matrices in sliced ELLPACK format
*/
void CreateSellLUColumnsValuesCuda(const local_int_t n, const int slice_size, local_int_t* ell_columns,
    double* ell_values, slice_ptr_t* ell_l_slice_offset, local_int_t* ell_l_columns, double* ell_l_values,
    slice_ptr_t* ell_u_slice_offset, local_int_t* ell_u_columns, double* ell_u_values, int level)
{
    local_int_t num_blocks = (n + slice_size - 1) / slice_size;
    local_int_t paddedRowLen = num_blocks * slice_size;

    /*Memory Estimation for lower and upper parts*/
    slice_ptr_t estimated_size = EstimateLUmem(n, (local_int_t) paddedRowLen, level, slice_size);

    const int BlockSize = 128;
    const int ELEMENTS_PER_THREAD = 8;
    const int ELEMENTS_PER_CTA = BlockSize * ELEMENTS_PER_THREAD;
    const slice_ptr_t grid_nnz = (estimated_size + ELEMENTS_PER_CTA - 1) / ELEMENTS_PER_CTA;
    local_int_t grid = (n + BlockSize - 1) / BlockSize;
    setLUValues_kernel<BlockSize, ELEMENTS_PER_THREAD><<<grid_nnz, BlockSize, 0, stream>>>(
        estimated_size, ell_u_values, ell_l_values);
    createSellLUColumnsValues_kernel<slice_ptr_t, local_int_t><<<grid, BlockSize, 0, stream>>>(n, slice_size,
        ell_columns, ell_values, ell_l_slice_offset, ell_l_columns, ell_l_values, ell_u_slice_offset, ell_u_columns,
        ell_u_values);
}
#endif

/*
    Permutes a vector using the coloring matrix
    Allocates and free device memory
*/
void PermVectorCuda(local_int_t* perm, Vector& x, local_int_t length)
{
    double* xv = x.values_d;
    double* tmp = NULL;
    CHECK_CUDART(cudaMalloc(&tmp, sizeof(double) * length));
    permVector_kernel<<<(length + 128 - 1) / 128, 128, 0>>>(length, tmp, xv, perm);
    CHECK_CUDART(cudaMemcpy(xv, tmp, sizeof(double) * length, cudaMemcpyDeviceToDevice));
    CHECK_CUDART(cudaFree(tmp));
}

/*
    Permutes the space injection operator
*/
void F2cPermCuda(local_int_t nrow_c, local_int_t* f2c, local_int_t* f2cPerm, local_int_t* perm_f, local_int_t* iperm_c)
{
    const local_int_t grid = (nrow_c + 128 - 1) / 128;
    f2cPerm_kernel<<<grid, 128, 0, stream>>>(nrow_c, f2c, f2cPerm, perm_f, iperm_c);
}

/*
    Reads a single slice-offset element (element `index`) from a width-agnostic
    device offset array and returns it widened to 64-bit host storage. Confines
    the mode-dependent element-width read to one place.
*/
long long ReadSellOffsetCuda(const void* arr, size_t index, IndexMode mode)
{
    if (offsetsAre64(mode))
    {
        long long value = 0;
        CHECK_CUDART(cudaMemcpy(
            &value, static_cast<const long long*>(arr) + index, sizeof(long long), cudaMemcpyDeviceToHost));
        return value;
    }
    int value = 0;
    CHECK_CUDART(
        cudaMemcpy(&value, static_cast<const int*>(arr) + index, sizeof(int), cudaMemcpyDeviceToHost));
    return (long long) value;
}

//////////////////////// Test CG //////////////////////////////////////////////
/*
    GPU Kernel
    Replaces matrix, in sliced ELLPACK format, diagonal with values in
        diagonal_buf
*/
template <class ColT>
__global__ void __launch_bounds__(128) replaceMatrixDiagonal_kernel(
    local_int_t localNumberOfRows, local_int_t slice_size, const ColT* ell_cols, double* ell_values, double* diagonal, double* diagonal_buf)
{

    local_int_t row_index = threadIdx.x + blockDim.x * blockIdx.x;
    if (row_index < localNumberOfRows)
    {
        local_int_t row_x = row_index % slice_size;
        local_int_t row_y = row_index / slice_size;
        // Element offsets into the A operator can exceed 2^31: compute in 64-bit.
        size_t start_id = (size_t) row_x + (size_t) row_y * slice_size * HPCG_MAX_ROW_LEN;
        size_t end_id = start_id + (size_t) HPCG_MAX_ROW_LEN * slice_size;
        size_t id = start_id;
        while (ell_cols[id] != (ColT) row_index && id < end_id)
            id += slice_size;
        double mydiag = diagonal_buf[row_index];
        ell_values[id] = mydiag;
        diagonal[row_index] = mydiag;
    }
}

/*
    Replaces the diagonal matrix, in sliced ELLPACK, with values in
        diagonal
*/
void ReplaceMatrixDiagonalCuda(SparseMatrix& A, Vector& diagonal)
{
    const int grid = (A.localNumberOfRows + 128 - 1) / 128;
    dispatchIndexMode(A.index_mode,
        [&](auto /*offTag*/, auto colTag)
        {
            using ColT = decltype(colTag);
            replaceMatrixDiagonal_kernel<ColT><<<grid, 128, 0, stream>>>(A.localNumberOfRows, A.slice_size,
                static_cast<const ColT*>(A.sellDev.aColumns), A.sellAPermValues, A.diagonal, diagonal.values_d);
        });
}

/*
    Copies the matrix, in sliced ELLPACK, diagonal into a GPU buffer
        diagonal
*/
void CopyMatrixDiagonalCuda(SparseMatrix& A, Vector& diagonal)
{
    CHECK_CUDART(cudaMemcpyAsync(
        diagonal.values_d, A.diagonal, sizeof(double) * A.localNumberOfRows, cudaMemcpyDeviceToDevice, stream));
}

//////////////////////// CG Support Kernels ///////////////////////////////////
//////////////////////// CG Support Kernels: MG ///////////////////////////////
/*
    GPU Kernel
    Computes restriction in MG
*/
template<int THREADS_PER_CTA, int ROUNDS>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    computeRestriction_kernel(local_int_t n, double* rfv, double* Axfv, double* rcv, local_int_t* f2c)
{
    const local_int_t base_idx = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    const local_int_t stride = THREADS_PER_CTA * gridDim.x;

    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t i = base_idx + round * stride;
        if (i < n)
        {
            rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
        }
    }
}

/*
    GPU Kernel
    Computes prolongation in MG
*/
template<int THREADS_PER_CTA, int ROUNDS>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    computeProlongation_kernel(local_int_t n, double* xcv, double* xfv, local_int_t* f2c)
{
    const local_int_t base_idx = blockIdx.x * THREADS_PER_CTA + threadIdx.x;
    const local_int_t stride = THREADS_PER_CTA * gridDim.x;
    
    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t i = base_idx + round * stride;
        if (i < n)
        {
            xfv[f2c[i]] += xcv[i];
        }
    }
}

/*
    Computes restriction in MG, calls computeRestriction_kernel
*/
void ComputeRestrictionCuda(const SparseMatrix& A, const Vector& r)
{
    local_int_t nc = A.mgData->rc->localLength;
    double* Axfv = A.mgData->Axf->values_d;
    double* rfv = r.values_d;
    double* rcv = A.mgData->rc->values_d;

    const int THREADS_PER_CTA = 256;
    const int ROUNDS = 2;
    const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS;
    const int grid = (nc + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
    computeRestriction_kernel<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(nc, rfv, Axfv, rcv, A.f2cPerm);
}

/*
    Computes prolongation in MG, calls computeProlongation_kernel
*/
void ComputeProlongationCuda(const SparseMatrix& A, Vector& x)
{
    local_int_t nc = A.mgData->rc->localLength;
    double* xfv = x.values_d;
    double* xcv = A.mgData->xc->values_d;

    const int THREADS_PER_CTA = 256;
    const int ROUNDS = 2;
    const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS;
    const int grid = (nc + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
    computeProlongation_kernel<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(nc, xcv, xfv, A.f2cPerm);
}

//////////////////////// CG Support Kernels: WAXPBY ///////////////////////////
/*
    GPU Kernel
    Computes WAXPBY - double2 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
 __global__ void __launch_bounds__(THREADS_PER_CTA)
    computeWAXPBY_kernel_double2(const local_int_t n, double alpha, double* __restrict__ x, double beta, double* __restrict__ y, double* w)
 {
    const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const local_int_t stride = blockDim.x * gridDim.x;
    
    // Process 2 rounds of double2 elements per thread
    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t base_idx = gid + round * stride;

         // Use double2 for vectorized loads/stores when possible
         if ( (base_idx * 2 + 1) < n)
         {
             double2 x_vec = *reinterpret_cast<double2*>(&x[base_idx * 2]);
             double2 y_vec = *reinterpret_cast<double2*>(&y[base_idx * 2]);
             
             double2 w_vec;
             w_vec.x = alpha * x_vec.x + beta * y_vec.x;
             w_vec.y = alpha * x_vec.y + beta * y_vec.y;
             
             *reinterpret_cast<double2*>(&w[base_idx * 2]) = w_vec;
         }
         else if ( (base_idx * 2) < n)
         {
             // Handle remaining element individually
             w[base_idx * 2] = alpha * x[base_idx * 2] + beta * y[base_idx * 2];
         }
     }
 }

/*
    GPU Kernel
    Computes WAXPBY - double4 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
 __global__ void __launch_bounds__(THREADS_PER_CTA)
    computeWAXPBY_kernel_double4(const local_int_t n, double alpha, double* __restrict__ x, double beta, double* __restrict__ y, double* w)
 {
    const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const local_int_t stride = blockDim.x * gridDim.x;
    
    // Process ROUNDS rounds of double4 elements per thread
    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t base_idx = gid + round * stride;

         // Use double4 for vectorized loads/stores when possible
         if ( (base_idx * 4 + 3) < n)
         {
             double4 x_vec = *reinterpret_cast<double4*>(&x[base_idx * 4]);
             double4 y_vec = *reinterpret_cast<double4*>(&y[base_idx * 4]);
             
             double4 w_vec;
             w_vec.x = alpha * x_vec.x + beta * y_vec.x;
             w_vec.y = alpha * x_vec.y + beta * y_vec.y;
             w_vec.z = alpha * x_vec.z + beta * y_vec.z;
             w_vec.w = alpha * x_vec.w + beta * y_vec.w;
             
             *reinterpret_cast<double4*>(&w[base_idx * 4]) = w_vec;
         }
         else if ( (base_idx * 4 + 2) < n)
         {
             // Handle 3 remaining elements
             w[base_idx * 4] = alpha * x[base_idx * 4] + beta * y[base_idx * 4];
             w[base_idx * 4 + 1] = alpha * x[base_idx * 4 + 1] + beta * y[base_idx * 4 + 1];
             w[base_idx * 4 + 2] = alpha * x[base_idx * 4 + 2] + beta * y[base_idx * 4 + 2];
         }
         else if ( (base_idx * 4 + 1) < n)
         {
             // Handle 2 remaining elements
             w[base_idx * 4] = alpha * x[base_idx * 4] + beta * y[base_idx * 4];
             w[base_idx * 4 + 1] = alpha * x[base_idx * 4 + 1] + beta * y[base_idx * 4 + 1];
         }
         else if ( (base_idx * 4) < n)
         {
             // Handle 1 remaining element
             w[base_idx * 4] = alpha * x[base_idx * 4] + beta * y[base_idx * 4];
         }
     }
 }
 

/*
    Computes WAXPBY followed by stream synchronization
*/
void ComputeWAXPBYCuda(
    const local_int_t n, const double alpha, const Vector& x, const double beta, const Vector& y, Vector& w)
{
    const int ROUNDS = 1;
    const int THREADS_PER_CTA = 256;
    if (g_config.VECTOR_WIDTH == 4) {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 4; // 4 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        computeWAXPBY_kernel_double4<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, alpha, x.values_d, beta, y.values_d, w.values_d);
    } else {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 2; // 2 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        computeWAXPBY_kernel_double2<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, alpha, x.values_d, beta, y.values_d, w.values_d);
    }
    CHECK_CUDART(cudaStreamSynchronize(stream));
}

//////////////////////// CG Support Kernels: SYMG /////////////////////////////
/*
    GPU Kernel
    Multiplies x values with d and accumultaes back to x - double2 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
 __global__ void __launch_bounds__(THREADS_PER_CTA)
    spmvDiag_kernel_double2(const local_int_t n, double* x, double* d)
 {
     const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
     const local_int_t stride = blockDim.x * gridDim.x;
     
     // Process 2 rounds of double2 elements per thread
     #pragma unroll
     for (int round = 0; round < ROUNDS; ++round)
     {
         local_int_t base_idx = gid + round * stride;
         
         // Use double2 for vectorized loads/stores when possible
         if (base_idx * 2 + 1 < n)
         {
             double2 x_vec = *reinterpret_cast<double2*>(&x[base_idx * 2]);
             double2 d_vec = *reinterpret_cast<double2*>(&d[base_idx * 2]);
             
             x_vec.x *= d_vec.x;
             x_vec.y *= d_vec.y;
             
             *reinterpret_cast<double2*>(&x[base_idx * 2]) = x_vec;
         }
         else if (base_idx * 2 < n)
         {
             // Handle remaining element individually
             x[base_idx * 2] *= d[base_idx * 2];
         }
     }
 }

/*
    GPU Kernel
    Multiplies x values with d and accumultaes back to x - double4 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
 __global__ void __launch_bounds__(THREADS_PER_CTA)
    spmvDiag_kernel_double4(const local_int_t n, double* x, double* d)
 {
     const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
     const local_int_t stride = blockDim.x * gridDim.x;
     
     // Process ROUNDS rounds of double4 elements per thread
     #pragma unroll
     for (int round = 0; round < ROUNDS; ++round)
     {
         local_int_t base_idx = gid + round * stride;
         
         // Use double4 for vectorized loads/stores when possible
         if (base_idx * 4 + 3 < n)
         {
             double4 x_vec = *reinterpret_cast<double4*>(&x[base_idx * 4]);
             double4 d_vec = *reinterpret_cast<double4*>(&d[base_idx * 4]);
             
             x_vec.x *= d_vec.x;
             x_vec.y *= d_vec.y;
             x_vec.z *= d_vec.z;
             x_vec.w *= d_vec.w;
             
             *reinterpret_cast<double4*>(&x[base_idx * 4]) = x_vec;
         }
         else if (base_idx * 4 + 2 < n)
         {
             // Handle 3 remaining elements
             x[base_idx * 4] *= d[base_idx * 4];
             x[base_idx * 4 + 1] *= d[base_idx * 4 + 1];
             x[base_idx * 4 + 2] *= d[base_idx * 4 + 2];
         }
         else if (base_idx * 4 + 1 < n)
         {
             // Handle 2 remaining elements
             x[base_idx * 4] *= d[base_idx * 4];
             x[base_idx * 4 + 1] *= d[base_idx * 4 + 1];
         }
         else if (base_idx * 4 < n)
         {
             // Handle 1 remaining element
             x[base_idx * 4] *= d[base_idx * 4];
         }
     }
 }

/*
    GPU Kernel
    Computes z = x - r - double2 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    axpby_kernel_double2(const local_int_t n, double* x, double* y, double* z)
{
    const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const local_int_t stride = blockDim.x * gridDim.x;
    
    // Process 2 rounds of double2 elements per thread
    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t base_idx = gid + round * stride;
        
        // Use double2 for vectorized loads/stores when possible
        if (base_idx * 2 + 1 < n)
        {
            double2 x_vec = *reinterpret_cast<double2*>(&x[base_idx * 2]);
            double2 y_vec = *reinterpret_cast<double2*>(&y[base_idx * 2]);
            
            double2 z_vec;
            z_vec.x = x_vec.x - y_vec.x;
            z_vec.y = x_vec.y - y_vec.y;
            
            *reinterpret_cast<double2*>(&z[base_idx * 2]) = z_vec;
        }
        else if (base_idx * 2 < n)
        {
            // Handle remaining element individually
            z[base_idx * 2] = x[base_idx * 2] - y[base_idx * 2];
        }
    }
}

/*
    GPU Kernel
    Computes z = x - r - double4 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    axpby_kernel_double4(const local_int_t n, double* x, double* y, double* z)
{
    const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const local_int_t stride = blockDim.x * gridDim.x;
    
    // Process ROUNDS rounds of double4 elements per thread
    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t base_idx = gid + round * stride;
        
        // Use double4 for vectorized loads/stores when possible
        if (base_idx * 4 + 3 < n)
        {
            double4 x_vec = *reinterpret_cast<double4*>(&x[base_idx * 4]);
            double4 y_vec = *reinterpret_cast<double4*>(&y[base_idx * 4]);
            
            double4 z_vec;
            z_vec.x = x_vec.x - y_vec.x;
            z_vec.y = x_vec.y - y_vec.y;
            z_vec.z = x_vec.z - y_vec.z;
            z_vec.w = x_vec.w - y_vec.w;
            
            *reinterpret_cast<double4*>(&z[base_idx * 4]) = z_vec;
        }
        else if (base_idx * 4 + 2 < n)
        {
            // Handle 3 remaining elements
            z[base_idx * 4] = x[base_idx * 4] - y[base_idx * 4];
            z[base_idx * 4 + 1] = x[base_idx * 4 + 1] - y[base_idx * 4 + 1];
            z[base_idx * 4 + 2] = x[base_idx * 4 + 2] - y[base_idx * 4 + 2];
        }
        else if (base_idx * 4 + 1 < n)
        {
            // Handle 2 remaining elements
            z[base_idx * 4] = x[base_idx * 4] - y[base_idx * 4];
            z[base_idx * 4 + 1] = x[base_idx * 4 + 1] - y[base_idx * 4 + 1];
        }
        else if (base_idx * 4 < n)
        {
            // Handle 1 remaining element
            z[base_idx * 4] = x[base_idx * 4] - y[base_idx * 4];
        }
    }
}

/*
    GPU Kernel
    Computes z += x * y - double2 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    spFma_kernel_double2(const local_int_t n, double* x, double* y, double* z)
{
    const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const local_int_t stride = blockDim.x * gridDim.x;
    
    // Process 2 rounds of double2 elements per thread
    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t base_idx = gid + round * stride;
        
        // Use double2 for vectorized loads/stores when possible
        if (base_idx * 2 + 1 < n)
        {
            double2 x_vec = *reinterpret_cast<double2*>(&x[base_idx * 2]);
            double2 y_vec = *reinterpret_cast<double2*>(&y[base_idx * 2]);
            double2 z_vec = *reinterpret_cast<double2*>(&z[base_idx * 2]);
            
            z_vec.x += x_vec.x * y_vec.x;
            z_vec.y += x_vec.y * y_vec.y;
            
            *reinterpret_cast<double2*>(&z[base_idx * 2]) = z_vec;
        }
        else if (base_idx * 2 < n)
        {
            // Handle remaining element individually
            z[base_idx * 2] += x[base_idx * 2] * y[base_idx * 2];
        }
    }
}

/*
    GPU Kernel
    Computes z += x * y - double4 version
*/
template<int THREADS_PER_CTA, int ROUNDS>
__global__ void __launch_bounds__(THREADS_PER_CTA)
    spFma_kernel_double4(const local_int_t n, double* x, double* y, double* z)
{
    const local_int_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const local_int_t stride = blockDim.x * gridDim.x;
    
    // Process ROUNDS rounds of double4 elements per thread
    #pragma unroll
    for (int round = 0; round < ROUNDS; ++round)
    {
        local_int_t base_idx = gid + round * stride;
        
        // Use double4 for vectorized loads/stores when possible
        if (base_idx * 4 + 3 < n)
        {
            double4 x_vec = *reinterpret_cast<double4*>(&x[base_idx * 4]);
            double4 y_vec = *reinterpret_cast<double4*>(&y[base_idx * 4]);
            double4 z_vec = *reinterpret_cast<double4*>(&z[base_idx * 4]);
            
            z_vec.x += x_vec.x * y_vec.x;
            z_vec.y += x_vec.y * y_vec.y;
            z_vec.z += x_vec.z * y_vec.z;
            z_vec.w += x_vec.w * y_vec.w;
            
            *reinterpret_cast<double4*>(&z[base_idx * 4]) = z_vec;
        }
        else if (base_idx * 4 + 2 < n)
        {
            // Handle 3 remaining elements
            z[base_idx * 4] += x[base_idx * 4] * y[base_idx * 4];
            z[base_idx * 4 + 1] += x[base_idx * 4 + 1] * y[base_idx * 4 + 1];
            z[base_idx * 4 + 2] += x[base_idx * 4 + 2] * y[base_idx * 4 + 2];
        }
        else if (base_idx * 4 + 1 < n)
        {
            // Handle 2 remaining elements
            z[base_idx * 4] += x[base_idx * 4] * y[base_idx * 4];
            z[base_idx * 4 + 1] += x[base_idx * 4 + 1] * y[base_idx * 4 + 1];
        }
        else if (base_idx * 4 < n)
        {
            // Handle 1 remaining element
            z[base_idx * 4] += x[base_idx * 4] * y[base_idx * 4];
        }
    }
}

/*
    Multiplies x values with d and accumultaes back to x
    Calls spmvDiag_kernel
*/
void SpmvDiagCuda(local_int_t n, double* x, double* d)
{
    const int ROUNDS = 1;
    const int THREADS_PER_CTA = 256;
    if (g_config.VECTOR_WIDTH == 4) {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 4; // 4 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        spmvDiag_kernel_double4<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, x, d);
    } else {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 2; // 2 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        spmvDiag_kernel_double2<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, x, d);
    }
}

/*
     Computes z = x - r
     Calls axpby_kernel
*/
void AxpbyCuda(local_int_t n, double* x, double* y, double* z)
{
    const int ROUNDS = 1;
    const int THREADS_PER_CTA = 256;
    if (g_config.VECTOR_WIDTH == 4) {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 4; // 4 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        axpby_kernel_double4<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, x, y, z);
    } else {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 2; // 2 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        axpby_kernel_double2<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, x, y, z);
    }
}

/*
    Computes z += x * y
    Calls spFma_kernel
*/
void SpFmaCuda(local_int_t n, double* x, double* y, double* z)
{
    const int ROUNDS = 1;
    const int THREADS_PER_CTA = 256;
    if (g_config.VECTOR_WIDTH == 4) {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 4; // 4 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        spFma_kernel_double4<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, x, y, z);
    } else {
        const int ELELEMENTS_PER_CTA = THREADS_PER_CTA * ROUNDS * 2; // 2 doubles per thread, # rounds per thread
        const int grid = (n + ELELEMENTS_PER_CTA - 1) / ELELEMENTS_PER_CTA;
        spFma_kernel_double2<THREADS_PER_CTA, ROUNDS><<<grid, THREADS_PER_CTA, 0, stream>>>(n, x, y, z);
    }
}

///////// CG Support Kernels: External Matrix SpMV + Scatter //////////////////
/*
    GPU Kernel
    SpMV of the external matrix, in CSR format, and vector x
    Scatter and permutes the results to the original y
*/
template <int THREADS_PER_CTA, int NTHREADS, int UNROLL>
__global__ void __launch_bounds__(THREADS_PER_CTA) extMv_kernel(const local_int_t n, slice_ptr_t* csr_offsets,
    local_int_t* columns, double* values, double alpha, double* x, double* y, local_int_t* ref2opt, local_int_t* map)
{

    enum
    {
        THREADS_PER_WARP = THREADS_PER_CTA / NTHREADS
    };

    namespace cg = cooperative_groups;
    auto warp = cg::tiled_partition<THREADS_PER_WARP>(cg::this_thread_block());

    const local_int_t tidx = threadIdx.x % THREADS_PER_WARP;
    const local_int_t tidy = threadIdx.x / THREADS_PER_WARP;
    const local_int_t row = blockIdx.x * NTHREADS + tidy;

    const local_int_t lrow = blockIdx.x * NTHREADS + tidy;

    const slice_ptr_t str = row < n ? csr_offsets[lrow] : 0;
    const slice_ptr_t end = row < n ? csr_offsets[lrow + 1] : 0;
    double sum = 0.0;
    columns += str + tidx;
    values += str + tidx;
    slice_ptr_t last = end - str - tidx;
#pragma unroll
    for (int i = 0; i < UNROLL; i++)
    {
        int pred = last > 0;
        local_int_t col = pred ? __ldcs(columns) : 0;
        double val = pred ? __ldcs(values) : 0.0;
        double xv = pred ? x[col] : 0.0;
        sum += val * xv;
        columns += THREADS_PER_WARP;
        values += THREADS_PER_WARP;
        last -= THREADS_PER_WARP;
    }
    sum = cg::reduce(warp, sum, cg::plus<double>());

    if (lrow < n && tidx == 0)
    {
        y[ref2opt[map[lrow]]] += alpha * sum;
    }
}

/*
    SpMV of the external matrix, in CSR format, and vector x
    Scatter and permutes the results to the original y
    Calls extMv_kernel
*/
void ExtSpMVCuda(SparseMatrix& A, double alpha, double* x, double* y)
{
    local_int_t rows = A.gpuAux.compressNumberOfRows;

    const int BlockSize = 128;
    const int ROWS_PER_BLOCK = 32;
    const local_int_t grid = (rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    extMv_kernel<BlockSize, ROWS_PER_BLOCK, 8><<<grid, BlockSize, 0, stream>>>(
        rows, A.csrExtOffsets, A.csrExtColumns, A.csrExtValues, alpha, x, y, A.ref2opt, A.gpuAux.map);
}

//////////////////////// Transfer Problem to CPU //////////////////////////////
/*
    Copies A matrix and b, x, and xexact vectors from GPU to CPU
*/
size_t CopyDataToHostCuda(SparseMatrix& A_in, Vector* b, Vector* x, Vector* xexact)
{
    SparseMatrix* A = &A_in;

    double* bv = 0;
    double* xv = 0;
    double* xexactv = 0;
    if (b != 0)
        bv = b->values;
    if (x != 0)
        xv = x->values;
    if (xexact != 0)
        xexactv = xexact->values;

    if (b != 0)
        CHECK_CUDART(cudaMemcpy(bv, b->values_d, sizeof(double) * A->localNumberOfRows, cudaMemcpyDeviceToHost));
    if (x != 0)
        CHECK_CUDART(cudaMemcpy(xv, x->values_d, sizeof(double) * A->localNumberOfRows, cudaMemcpyDeviceToHost));
    if (xexact != 0)
        CHECK_CUDART(cudaMemcpy(xexactv, xexact->values_d, sizeof(double) * A->localNumberOfRows, cudaMemcpyDeviceToHost));

    local_int_t numberOfMgLevels = 4;
    for (int level = 0; level < numberOfMgLevels; ++level)
    {
        local_int_t localNumberOfRows = A->localNumberOfRows;
        local_int_t numberOfNonzerosPerRow = HPCG_MAX_ROW_LEN;

        local_int_t* nonzerosInRow = new local_int_t[localNumberOfRows];
        local_int_t** mtxIndL = new local_int_t*[localNumberOfRows];
        double** matrixValues = new double*[localNumberOfRows];
        double** matrixDiagonal = new double*[localNumberOfRows];

        A->localToGlobalMap.resize(localNumberOfRows);

        const slice_ptr_t totalEntries = (slice_ptr_t) localNumberOfRows * numberOfNonzerosPerRow;
        mtxIndL[0] = new local_int_t[totalEntries];
        matrixValues[0] = new double[totalEntries];

        memset(mtxIndL[0], 0x00, sizeof(local_int_t) * totalEntries);
        memset(matrixValues[0], 0x00, sizeof(double) * totalEntries);

        CHECK_CUDART(cudaMemcpy(mtxIndL[0], A->gpuAux.columns,
            sizeof(local_int_t) * (slice_ptr_t) A->localNumberOfRows * HPCG_MAX_ROW_LEN,
            cudaMemcpyDeviceToHost));
        CHECK_CUDART(cudaMemcpy(matrixValues[0], A->gpuAux.values,
            sizeof(double) * (slice_ptr_t) A->localNumberOfRows * HPCG_MAX_ROW_LEN,
            cudaMemcpyDeviceToHost));
        CHECK_CUDART(cudaMemcpy(
            nonzerosInRow, A->gpuAux.nnzPerRow, sizeof(local_int_t) * A->localNumberOfRows, cudaMemcpyDeviceToHost));

        slice_ptr_t* diagonalIdx = new slice_ptr_t[localNumberOfRows];
        memset(diagonalIdx, 0x00, sizeof(slice_ptr_t) * (localNumberOfRows));
        CHECK_CUDART(cudaMemcpy(diagonalIdx, A->gpuAux.diagonalIdx, sizeof(slice_ptr_t) * localNumberOfRows, cudaMemcpyDeviceToHost));

        memset(&(A->localToGlobalMap[0]), 0x00, sizeof(global_int_t) * (localNumberOfRows));
        CHECK_CUDART(cudaMemcpy(&(A->localToGlobalMap[0]), A->gpuAux.localToGlobalMap, sizeof(global_int_t) * localNumberOfRows,
            cudaMemcpyDeviceToHost));

#ifndef HPCG_NO_OPENMP
#pragma omp parallel for
#endif
        for (auto i = 0; i < localNumberOfRows; ++i)
        {
            mtxIndL[i] = mtxIndL[0] + (slice_ptr_t) i * HPCG_MAX_ROW_LEN;
            matrixValues[i] = matrixValues[0] + (slice_ptr_t) i * HPCG_MAX_ROW_LEN;
            matrixDiagonal[i] = matrixValues[0] + diagonalIdx[i];
        }
        delete[] diagonalIdx;

        A->nonzerosInRow = nonzerosInRow;
        A->mtxIndL = mtxIndL;
        A->matrixValues = matrixValues;
        A->matrixDiagonal = matrixDiagonal;
        A = A->Ac;
    }

    // Estimate Reference Cpu Memory
    // Borrowed from ReportResults.cpp
    local_int_t fnrow = A_in.localNumberOfRows;
    const SparseMatrix* Af = &A_in;
    double numberOfNonzerosPerRow
        = 27.0; // We are approximating a 27-point finite element/volume/difference 3D stencil

    double fnbytes = ((double) sizeof(Geometry));           // Geometry struct in main.cpp
    //fnbytes += ((double) sizeof(double) * fNumberOfCgSets); // testnorms_data in main.cpp

    // Model for GenerateProblem_ref.cpp
    fnbytes += fnrow * sizeof(char);                                             // array nonzerosInRow
    fnbytes += fnrow * ((double) sizeof(global_int_t*));                         // mtxIndG
    fnbytes += fnrow * ((double) sizeof(local_int_t*));                          // mtxIndL
    fnbytes += fnrow * ((double) sizeof(double*));                               // matrixValues
    fnbytes += fnrow * ((double) sizeof(double*));                               // matrixDiagonal
    fnbytes += fnrow * numberOfNonzerosPerRow * ((double) sizeof(local_int_t));  // mtxIndL[1..nrows]
    fnbytes += fnrow * numberOfNonzerosPerRow * ((double) sizeof(double));       // matrixValues[1..nrows]
    //fnbytes += fnrow * numberOfNonzerosPerRow * ((double) sizeof(global_int_t)); // mtxIndG[1..nrows]
    fnbytes += fnrow * ((double) 3 * sizeof(double));                            // x, b, xexact

    // Model for CGData.hpp
    double fncol = ((global_int_t) A_in.localNumberOfColumns);
    fnbytes += fnrow * ((double) 2 * sizeof(double)); // r, Ap
    fnbytes += fncol * ((double) 2 * sizeof(double)); // z, p

    std::vector<double> fnbytesPerLevel(numberOfMgLevels); // Count byte usage per level (level 0 is main CG level)
    fnbytesPerLevel[0] = fnbytes;

    Af = A_in.Ac;
    for (int i = 1; i < numberOfMgLevels; ++i)
    {
        double fnrow_Af = Af->localNumberOfRows;
        double fncol_Af = ((global_int_t) Af->localNumberOfColumns);
        double fnbytes_Af = 0.0;
        // Model for GenerateCoarseProblem.cpp
        fnbytes_Af += fnrow_Af * ((double) sizeof(local_int_t)); // f2cOperator
        fnbytes_Af += fnrow_Af * ((double) sizeof(double));      // rc
        fnbytes_Af += 2.0 * fncol_Af
            * ((double) sizeof(double)); // xc, Axf are estimated based on the size of these arrays on rank 0
        fnbytes_Af += ((double) (sizeof(Geometry) + sizeof(SparseMatrix) + 3 * sizeof(Vector)
            + sizeof(MGData))); // Account for structs geomc, Ac, rc, xc, Axf - (minor)

        // Model for GenerateProblem.cpp (called within GenerateCoarseProblem.cpp)
        fnbytes_Af += fnrow_Af * sizeof(char);                                             // array nonzerosInRow
        fnbytes_Af += fnrow_Af * ((double) sizeof(global_int_t*));                         // mtxIndG
        fnbytes_Af += fnrow_Af * ((double) sizeof(local_int_t*));                          // mtxIndL
        fnbytes_Af += fnrow_Af * ((double) sizeof(double*));                               // matrixValues
        fnbytes_Af += fnrow_Af * ((double) sizeof(double*));                               // matrixDiagonal
        fnbytes_Af += fnrow_Af * numberOfNonzerosPerRow * ((double) sizeof(local_int_t));  // mtxIndL[1..nrows]
        fnbytes_Af += fnrow_Af * numberOfNonzerosPerRow * ((double) sizeof(double));       // matrixValues[1..nrows]
        //fnbytes_Af += fnrow_Af * numberOfNonzerosPerRow * ((double) sizeof(global_int_t)); // mtxIndG[1..nrows]

// Model for SetupHalo_ref.cpp
#ifndef HPCG_NO_MPI
        fnbytes_Af += ((double) sizeof(double) * Af->totalToBeSent);              // sendBuffer
        fnbytes_Af += ((double) sizeof(local_int_t) * Af->totalToBeSent);         // elementsToSend
        fnbytes_Af += ((double) sizeof(int) * Af->numberOfSendNeighbors);         // neighbors
        fnbytes_Af += ((double) sizeof(local_int_t) * Af->numberOfSendNeighbors); // receiveLength, sendLength
#endif
        fnbytesPerLevel[i] = fnbytes_Af;
        fnbytes += fnbytes_Af; // Running sum
        Af = Af->Ac;           // Go to next coarse level
    }

    return fnbytes;
}



template<int ThreadsPerCTA, int Unroll>
__global__ __launch_bounds__(ThreadsPerCTA)
void ex_spsv_sell_single_color_v1_kernel(int slice_size,
                                      int color_str,
                                      int color_end,
                                      double* __restrict__ x,
                                      const double* __restrict__ y,
                                      slice_ptr_t* slice_offsets,
                                      const local_int_t* __restrict__ col_idx,
                                      const double* __restrict__ values,
                                      double* inv_d_values,
                                      double alpha) 
{
    auto row_original_id = blockIdx.x * ThreadsPerCTA + threadIdx.x + color_str;
    int row_inblock_id = row_original_id % slice_size;
    int row_block_id   = row_original_id / slice_size;
    double  sell_sum       = 0.0f;
    if (row_original_id < color_end) {
        slice_ptr_t  row_start_index = slice_offsets[row_block_id] + row_inblock_id;
        // Per-slice nnz is bounded by slice_size*HPCG_MAX_ROW_LEN (fits in int).
        // Cast to int BEFORE the divide so we get a 32-bit idiv instead of the
        // slow emulated 64-bit idiv, since slice_ptr_t is long long.
        int slice_nnz       = (int)(slice_offsets[row_block_id + 1] -
                                    slice_offsets[row_block_id]);
        int max_row_len     = slice_nnz / slice_size;
        auto           max_nnz       = max_row_len;
        const local_int_t*   cols    = col_idx + row_start_index;
        const double*  vals          = values + row_start_index;
        int            unroll_nz     = (max_nnz + Unroll - 1) / Unroll;

        local_int_t col[Unroll];
        double  a_val[Unroll];

        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            if (K < max_nnz) {
                col[K]   = __ldcs(cols);
                a_val[K] = __ldcs(vals);
            }
            else {
                col[K]   = -1;
                a_val[K] = 0.0f;
            }
            cols += (slice_size);
            vals += (slice_size);
        }
        max_nnz -= Unroll;

        double b_val[Unroll];
        #pragma unroll 1
        for (auto q = 1; q < unroll_nz - 1; q++) {
            #pragma unroll Unroll
            for (auto K = 0; K < Unroll; K++) {
                if (col[K] >= 0) {
                    b_val[K] = x[col[K]];
                    sell_sum = a_val[K] * b_val[K] + sell_sum;
                }

                col[K]    = __ldcs(cols);
                a_val[K]  = __ldcs(vals);
                cols     += (slice_size);
                vals     += (slice_size);
            }
            max_nnz -= Unroll;
        }
        if (1 < unroll_nz) {
            #pragma unroll Unroll
            for (auto K = 0; K < Unroll; K++) {
                if (col[K] >= 0) {
                    b_val[K] = x[col[K]];
                    sell_sum = a_val[K] * b_val[K] + sell_sum;
                }
                a_val[K] = 0.0f;
                if (0 < max_nnz) {
                    col[K]   = __ldcs(cols);
                    a_val[K] = __ldcs(vals);
                }
                max_nnz -= 1;
                cols    += (slice_size);
                vals    += (slice_size);
            }
        }
        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            if (col[K] >= 0) {
                b_val[K] = x[col[K]];
                sell_sum = a_val[K] * b_val[K] + sell_sum;
            }
        }

        auto diag          = __ldcs(&inv_d_values[row_original_id]);
        auto b_store       = y[row_original_id];
        b_store            = ((alpha * b_store) - sell_sum) / diag;
        x[row_original_id] = b_store;
    }
}

// Helper functions and macros for TMA kernels (always compiled since TMA is runtime-configurable)
#define SWAP(x, y) do { \
    typeof(x) temp = x; \
    x = y; \
    y = temp; \
} while(0)

#define MIN(a, b) ((a) < (b) ? (a) : (b))
// Separate warp0 detection logic
__device__ __forceinline__ bool is_warp_zero()
{
    unsigned int tid = threadIdx.x;
    unsigned int warp_id = tid / 32;
    unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // Broadcast from lane 0
    return uniform_warp_id == 0;
}

__device__ __forceinline__ bool is_thread_zero()
{
    uint32_t elected;
    asm("{"
            ".reg .pred %p;"
            "elect.sync _|%p, 0xFFFFFFFF;"
            "selp.b32 %0, 1, 0, %p;"
        "}"
        : "=r"(elected)

    );
    return static_cast<bool>(elected);
}


inline __device__ void cp_async_bulk_prefetch_L2(const void *global_ptr, const uint32_t num_bytes) {
    asm volatile(
        "cp.async.bulk.prefetch.L2.global [%0], %1;"
        :
        : "l"(__cvta_generic_to_global(global_ptr)),
          "r"(num_bytes)
        :
    );
}

//TMA version for ex_spsv_sell_single_color_v1_kernel (single color)
template<int BlockSize, int Unroll>
__global__ __launch_bounds__(BlockSize)
void ex_spsv_sell_single_color_v1_tma_kernel(
                const __grid_constant__ CUtensorMap             cols_tensor_map,
                const __grid_constant__ CUtensorMap             vals_tensor_map,
                int                                             slice_size,
                int                                             color_str,
                int                                             color_end,
                double*                                         x,
                const double*                                   y,
                const slice_ptr_t*                              slice_offsets,
                const local_int_t*                              col_idx,
                const double*                                   values,
                double*                                         inv_d_values,
                double                                          alpha,
                slice_ptr_t                                     base)
{
    auto row_original_id    = blockIdx.x * BlockSize + threadIdx.x + color_str;
    
    int row_inblock_id = row_original_id % slice_size;
    int row_block_id   = row_original_id / slice_size;

    // TMA infrastructure
    __shared__ uint64_t bar[8];
    uint64_t* bar_curr = &bar[0];
    uint64_t* bar_next = &bar[1];

    __shared__ char __attribute__((aligned(128))) a_vals_buf[Unroll * BlockSize * sizeof(double)];
    __shared__ char __attribute__((aligned(128))) a_cols_buf[Unroll * BlockSize * sizeof(int)];
    __shared__ char __attribute__((aligned(128))) b_val_buf[Unroll * BlockSize * sizeof(double)];
    __shared__ char __attribute__((aligned(128))) b_cols_buf[Unroll * BlockSize * sizeof(int)];

    double* curr_vals_buf_ptr = reinterpret_cast<double*>(a_vals_buf);
    int* curr_cols_buf_ptr = reinterpret_cast<int*>(a_cols_buf);
    double* next_vals_buf_ptr = reinterpret_cast<double*>(b_val_buf);
    int* next_cols_buf_ptr = reinterpret_cast<int*>(b_cols_buf);

    __shared__ slice_ptr_t row_start_index_shared, row_end_index_shared;
    
    //if (row_original_id >= color_end) return;
    
    if(is_warp_zero() && is_thread_zero()) {
        ptx::mbarrier_init(bar_curr, 1);
        ptx::mbarrier_init(bar_next, 1);
        ptx::fence_proxy_async(ptx::space_shared);

        row_start_index_shared = slice_offsets[row_block_id] - base;
        row_end_index_shared   = slice_offsets[row_block_id + 1] - base;
    }
    __syncthreads();

    int stage           = 0;
    int x_offset        = blockIdx.x % (slice_size/BlockSize);
    auto row_start_index = row_inblock_id % BlockSize;
    // Cast to int BEFORE the divide to avoid an emulated 64-bit idiv, since
    // slice_ptr_t is long long.
    int slice_nnz        = (int)(row_end_index_shared - row_start_index_shared);
    int max_row_len      = slice_nnz / slice_size;
    auto sell_sum        = double{};
    auto unroll_nz       = (max_row_len + Unroll - 1) / Unroll;

    double b_val[Unroll];

    //Prefetch: Assumes at least unroll_nz >= 1
    if(is_warp_zero() && is_thread_zero()) {
        int num_elements = BlockSize * Unroll;
        int32_t coords[2] = {x_offset * BlockSize, (int32_t)(row_start_index_shared/slice_size)};
        ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, curr_vals_buf_ptr, &vals_tensor_map, coords, bar_curr);
        ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, curr_cols_buf_ptr, &cols_tensor_map, coords, bar_curr);
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_curr, num_elements * (sizeof(double) + sizeof(int)));
    }

    stage ^= 1;
    __syncthreads();
    int parity = 0;

    #pragma unroll 1
    for (auto q = 1; q < unroll_nz - 1; q++) {

        if (is_warp_zero() && is_thread_zero()) {
            int num_elements = BlockSize * Unroll;
            int32_t coords[2] = {x_offset * BlockSize, (int32_t)(row_start_index_shared/slice_size) + q * Unroll};
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_vals_buf_ptr, &vals_tensor_map, coords, bar_next);
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_cols_buf_ptr, &cols_tensor_map, coords, bar_next);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_next, num_elements * (sizeof(double) + sizeof(int)));
        }

        stage ^= 1;
        while(!ptx::mbarrier_try_wait_parity(bar_curr, parity)) {}
        parity ^= stage;
        
        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            int col_id = row_start_index + K * BlockSize;
            if(curr_cols_buf_ptr[col_id] >= 0) {
                b_val[K] = x[curr_cols_buf_ptr[col_id]];
                sell_sum += curr_vals_buf_ptr[col_id] * b_val[K];
            }
        }

        __syncthreads();

        SWAP(bar_curr, bar_next);
        SWAP(curr_vals_buf_ptr, next_vals_buf_ptr);
        SWAP(curr_cols_buf_ptr, next_cols_buf_ptr);
    
        max_row_len -= Unroll;
    }
    
    if (1 < unroll_nz) {
       max_row_len -= Unroll;
       if (is_warp_zero() && is_thread_zero()) {
            int32_t coords[2] = {x_offset * BlockSize, (int32_t)(row_start_index_shared/slice_size) + (unroll_nz - 1) * Unroll};
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_vals_buf_ptr, &vals_tensor_map, coords, bar_next);
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_cols_buf_ptr, &cols_tensor_map, coords, bar_next);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_next, BlockSize * Unroll * (sizeof(double) + sizeof(int)));
        }

        stage ^= 1;
        while(!ptx::mbarrier_try_wait_parity(bar_curr, parity)) {}
        parity ^= stage;

        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            int col_id = row_start_index + K * BlockSize;
            if(curr_cols_buf_ptr[col_id] >= 0) {
                b_val[K] = x[curr_cols_buf_ptr[col_id]];
                sell_sum += curr_vals_buf_ptr[col_id] * b_val[K];
            }
        }

        SWAP(bar_curr, bar_next);
        SWAP(curr_vals_buf_ptr, next_vals_buf_ptr);
        SWAP(curr_cols_buf_ptr, next_cols_buf_ptr);
    }

    stage ^= 1;
    while(!ptx::mbarrier_try_wait_parity(bar_curr, parity)) {}
    parity ^= stage;

    #pragma unroll Unroll
    for (auto K = 0; K < Unroll; K++) {
        int col_id = row_start_index + K * BlockSize;
        if(curr_cols_buf_ptr[col_id] >= 0 && K < max_row_len) {
            b_val[K] = x[curr_cols_buf_ptr[col_id]];
            sell_sum += curr_vals_buf_ptr[col_id] * b_val[K];
        }
    }

    if(row_original_id < color_end) {       
        auto diag          = __ldcs(&inv_d_values[row_original_id]);
        auto b_store       = y[row_original_id];
        b_store            = ((alpha * b_store) - sell_sum) / diag;
        x[row_original_id] = b_store;
    }
}

template<int BlockSize, int Unroll>
__global__ __launch_bounds__(BlockSize)
void sellmv_v1_2D_kernel(
    int                         m,
    double                      alpha,
    double                      beta,
    int                         slice_size,
    const slice_ptr_t* __restrict__  d_sell_offsets,
    const local_int_t* __restrict__  d_columns,
    const double* __restrict__  d_values,
    const double* __restrict__  d_X,
    double* __restrict__        d_Y) {

    auto tx                 = threadIdx.x;
    auto ty                 = blockIdx.x;
    local_int_t col[Unroll];
    double a_val[Unroll];
    double b_val[Unroll];

    // Grid launch guarantees gridDim.y * BlockSize >= m/8, so each thread
    // handles at most one row. Matches the cuSPARSE sellmv_v1_2D_kernel shape.
    auto row_original_id    = tx + blockDim.x * blockIdx.y;
    if (row_original_id < m / 8) {
        row_original_id         += ty * (m / 8);
        auto row_in_slice_id    = row_original_id % slice_size;
        auto row_slice_id       = row_original_id / slice_size;
        slice_ptr_t row_start_index = d_sell_offsets[row_slice_id] + row_in_slice_id;
        // Per-slice nnz fits in int; cast BEFORE divide to avoid the emulated
        // 64-bit idiv, since slice_ptr_t is long long.
        int slice_nnz         = (int)(d_sell_offsets[row_slice_id + 1] - d_sell_offsets[row_slice_id]);
        double sell_sum       = 0.0f;
        int max_row_len       = slice_nnz / slice_size;
        const local_int_t *cols = d_columns + row_start_index;
        const double *vals       = d_values  + row_start_index;
        int unroll_nz          = (max_row_len + Unroll - 1) / Unroll;

        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            if (K < max_row_len) {
                col[K]   = __ldcs(cols);
                a_val[K] = __ldcs(vals);
            }
            else {
                col[K]   = -1;
                a_val[K] = 0.0f;
            }
            cols+=(slice_size); vals+=(slice_size);
        }
        max_row_len -= Unroll;


        #pragma unroll 1
        for (auto q = 1; q < unroll_nz - 1; q++) {
            #pragma unroll Unroll
            for (auto K = 0; K < Unroll; K++) {
                if(col[K] >= 0) {
                    b_val[K] = d_X[col[K]];
                    sell_sum = a_val[K] * b_val[K] + sell_sum;
                }

                col[K]   = __ldcs(cols);
                a_val[K] = __ldcs(vals);
                cols+=(slice_size); vals+=(slice_size);
            }
            max_row_len -= Unroll;
        }
        if (1 < unroll_nz) {
            #pragma unroll Unroll
            for (auto K = 0; K < Unroll; K++) {
                if(col[K] >= 0) {
                    b_val[K] = d_X[col[K]];
                    sell_sum = a_val[K] * b_val[K] + sell_sum;
                }
                a_val[K] = 0.0f;
                if (0 < max_row_len) {
                    col[K] = __ldcs(cols);
                    a_val[K] = __ldcs(vals);
                }
                max_row_len--;
                cols+=(slice_size); vals+=(slice_size);
            }
        }

        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            if(col[K] >= 0) {
                b_val[K] = d_X[col[K]];
                sell_sum = a_val[K] * b_val[K] + sell_sum;
            }
        }

        if (beta == 0.0) {
            d_Y[row_original_id] = alpha * sell_sum;
        }
        else {
            double result = beta * d_Y[row_original_id]
                        + alpha * sell_sum;
            d_Y[row_original_id] = result;
        }
    }
}

//Assumes a 2D Matrix (Slice size x Number of padded rows)
template<int BlockSize, int Unroll>
__global__ __launch_bounds__(BlockSize)
void sellmv_v1_tma_2D_tensor_kernel_double_int32(
                const __grid_constant__ CUtensorMap             cols_tensor_map,
                const __grid_constant__ CUtensorMap             vals_tensor_map,
                int                     m,
                 double                  alpha,
                 double                  beta,
                 int                     slice_size,
                 const slice_ptr_t*            d_sell_offsets,
                 const local_int_t*            d_columns,
                 const double*                 d_values,
                 slice_ptr_t             base,
                 const double*                 d_X,
                 double*                       d_Y) 
                
{
    auto is_beta_zero       = beta == 0.0;
    auto tx                 = threadIdx.x;
    auto ty                 = blockIdx.x;
    auto row_original_id    = tx + blockDim.x*blockIdx.y;
    row_original_id         += ty * m/8;

    auto row_in_slice_id    = row_original_id % slice_size;
    auto row_slice_id       = row_original_id / slice_size;
   

    // TMA infrastructure - Use static shared memory
    __shared__ uint64_t bar[8];
    uint64_t* bar_curr = &bar[0];
    uint64_t* bar_next = &bar[1];

    __shared__ char __attribute__((aligned(128))) a_vals_buf[Unroll * BlockSize * sizeof(double)];
    __shared__ char __attribute__((aligned(128))) a_cols_buf[Unroll * BlockSize * sizeof(int)];
    __shared__ char __attribute__((aligned(128))) b_val_buf[Unroll * BlockSize * sizeof(double)];
    __shared__ char __attribute__((aligned(128))) b_cols_buf[Unroll * BlockSize * sizeof(int)];

    double* curr_vals_buf_ptr = reinterpret_cast<double*>(a_vals_buf);
    int* curr_cols_buf_ptr = reinterpret_cast<int*>(a_cols_buf);
    double* next_vals_buf_ptr = reinterpret_cast<double*>(b_val_buf);
    int* next_cols_buf_ptr = reinterpret_cast<int*>(b_cols_buf);

    __shared__ slice_ptr_t row_start_index_shared, row_end_index_shared;
    if(is_warp_zero() && is_thread_zero()) {
        ptx::mbarrier_init(bar_curr, 1);
        ptx::mbarrier_init(bar_next, 1);
        ptx::fence_proxy_async(ptx::space_shared);

        row_start_index_shared    = d_sell_offsets[row_slice_id] - base;
        row_end_index_shared      = d_sell_offsets[row_slice_id + 1] - base;
    }

    __syncthreads();


    int stage           = 0;
    int x_offset            = blockIdx.y % (slice_size/BlockSize);
    auto row_start_index    = row_in_slice_id % (BlockSize);
    // Cast to int BEFORE the divide to avoid an emulated 64-bit idiv, since
    // slice_ptr_t is long long.
    int slice_nnz           = (int)(row_end_index_shared - row_start_index_shared);
    int max_row_len         = slice_nnz / slice_size;
    auto sell_sum           = double{};
    auto unroll_nz          = (max_row_len + Unroll - 1) / Unroll;

    //Prefetch: Assumes at least unroll_nz >= 1
    if(is_warp_zero() && is_thread_zero()) {
        int num_elements = BlockSize * Unroll;
        int32_t coords[2] = {x_offset * BlockSize, (int32_t)(row_start_index_shared/slice_size)};
        ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, curr_vals_buf_ptr, &vals_tensor_map, coords, bar_curr);
        ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, curr_cols_buf_ptr, &cols_tensor_map, coords, bar_curr);
        ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_curr, num_elements * (sizeof(double) + sizeof(int)));
    }

    //Stage and parity represent two step barrier
    stage ^= 1;
    __syncthreads();
    int parity = 0;

    #pragma unroll 1
    for (auto q = 1; q < unroll_nz - 1; q++) {

        if (is_warp_zero() && is_thread_zero()) {
            int num_elements = BlockSize * Unroll;
            int32_t coords[2] = {x_offset * BlockSize, (int32_t)(row_start_index_shared/slice_size) + q * (Unroll)};
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_vals_buf_ptr, &vals_tensor_map, coords, bar_next);
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_cols_buf_ptr, &cols_tensor_map, coords, bar_next);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_next, num_elements * (sizeof(double) + sizeof(int)));
        }

        stage ^= 1;
        while(!ptx::mbarrier_try_wait_parity(bar_curr, parity)) {}
        parity ^= stage;
        
        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            int col_id = row_start_index + K * BlockSize;
            //FMA
            if(curr_cols_buf_ptr[col_id] >= 0) {
                double b_val = d_X[curr_cols_buf_ptr[col_id]];
                sell_sum += curr_vals_buf_ptr[col_id] * b_val;
            }
        }

        //This one is critical to avoid race conditions
        __syncthreads();

        SWAP(bar_curr, bar_next);
        SWAP(curr_vals_buf_ptr, next_vals_buf_ptr);
        SWAP(curr_cols_buf_ptr, next_cols_buf_ptr);
    
        max_row_len -= Unroll;
    }
    if (1 < unroll_nz) {
        max_row_len -= Unroll;
        if (is_warp_zero() && is_thread_zero()) {
            int32_t coords[2] = {x_offset * BlockSize, (int32_t)(row_start_index_shared/slice_size) + (unroll_nz - 1) * Unroll};
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_vals_buf_ptr, &vals_tensor_map, coords, bar_next);
            ptx::cp_async_bulk_tensor(ptx::space_shared, cuda::ptx::space_global, next_cols_buf_ptr, &cols_tensor_map, coords, bar_next);
            ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta, ptx::space_shared, bar_next, BlockSize * Unroll * (sizeof(double) + sizeof(int)));
            
        }

        stage ^= 1;
        while(!ptx::mbarrier_try_wait_parity(bar_curr, parity)) {}
        parity ^= stage;

        #pragma unroll Unroll
        for (auto K = 0; K < Unroll; K++) {
            int col_id = row_start_index + K * BlockSize;
            if(curr_cols_buf_ptr[col_id] >= 0) {
                double b_val = d_X[curr_cols_buf_ptr[col_id]];
                sell_sum += curr_vals_buf_ptr[col_id] * b_val;
            }
        }

        SWAP(bar_curr, bar_next);
        SWAP(curr_vals_buf_ptr, next_vals_buf_ptr);
        SWAP(curr_cols_buf_ptr, next_cols_buf_ptr);
    }

    stage ^= 1;
    while(!ptx::mbarrier_try_wait_parity(bar_curr, parity)) {}
    parity ^= stage;

    #pragma unroll Unroll
    for (auto K = 0; K < Unroll; K++) {
        int col_id = row_start_index + K * BlockSize;
        if(curr_cols_buf_ptr[col_id] >= 0 && K < max_row_len) {
            double b_val = d_X[curr_cols_buf_ptr[col_id]];
            sell_sum += curr_vals_buf_ptr[col_id] * b_val;
        }
    }

    if (is_beta_zero) {
        d_Y[row_original_id] = alpha * sell_sum;
    }
    else {
        double result = beta * d_Y[row_original_id] + alpha * sell_sum;
        d_Y[row_original_id] = result;
    }
}

enum DIR{Forward = 0, Backward = 1, General = 2};

// Template dispatch helper for sv_sell with TMA (with device-side color loop when possible)
template<int THREADS_PER_CTA, int SV_UNROLL>
void sv_sell_tma_dispatch(DIR d, const SparseMatrix & A, double *rv, double *xv, 
                           int color_size, local_int_t rows, local_int_t grid,
                           slice_ptr_t *sell_block_offset, local_int_t *sell_columns, double *sell_values, slice_ptr_t last_nnz) {
    CUtensorMap cols_tensor_map{};
    CUtensorMap vals_tensor_map{};

    constexpr uint32_t rank = 2;
    int slice_size = A.slice_size;
    uint64_t size[rank] = {(uint64_t)slice_size, (uint64_t)last_nnz/slice_size};
    uint64_t stride1[rank - 1] = {slice_size * sizeof(int)};
    uint64_t stride2[rank - 1] = {slice_size * sizeof(double)};
    uint32_t box_size[rank] = {THREADS_PER_CTA, SV_UNROLL};
    uint32_t elem_stride[rank] = {1, 1};

    cuTensorMapEncodeTiled(&cols_tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank, (void*)sell_columns, size, stride1, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    cuTensorMapEncodeTiled(&vals_tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        rank, (void*)sell_values, size, stride2, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    if(d == Forward) {
        for (int color = 0; color < A.totalColors; color++) {
            auto color_str = color * color_size;
            auto color_end = std::min<local_int_t>((color + 1) * color_size, rows);
            ex_spsv_sell_single_color_v1_tma_kernel<THREADS_PER_CTA, SV_UNROLL><<<grid, THREADS_PER_CTA, 0, stream>>>(
                cols_tensor_map, vals_tensor_map,
                A.slice_size, color_str, color_end, xv, rv, 
                sell_block_offset, sell_columns, sell_values, A.diagonal, 1.0, 0);        
        }
    }
    else {
        for (int color = A.totalColors - 1; color >= 0; color--) {
            auto color_str = color * color_size;
            auto color_end = std::min<local_int_t>((color + 1) * color_size, rows);
            ex_spsv_sell_single_color_v1_tma_kernel<THREADS_PER_CTA, SV_UNROLL><<<grid, THREADS_PER_CTA, 0, stream>>>(
                cols_tensor_map, vals_tensor_map,
                A.slice_size, color_str, color_end, xv, rv, 
                sell_block_offset, sell_columns, sell_values, A.diagonal, 1.0, 0);  
        }
    }
}

// Template dispatch helper for sv_sell without TMA (host-side color loop for better performance)
template<int THREADS_PER_CTA, int SV_UNROLL>
void sv_sell_no_tma_dispatch(DIR d, const SparseMatrix & A, double *rv, double *xv, 
                               int color_size, local_int_t rows, local_int_t grid,
                               slice_ptr_t *sell_block_offset, local_int_t *sell_columns, double *sell_values) {
    if(d == Forward) {
        for (int color = 0; color < A.totalColors; color++) {
            auto color_str = color * color_size;
            auto color_end = std::min<local_int_t>((color + 1) * color_size, rows);
            ex_spsv_sell_single_color_v1_kernel<THREADS_PER_CTA, SV_UNROLL><<<grid, THREADS_PER_CTA, 0, stream>>>(
                A.slice_size, color_str, color_end,  xv, rv, 
                sell_block_offset, sell_columns, sell_values, A.diagonal, 1.0);        
        }
    }
    else {
        for (int color = A.totalColors - 1; color >= 0; color--) {
            auto color_str = color * color_size;
            auto color_end = std::min<local_int_t>((color + 1) * color_size, rows);
            ex_spsv_sell_single_color_v1_kernel<THREADS_PER_CTA, SV_UNROLL><<<grid, THREADS_PER_CTA, 0, stream>>>(
                A.slice_size, color_str, color_end,  xv, rv, 
                sell_block_offset, sell_columns, sell_values, A.diagonal, 1.0);  
        }
    }
}

// Runtime dispatch macro for sv_sell configurations
#define SV_DISPATCH(BLOCK_SIZE, UNROLL, ...) \
    if (g_config.SV_BLOCK_SIZE == BLOCK_SIZE && g_config.SV_UNROLL == UNROLL) { \
        __VA_ARGS__(BLOCK_SIZE, UNROLL); \
        return; \
    }

bool SpsvTmaSell(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, cudaStream_t stream);
bool SpsvTmaSellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, cudaStream_t stream, int blk, int unroll, int rpt);
bool SpsvLdgV2SellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, cudaStream_t stream, int blk, int unroll, int w);
bool SpsvTma2dSellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, slice_ptr_t last_nnz, cudaStream_t stream, int blk, int unroll,
    int rpt);
bool SpsvLdgV3SellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, cudaStream_t stream, int blk, int unroll, int w, bool wide,
    bool cached);
bool SpsvTmaExSellCfg(int forward, const SparseMatrix& A, double* rv, double* xv, slice_ptr_t* sell_block_offset,
    local_int_t* sell_columns, double* sell_values, slice_ptr_t last_nnz, cudaStream_t stream, int blk, int unroll,
    int rpt);

// Duplicated from SvKernelKind in CudaKernels.hpp, which is how the autotuner
// sees these numbers. The header cannot simply be included here: this file also
// defines KernelConfig, g_config and DIR independently of it. So a family added
// in one place must be added in the other, or one number means two things in two
// translation units and nothing warns. Keep kNumSvKernelKinds in step too.
enum
{
    SV_KIND_LDG = 0,
    SV_KIND_TMA = 1,
    SV_KIND_LDGV2 = 2,
    SV_KIND_TMA2D = 3,
    SV_KIND_LDGV3 = 4,
    SV_KIND_TMAEX = 5
};

// LDG3 packs four things into rpt: W, the widest-load request in the sign, the
// cache policy in the hundreds digit, and the MV row-partition count in the
// thousands. Mirrors SvRptWidth / SvRptWide / SvRptCached / SvRptPart in
// CudaKernels.hpp, which carries the full description; this file does not
// include that header, so the arithmetic is repeated here and the two must be
// edited together.
//
// Cached tests one digit rather than ">= 100". The older form was equivalent
// while the hundreds digit was the top of the field, but reads any partition
// value as cached now that a digit sits above it.
static inline int LdgV3Width(int rpt) { return (rpt < 0 ? -rpt : rpt) % 100; }
static inline bool LdgV3Wide(int rpt) { return rpt < 0; }
static inline bool LdgV3Cached(int rpt) { return ((rpt < 0 ? -rpt : rpt) / 100) % 10 != 0; }
static inline int LdgV3Part(int rpt)
{
    const int f = ((rpt < 0 ? -rpt : rpt) / 1000) % 10;
    return f == 0 ? 8 : (1 << (f - 1));
}

namespace {
constexpr int kMaxSvLevels = 8;
struct SvChoice { bool set; int kind; int blk; int unroll; int rpt; };
SvChoice g_sv_choice[kMaxSvLevels] = {};

// HPCG_PIN_MV / HPCG_PIN_SV name one configuration as "kind,blk,unroll,rpt" and
// apply it to every level, taking precedence over the autotuner.
//
// This exists to answer a question the sweep cannot: when a kernel is edited,
// the sweep may respond by picking a different configuration, so a before/after
// comparison conflates the edit with the search. Pinning holds the
// configuration fixed and isolates the edit.
struct Pin { bool set; int kind; int blk; int unroll; int rpt; };

Pin ParsePin(const char* name)
{
    Pin p{};
    const char* v = std::getenv(name);
    if (!v || !*v)
        return p;
    if (std::sscanf(v, "%d,%d,%d,%d", &p.kind, &p.blk, &p.unroll, &p.rpt) != 4)
    {
        std::fprintf(stderr, "ERROR: %s must be \"kind,blk,unroll,rpt\", got \"%s\"\n", name, v);
        std::exit(1);
    }
    p.set = true;
    return p;
}

// A launcher returns false for a configuration it cannot serve, and the caller
// then falls through to a different kernel entirely. Under a pin that is the
// worst kind of failure: the run still produces a plausible number, but for a
// kernel nobody asked for. PinRefused turns it into an abort, and the pin is
// announced so the log can be checked against the intent.
bool g_mv_pinned = false;
bool g_sv_pinned = false;

void PinRefused(const char* what, int level, int kind, int blk, int unroll, int rpt)
{
    std::fprintf(stderr,
        "ERROR: %s was pinned to kind=%d blk=%d unroll=%d rpt=%d, but no launcher\n"
        "       accepts that configuration at level %d. Falling through would have\n"
        "       measured a different kernel under the pinned name.\n",
        what, kind, blk, unroll, rpt, level);
    std::exit(1);
}
} // namespace

// This section (pin/choice bookkeeping, the SV/MV autotune timers, and the
// sv_sell/mv_sell dispatchers themselves) launches the explicit LDG/TMA kernel
// families directly against the SparseMatrix's sliced-ELL device arrays. Those
// arrays are typed pointers only under EXPLICIT_KERNELS; under the default
// build (cuSPARSE/NVPL via IndexMode) they are void*, and ComputeSPMV.cpp /
// ComputeSYMGS.cpp / autotune.cpp already only call into this section under
// EXPLICIT_KERNELS.
#ifdef EXPLICIT_KERNELS

bool ApplyPin(const Pin& p, const char* what, bool& announced, int& kind, int& blk, int& unroll, int& rpt)
{
    if (!p.set)
        return false;
    if (!announced)
    {
        std::fprintf(stderr, "%s pinned to kind=%d blk=%d unroll=%d rpt=%d on all levels\n", what, p.kind, p.blk,
            p.unroll, p.rpt);
        announced = true;
    }
    kind = p.kind;
    blk = p.blk;
    unroll = p.unroll;
    rpt = p.rpt;
    return true;
}

void SetSvChoice(int level, int kind, int blk, int unroll, int rpt)
{
    if (level >= 0 && level < kMaxSvLevels)
        g_sv_choice[level] = SvChoice{true, kind, blk, unroll, rpt};
}

static bool GetSvChoice(int level, int& kind, int& blk, int& unroll, int& rpt)
{
    static const Pin pin = ParsePin("HPCG_PIN_SV");
    static bool announced = false;
    if (ApplyPin(pin, "SV", announced, kind, blk, unroll, rpt))
    {
        g_sv_pinned = true;
        return true;
    }

    if (level >= 0 && level < kMaxSvLevels && g_sv_choice[level].set)
    {
        kind = g_sv_choice[level].kind;
        blk = g_sv_choice[level].blk;
        unroll = g_sv_choice[level].unroll;
        rpt = g_sv_choice[level].rpt;
        return true;
    }
    return false;
}

static bool DispatchSvNoTma(int blk, int unroll, DIR d, const SparseMatrix& A, double* rv, double* xv, int color_size,
    local_int_t rows, local_int_t grid, slice_ptr_t* off, local_int_t* cols, double* vals)
{
#define C(B, U)                                                                                                        \
    if (blk == (B) && unroll == (U))                                                                                   \
    {                                                                                                                  \
        sv_sell_no_tma_dispatch<B, U>(d, A, rv, xv, color_size, rows, grid, off, cols, vals);                          \
        return true;                                                                                                   \
    }
    C(64, 4) C(64, 6) C(64, 7) C(64, 8) C(64, 10) C(64, 12) C(64, 14) C(64, 16)
    C(128, 4) C(128, 6) C(128, 7) C(128, 8) C(128, 10) C(128, 12) C(128, 14) C(128, 16)
    C(256, 4) C(256, 6) C(256, 7) C(256, 8) C(256, 10) C(256, 12) C(256, 14) C(256, 16)
#undef C
    return false;
}

bool SvSellCfg(DIR d, const SparseMatrix& A, double* rv, double* xv, int kind, int blk, int unroll, int rpt)
{
    slice_ptr_t* off;
    local_int_t* cols;
    double* vals;
    slice_ptr_t last_nnz;
    if (d == Forward)
    {
        off = A.sellLSliceMrl;
        cols = A.sellLPermColumns;
        vals = A.sellLPermValues;
        last_nnz = A.sellLLocalNumberOfNonzeros;
    }
    else
    {
        off = A.sellUSliceMrl;
        cols = A.sellUPermColumns;
        vals = A.sellUPermValues;
        last_nnz = A.sellULocalNumberOfNonzeros;
    }
    const local_int_t rows = A.localNumberOfRows;
    const int color_size = (rows + A.totalColors - 1) / A.totalColors;
    if (kind == SV_KIND_TMA)
        return SpsvTmaSellCfg(d == Forward, A, rv, xv, off, cols, vals, stream, blk, unroll, rpt);
    if (kind == SV_KIND_LDGV2)
        return SpsvLdgV2SellCfg(d == Forward, A, rv, xv, off, cols, vals, stream, blk, unroll, rpt);
    if (kind == SV_KIND_TMA2D)
        return SpsvTma2dSellCfg(d == Forward, A, rv, xv, off, cols, vals, last_nnz, stream, blk, unroll, rpt);
    if (kind == SV_KIND_LDGV3)
        return SpsvLdgV3SellCfg(
            d == Forward, A, rv, xv, off, cols, vals, stream, blk, unroll, LdgV3Width(rpt), LdgV3Wide(rpt),
            LdgV3Cached(rpt));
    if (kind == SV_KIND_TMAEX)
        return SpsvTmaExSellCfg(d == Forward, A, rv, xv, off, cols, vals, last_nnz, stream, blk, unroll, rpt);
    const local_int_t grid = (color_size + blk - 1) / blk;
    return DispatchSvNoTma(blk, unroll, d, A, rv, xv, color_size, rows, grid, off, cols, vals);
}

float TimeSvConfig(
    const SparseMatrix& A, double* rv, double* xv, int kind, int blk, int unroll, int rpt, int iters, int sweep)
{
    const bool do_fwd = sweep != 2;
    const bool do_bwd = sweep != 1;

    // sweep==0 keeps the original probe exactly: check Forward's feasibility,
    // run Backward once unchecked. Isolated modes (1, 2) check whichever
    // direction they will actually time, since a backward-only call probing
    // Forward would be checking a sweep it never runs.
    if (sweep == 0)
    {
        if (!SvSellCfg(Forward, A, rv, xv, kind, blk, unroll, rpt))
            return -1.0f;
        SvSellCfg(Backward, A, rv, xv, kind, blk, unroll, rpt);
    }
    else if (do_fwd)
    {
        if (!SvSellCfg(Forward, A, rv, xv, kind, blk, unroll, rpt))
            return -1.0f;
    }
    else if (!SvSellCfg(Backward, A, rv, xv, kind, blk, unroll, rpt))
        return -1.0f;
    const cudaError_t le = cudaGetLastError();
    const cudaError_t se = cudaStreamSynchronize(stream);
    if (le != cudaSuccess || se != cudaSuccess)
    {
        cudaGetLastError();
        return -1.0f;
    }

    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg, stream);
    for (int i = 0; i < iters; ++i)
    {
        if (do_fwd)
            SvSellCfg(Forward, A, rv, xv, kind, blk, unroll, rpt);
        if (do_bwd)
            SvSellCfg(Backward, A, rv, xv, kind, blk, unroll, rpt);
    }
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, beg, end);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return ms / iters;
}

void sv_sell(DIR d, const SparseMatrix & A, double *rv, double *xv) {
    local_int_t rows = A.localNumberOfRows;
    int color_size = (rows + A.totalColors - 1) / A.totalColors;
    const local_int_t grid = (color_size + g_config.SV_BLOCK_SIZE - 1) / g_config.SV_BLOCK_SIZE;

    slice_ptr_t *sell_block_offset;
    local_int_t *sell_columns;
    double      *sell_values;
    slice_ptr_t  last_nnz = 0;
    
    if(d == Forward) {
        sell_block_offset = A.sellLSliceMrl;
        sell_columns = A.sellLPermColumns;
        sell_values  = A.sellLPermValues;
        last_nnz = A.sellLLocalNumberOfNonzeros;
    }
    else {
        sell_block_offset = A.sellUSliceMrl;
        sell_columns = A.sellUPermColumns;
        sell_values  = A.sellUPermValues;
        last_nnz = A.sellULocalNumberOfNonzeros;
    }

    {
        int a_kind, a_blk, a_unroll, a_rpt;
        if (GetSvChoice(A.level, a_kind, a_blk, a_unroll, a_rpt)) {
            if (SvSellCfg(d, A, rv, xv, a_kind, a_blk, a_unroll, a_rpt))
                return;
            if (g_sv_pinned)
                PinRefused("SV", A.level, a_kind, a_blk, a_unroll, a_rpt);
        }
    }

    static const bool use_spsv_tma = [] {
        const char* e = std::getenv("USE_SPSV_TMA");
        return e && std::atoi(e) != 0;
    }();
    if (use_spsv_tma) {
        if (SpsvTmaSell(d == Forward, A, rv, xv, sell_block_offset, sell_columns, sell_values, stream))
            return;
    }

    // Dispatch based on runtime configuration
    // Try common configurations
    if(g_config.USE_TMA_SV) {
        int m = A.localNumberOfRows;
        bool can_use_tma_2D = ((m/A.totalColors) % A.slice_size) == 0;
        if(can_use_tma_2D) {
            #define CALL_SV_TMA(BLOCK, UNROLL) sv_sell_tma_dispatch<BLOCK, UNROLL>(d, A, rv, xv, color_size, rows, grid, sell_block_offset, sell_columns, sell_values, last_nnz)
            SV_DISPATCH(64, 4, CALL_SV_TMA);
            SV_DISPATCH(64, 6, CALL_SV_TMA);
            SV_DISPATCH(64, 7, CALL_SV_TMA);
            SV_DISPATCH(64, 8, CALL_SV_TMA);
            SV_DISPATCH(64, 10, CALL_SV_TMA);
            SV_DISPATCH(64, 12, CALL_SV_TMA);
            SV_DISPATCH(128, 4, CALL_SV_TMA);
            SV_DISPATCH(128, 6, CALL_SV_TMA);
            SV_DISPATCH(128, 8, CALL_SV_TMA);
            SV_DISPATCH(256, 4, CALL_SV_TMA);
            SV_DISPATCH(256, 6, CALL_SV_TMA);
            SV_DISPATCH(256, 7, CALL_SV_TMA); //High shared memory usage
            #undef CALL_SV_TMA
        }
        else { //Since matrix is divded into 8 colors, we cannot use TMA for small matrices that does not have slice_size as its width
            //Fallback to no TMA
            #define CALL_SV_NO_TMA(BLOCK, UNROLL) sv_sell_no_tma_dispatch<BLOCK, UNROLL>(d, A, rv, xv, color_size, rows, grid, sell_block_offset, sell_columns, sell_values)
            SV_DISPATCH(64, 4, CALL_SV_NO_TMA);
            #undef CALL_SV_NO_TMA
        }
    }
    
    #define CALL_SV_NO_TMA(BLOCK, UNROLL) sv_sell_no_tma_dispatch<BLOCK, UNROLL>(d, A, rv, xv, color_size, rows, grid, sell_block_offset, sell_columns, sell_values)
    SV_DISPATCH(64, 4, CALL_SV_NO_TMA);
    SV_DISPATCH(64, 6, CALL_SV_NO_TMA);
    SV_DISPATCH(64, 7, CALL_SV_NO_TMA);
    SV_DISPATCH(64, 8, CALL_SV_NO_TMA);
    SV_DISPATCH(64, 10, CALL_SV_NO_TMA);
    SV_DISPATCH(64, 12, CALL_SV_NO_TMA);
    SV_DISPATCH(64, 14, CALL_SV_NO_TMA);
    SV_DISPATCH(64, 16, CALL_SV_NO_TMA);
    SV_DISPATCH(128, 4, CALL_SV_NO_TMA);
    SV_DISPATCH(128, 6, CALL_SV_NO_TMA);
    SV_DISPATCH(128, 8, CALL_SV_NO_TMA);
    SV_DISPATCH(128, 10, CALL_SV_NO_TMA);
    SV_DISPATCH(128, 12, CALL_SV_NO_TMA);
    SV_DISPATCH(128, 14, CALL_SV_NO_TMA);
    SV_DISPATCH(128, 16, CALL_SV_NO_TMA);
    SV_DISPATCH(256, 4, CALL_SV_NO_TMA);
    SV_DISPATCH(256, 6, CALL_SV_NO_TMA);
    SV_DISPATCH(256, 8, CALL_SV_NO_TMA);
    SV_DISPATCH(256, 10, CALL_SV_NO_TMA);
    SV_DISPATCH(256, 12, CALL_SV_NO_TMA);
    SV_DISPATCH(256, 14, CALL_SV_NO_TMA);
    SV_DISPATCH(256, 16, CALL_SV_NO_TMA);
    #undef CALL_SV_NO_TMA
    
    // If we reach here, configuration is not supported
    fprintf(stderr, "ERROR: Unsupported SV configuration: BLOCK_SIZE=%d, UNROLL=%d\n", 
            g_config.SV_BLOCK_SIZE, g_config.SV_UNROLL);
}

// Template dispatch helper for mv_sell with TMA
template<int BLOCK_SIZE, int UNROLL>
void mv_sell_tma_dispatch(DIR d, const SparseMatrix & A, double alpha, double beta, double *x, double *y,
                           local_int_t m, dim3 grid2D,
                           slice_ptr_t *sell_block_offset, local_int_t *sell_columns, double *sell_values, slice_ptr_t last_nnz) {
    CUtensorMap cols_tensor_map{};
    CUtensorMap vals_tensor_map{};

    constexpr uint32_t rank = 2;
    int slice_size = A.slice_size;
    uint64_t size[rank] = {(uint64_t)slice_size, (uint64_t)last_nnz/slice_size};
    uint64_t stride1[rank - 1] = {slice_size * sizeof(int)};
    uint64_t stride2[rank - 1] = {slice_size * sizeof(double)};
    uint32_t box_size[rank] = {BLOCK_SIZE, UNROLL};
    uint32_t elem_stride[rank] = {1, 1};

    cuTensorMapEncodeTiled(&cols_tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank, (void*)sell_columns, size, stride1, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    
    cuTensorMapEncodeTiled(&vals_tensor_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        rank, (void*)sell_values, size, stride2, box_size, elem_stride,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

    // Calculate dynamic shared memory size with 128-byte alignment
    // constexpr size_t alignment = 128;
    // size_t shared_mem_size = 0;
    
    // Row indices (2 x int)
    // size_t row_indices_size = 2 * sizeof(int);
    // shared_mem_size += row_indices_size;
    
    // Align to 8 bytes for uint64_t barriers
    // shared_mem_size = ((shared_mem_size + 7) / 8) * 8;
    // size_t bar_array_size = 8 * sizeof(uint64_t);
    // shared_mem_size += bar_array_size;
    
    // Align to 128 bytes for TMA buffers
    // size_t before_tma_alignment = shared_mem_size;
    // shared_mem_size = ((shared_mem_size + alignment - 1) / alignment) * alignment;
    
    // Calculate TMA buffer sizes with 128-byte alignment
    // constexpr size_t a_vals_size = UNROLL * BLOCK_SIZE * sizeof(double);
    // constexpr size_t a_cols_size = UNROLL * BLOCK_SIZE * sizeof(int);
    // constexpr size_t b_val_size = UNROLL * BLOCK_SIZE * sizeof(double);
    // constexpr size_t b_cols_size = UNROLL * BLOCK_SIZE * sizeof(int);
    
    // constexpr size_t a_vals_aligned = ((a_vals_size + alignment - 1) / alignment) * alignment;
    // constexpr size_t a_cols_aligned = ((a_cols_size + alignment - 1) / alignment) * alignment;
    // constexpr size_t b_val_aligned = ((b_val_size + alignment - 1) / alignment) * alignment;
    // constexpr size_t b_cols_aligned = ((b_cols_size + alignment - 1) / alignment) * alignment;
    
    // size_t tma_buffers_size = a_vals_aligned + a_cols_aligned + b_val_aligned + b_cols_aligned;
    // shared_mem_size += tma_buffers_size;
    
   
    sellmv_v1_tma_2D_tensor_kernel_double_int32<BLOCK_SIZE, UNROLL>
        <<< grid2D, BLOCK_SIZE, 0, stream >>>
        (cols_tensor_map, vals_tensor_map, m, alpha, beta, A.slice_size, sell_block_offset, sell_columns, sell_values, 0, x, y);
}

// Template dispatch helper for mv_sell without TMA
template<int BLOCK_SIZE, int UNROLL>
void mv_sell_no_tma_dispatch(DIR d, const SparseMatrix & A, double alpha, double beta, double *x, double *y,
                               local_int_t m, dim3 grid2D,
                               slice_ptr_t *sell_block_offset, local_int_t *sell_columns, double *sell_values) {

        //Find number of blocks per SM
        int num_blocks_per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, sellmv_v1_2D_kernel<BLOCK_SIZE, UNROLL>, BLOCK_SIZE, 0);
        // number of SMs    
        int num_sms = 0;
        cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
        //printf("Number of blocks per SM: %d\n", num_blocks_per_sm);

        auto active_blocks = num_blocks_per_sm * num_sms;

        dim3 grid(8, (active_blocks + 8 - 1)/8, 1); 

    sellmv_v1_2D_kernel<BLOCK_SIZE, UNROLL>
        <<< grid2D, BLOCK_SIZE, 0, stream >>>
        (m, alpha, beta, A.slice_size, sell_block_offset, sell_columns, sell_values, x, y);
}

// Runtime dispatch macro for mv_sell configurations
#define MV_DISPATCH(BLOCK_SIZE, UNROLL, ...) \
    if (mv_block_size == BLOCK_SIZE && mv_unroll == UNROLL) { \
        __VA_ARGS__(BLOCK_SIZE, UNROLL); \
        return; \
    }

bool MvTmaSellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y, slice_ptr_t* off,
    local_int_t* cols, double* vals, cudaStream_t stream, int blk, int unroll, int rpt);
bool MvLdgV2SellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y, slice_ptr_t* off,
    local_int_t* cols, double* vals, cudaStream_t stream, int blk, int unroll, int w, int parts);
bool MvTma2dSellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y, slice_ptr_t* off,
    local_int_t* cols, double* vals, slice_ptr_t last_nnz, cudaStream_t stream, int blk, int unroll, int rpt);
bool MvLdgV3SellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y, slice_ptr_t* off,
    local_int_t* cols, double* vals, cudaStream_t stream, int blk, int unroll, int w, bool wide, bool cached, int part);
bool MvTmaExSellCfg(const SparseMatrix& A, double alpha, double beta, const double* x, double* y, slice_ptr_t* off,
    local_int_t* cols, double* vals, slice_ptr_t last_nnz, cudaStream_t stream, int blk, int unroll, int rpt);

namespace {
struct MvChoice { bool set; int kind; int blk; int unroll; int rpt; int parts; };
MvChoice g_mv_choice[kMaxSvLevels] = {};

/*
    The L and U multiplies inside SymGS, kept apart from the full-matrix choice.

    Each carries half the nonzeros of A, and the shape that wins on A is not the
    one that wins on either triangle: on a Rubin uGPU at 512x512x288 the full
    matrix takes 64/7/2p16 at L0 while L takes 128/4/4wp8 and U 128/4/4wcp8, so
    running them on A's choice costs 6.3% and 3.8% there and 5.0% over all eight
    slots of the hierarchy.

    Nothing here needs a wider record: rpt already carries the access width and
    the cache policy in its sign and hundreds digit, which is what LdgV3Wide and
    LdgV3Cached read, so -104 names the wide cached W=4 shape those winners use.

    Indexed by DIR, so only Forward and Backward are stored; General is the full
    matrix and already has g_mv_choice. Unset falls back to it, leaving every
    run that does not sweep the directions exactly as it was.
*/
MvChoice g_mv_choice_dir[2][kMaxSvLevels] = {};
}

void SetMvChoice(int level, int kind, int blk, int unroll, int rpt, int parts)
{
    if (level >= 0 && level < kMaxSvLevels)
        g_mv_choice[level] = MvChoice{true, kind, blk, unroll, rpt, parts};
}

void SetMvChoiceDir(int level, DIR d, int kind, int blk, int unroll, int rpt, int parts)
{
    if (level >= 0 && level < kMaxSvLevels && (d == Forward || d == Backward))
        g_mv_choice_dir[d][level] = MvChoice{true, kind, blk, unroll, rpt, parts};
}

static bool GetMvChoice(int level, DIR d, int& kind, int& blk, int& unroll, int& rpt, int& parts)
{
    static const Pin pin = ParsePin("HPCG_PIN_MV");
    static bool announced = false;
    if (ApplyPin(pin, "MV", announced, kind, blk, unroll, rpt))
    {
        g_mv_pinned = true;
        return true;
    }

    // A choice made for this triangle specifically outranks the full-matrix
    // one, which is otherwise what L and U inherit. Only a selecting
    // directional sweep sets these, so this is a no-op for every other run.
    if ((d == Forward || d == Backward) && level >= 0 && level < kMaxSvLevels
        && g_mv_choice_dir[d][level].set)
    {
        const MvChoice& c = g_mv_choice_dir[d][level];
        kind = c.kind;
        blk = c.blk;
        unroll = c.unroll;
        rpt = c.rpt;
        parts = c.parts;
        return true;
    }

    if (level >= 0 && level < kMaxSvLevels && g_mv_choice[level].set)
    {
        kind = g_mv_choice[level].kind;
        blk = g_mv_choice[level].blk;
        unroll = g_mv_choice[level].unroll;
        rpt = g_mv_choice[level].rpt;
        parts = g_mv_choice[level].parts;
        return true;
    }
    return false;
}

static bool DispatchMvNoTma(int blk, int unroll, DIR d, const SparseMatrix& A, double alpha, double beta, double* x,
    double* y, local_int_t m, dim3 grid2D, slice_ptr_t* off, local_int_t* cols, double* vals)
{
#define C(B, U)                                                                                                        \
    if (blk == (B) && unroll == (U))                                                                                   \
    {                                                                                                                  \
        mv_sell_no_tma_dispatch<B, U>(d, A, alpha, beta, x, y, m, grid2D, off, cols, vals);                            \
        return true;                                                                                                   \
    }
    C(64, 1) C(64, 4) C(64, 6) C(64, 7) C(64, 8) C(64, 10) C(64, 12) C(64, 14) C(64, 16)
    C(128, 1) C(128, 4) C(128, 6) C(128, 7) C(128, 8) C(128, 10) C(128, 12) C(128, 14) C(128, 16)
    C(256, 1) C(256, 4) C(256, 6) C(256, 7) C(256, 8) C(256, 10) C(256, 12) C(256, 14) C(256, 16)
#undef C
    return false;
}

bool MvSellCfg(DIR d, const SparseMatrix& A, double alpha, double beta, double* x, double* y, int kind, int blk,
    int unroll, int rpt, int parts)
{
    slice_ptr_t* off = A.sellASliceMrl;
    local_int_t* cols = A.sellAPermColumns;
    double* vals = A.sellAPermValues;
    slice_ptr_t last_nnz = A.sellALocalNumberOfNonzeros;
    if (d == Forward)
    {
        off = A.sellLSliceMrl;
        cols = A.sellLPermColumns;
        vals = A.sellLPermValues;
        last_nnz = A.sellLLocalNumberOfNonzeros;
    }
    else if (d == Backward)
    {
        off = A.sellUSliceMrl;
        cols = A.sellUPermColumns;
        vals = A.sellUPermValues;
        last_nnz = A.sellULocalNumberOfNonzeros;
    }
    const local_int_t m = A.localNumberOfRows;
    if (kind == SV_KIND_TMA)
        return MvTmaSellCfg(A, alpha, beta, x, y, off, cols, vals, stream, blk, unroll, rpt);
    if (kind == SV_KIND_LDGV2)
        return MvLdgV2SellCfg(A, alpha, beta, x, y, off, cols, vals, stream, blk, unroll, rpt, parts);
    if (kind == SV_KIND_TMA2D)
        return MvTma2dSellCfg(A, alpha, beta, x, y, off, cols, vals, last_nnz, stream, blk, unroll, rpt);
    if (kind == SV_KIND_LDGV3)
        return MvLdgV3SellCfg(A, alpha, beta, x, y, off, cols, vals, stream, blk, unroll, LdgV3Width(rpt),
            LdgV3Wide(rpt), LdgV3Cached(rpt), LdgV3Part(rpt));
    if (kind == SV_KIND_TMAEX)
        return MvTmaExSellCfg(A, alpha, beta, x, y, off, cols, vals, last_nnz, stream, blk, unroll, rpt);
    dim3 grid2D(8, (unsigned int) ((m / 8 + blk - 1) / blk), 1);
    if (grid2D.y > 65535u)
        return false;
    return DispatchMvNoTma(blk, unroll, d, A, alpha, beta, x, y, m, grid2D, off, cols, vals);
}

static float TimeMvDir(const SparseMatrix& A, double* x, double* y, int kind, int blk, int unroll, int rpt,
    int parts, int iters, DIR d, double alpha, double beta)
{
    if (!MvSellCfg(d, A, alpha, beta, x, y, kind, blk, unroll, rpt, parts))
        return -1.0f;
    const cudaError_t le = cudaGetLastError();
    const cudaError_t se = cudaStreamSynchronize(stream);
    if (le != cudaSuccess || se != cudaSuccess)
    {
        cudaGetLastError();
        return -1.0f;
    }
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);
    cudaEventRecord(beg, stream);
    for (int i = 0; i < iters; ++i)
        MvSellCfg(d, A, alpha, beta, x, y, kind, blk, unroll, rpt, parts);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, beg, end);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    return ms / iters;
}

float TimeMvConfig(
    const SparseMatrix& A, double* x, double* y, int kind, int blk, int unroll, int rpt, int parts, int iters)
{
    return TimeMvDir(A, x, y, kind, blk, unroll, rpt, parts, iters, General, 1.0, 0.0);
}

// The same MV choice serves three different multiplies: A for ComputeSPMV, and
// L and U inside ComputeSYMGS. Only A is used to select it, so the two that run
// in the MG hot path are never measured. This times them as they are actually
// called -- L accumulates (beta=1), U overwrites (beta=0) -- so the tuning gap
// can be seen rather than assumed.
float TimeMvConfigDir(const SparseMatrix& A, double* x, double* y, int kind, int blk, int unroll, int rpt, int parts,
    int iters, int dir)
{
    if (dir == 1)
        return TimeMvDir(A, x, y, kind, blk, unroll, rpt, parts, iters, Forward, 1.0, 1.0);
    if (dir == 2)
        return TimeMvDir(A, x, y, kind, blk, unroll, rpt, parts, iters, Backward, 1.0, 0.0);
    return TimeMvDir(A, x, y, kind, blk, unroll, rpt, parts, iters, General, 1.0, 0.0);
}

void mv_sell(DIR d, const SparseMatrix & A, double alpha, double beta, double *x, double *y) {
    local_int_t m = A.localNumberOfRows;
    int mv_block_size = g_config.MV_BLOCK_SIZE;
    int mv_unroll     = g_config.MV_UNROLL;

    // CUDA hardware limit for gridDim.y is 65535. This check is unconditional:
    // it used to be compiled only under a wide-index build flag, on the theory
    // that a 32-bit-index build could not reach a problem size large enough to
    // exceed the cap. That is no longer true -- nonzero counters/offsets are
    // unconditionally 64-bit (slice_ptr_t, see Geometry.hpp), so nnz no longer
    // overflows first, and local_int_t alone permits ~2^31 rows. 512^3 already
    // needs m/8/128 = 131k > 65535, so compiling this out silently launched an
    // invalid grid. The cost is a couple of host-side integer ops per call.
    constexpr unsigned int MAX_GRID_Y = 65535u;
    auto compute_needed_y = [&](int bs) -> slice_ptr_t {
        return ((slice_ptr_t) m / 8 + bs - 1) / bs;
    };
    slice_ptr_t needed_y = compute_needed_y(mv_block_size);
    if (needed_y > (slice_ptr_t) MAX_GRID_Y) {
        int orig_block = mv_block_size;
        int new_block  = mv_block_size < 512 ? 512 : 1024;
        while (new_block <= 1024) {
            slice_ptr_t ny = compute_needed_y(new_block);
            if (ny <= (slice_ptr_t) MAX_GRID_Y) {
                // fprintf(stderr,
                //     "WARNING: mv_sell needs gridDim.y=%lld (> %u) for m=%d with "
                //     "MV_BLOCK_SIZE=%d; falling back to MV_BLOCK_SIZE=%d.\n",
                //     (long long) needed_y, MAX_GRID_Y, (int) m, orig_block, new_block);
                mv_block_size = new_block;
                needed_y = ny;
                break;
            }
            new_block *= 2;
        }
        if (needed_y > (slice_ptr_t) MAX_GRID_Y) {
            fprintf(stderr,
                "ERROR: mv_sell launch exceeds gridDim.y limit (needed %lld > %u) "
                "for m=%d even with MV_BLOCK_SIZE=1024. Reduce problem size.\n",
                (long long) needed_y, MAX_GRID_Y, (int) m);
            return;
        }
    }
    dim3 grid2D(8, (unsigned int) needed_y, 1);

    slice_ptr_t *sell_block_offset = A.sellASliceMrl;
    local_int_t *sell_columns = A.sellAPermColumns;
    double      *sell_values  = A.sellAPermValues;
    slice_ptr_t  last_nnz = A.sellALocalNumberOfNonzeros;

    if(d == Forward) {
        sell_block_offset = A.sellLSliceMrl;
        sell_columns = A.sellLPermColumns;
        sell_values  = A.sellLPermValues;
        last_nnz = A.sellLLocalNumberOfNonzeros;
    }
    else if(d == Backward) {
        sell_block_offset = A.sellUSliceMrl;
        sell_columns = A.sellUPermColumns;
        sell_values  = A.sellUPermValues;
        last_nnz = A.sellULocalNumberOfNonzeros;
    }

    {
        int a_kind, a_blk, a_unroll, a_rpt, a_parts;
        if (GetMvChoice(A.level, d, a_kind, a_blk, a_unroll, a_rpt, a_parts)) {
            if (MvSellCfg(d, A, alpha, beta, x, y, a_kind, a_blk, a_unroll, a_rpt, a_parts))
                return;
            if (g_mv_pinned)
                PinRefused("MV", A.level, a_kind, a_blk, a_unroll, a_rpt);
        }
    }

    // Dispatch based on runtime configuration
    if(g_config.USE_TMA_MV) {
        #define CALL_MV_TMA(BLOCK, UNROLL) mv_sell_tma_dispatch<BLOCK, UNROLL>(d, A, alpha, beta, x, y, m, grid2D, sell_block_offset, sell_columns, sell_values, last_nnz)
        MV_DISPATCH(64, 1, CALL_MV_TMA);
        MV_DISPATCH(64, 4, CALL_MV_TMA);
        MV_DISPATCH(64, 6, CALL_MV_TMA);
        MV_DISPATCH(64, 7, CALL_MV_TMA);
        MV_DISPATCH(64, 8, CALL_MV_TMA);
        MV_DISPATCH(128, 1, CALL_MV_TMA);
        MV_DISPATCH(128, 4, CALL_MV_TMA);
        MV_DISPATCH(128, 6, CALL_MV_TMA);
        MV_DISPATCH(128, 7, CALL_MV_TMA);
        MV_DISPATCH(128, 8, CALL_MV_TMA);
        MV_DISPATCH(256, 1, CALL_MV_TMA);
        MV_DISPATCH(256, 4, CALL_MV_TMA);
        MV_DISPATCH(256, 6, CALL_MV_TMA);
        MV_DISPATCH(256, 7, CALL_MV_TMA);
        // MV_DISPATCH(256, 8, CALL_MV_TMA);
        // MV_DISPATCH(256, 9, CALL_MV_TMA);
        // MV_DISPATCH(256, 12, CALL_MV_TMA);
        #undef CALL_MV_TMA
        fprintf(stderr, "ERROR: Unsupported MV TMA configuration: BLOCK_SIZE=%d, UNROLL=%d, m=%d\n", 
            mv_block_size, mv_unroll, m);
        return; //No fallback for TMA
    }
    
  
    #define CALL_MV_NO_TMA(BLOCK, UNROLL) mv_sell_no_tma_dispatch<BLOCK, UNROLL>(d, A, alpha, beta, x, y, m, grid2D, sell_block_offset, sell_columns, sell_values)
    MV_DISPATCH(64, 1, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 4, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 6, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 7, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 8, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 10, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 12, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 14, CALL_MV_NO_TMA);
    MV_DISPATCH(64, 16, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 1, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 4, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 6, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 7, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 8, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 10, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 12, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 14, CALL_MV_NO_TMA);
    MV_DISPATCH(128, 16, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 1, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 4, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 6, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 7, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 8, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 10, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 12, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 14, CALL_MV_NO_TMA);
    MV_DISPATCH(256, 16, CALL_MV_NO_TMA);
    MV_DISPATCH(512, 1, CALL_MV_NO_TMA);
    MV_DISPATCH(512, 4, CALL_MV_NO_TMA);
    MV_DISPATCH(512, 6, CALL_MV_NO_TMA);
    MV_DISPATCH(512, 7, CALL_MV_NO_TMA);
    MV_DISPATCH(512, 8, CALL_MV_NO_TMA);
    MV_DISPATCH(1024, 1, CALL_MV_NO_TMA);
    MV_DISPATCH(1024, 4, CALL_MV_NO_TMA);
    MV_DISPATCH(1024, 6, CALL_MV_NO_TMA);
    MV_DISPATCH(1024, 7, CALL_MV_NO_TMA);
    MV_DISPATCH(1024, 8, CALL_MV_NO_TMA);
    #undef CALL_MV_NO_TMA
    
    // If we reach here, configuration is not supported
    fprintf(stderr, "ERROR: Unsupported MV configuration: BLOCK_SIZE=%d, UNROLL=%d\n", 
            mv_block_size, mv_unroll);
}

#endif // EXPLICIT_KERNELS
//////// SV and MV ///////


#endif
