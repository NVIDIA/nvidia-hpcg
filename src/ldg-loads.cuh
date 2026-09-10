/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

/*
 * Wide sliced-ELL gather loads for the LDG_V2 kernels, in both cache policies.
 *
 * Default (.wb, "Wb" below) is what LDG_V2 has always used: ordinary caching
 * loads, widened to 128/256-bit via int4 / __align__(32) double4.
 *
 * Streaming (.cs, "Cs" below) marks the line evict-first. The matrix is read
 * exactly once per call with no reuse, so streaming it stops it from evicting
 * the vector being gathered from -- which IS reused. That is the LDG3 result;
 * the reason it is written in PTX here is that __ldcs/__stcs bottom out at
 * 2-wide for doubles, so LDG3 had to drop from 128/256-bit accesses to pairs of
 * 64-bit ones to get the policy. Issuing the cache operator directly keeps the
 * wide accesses, so a kernel can have both.
 *
 * The 256-bit forms (ld.global.cs.v4.f64, ld.global.cs.v8.u32) are Blackwell and
 * later; sm_90 falls back to a pair of 128-bit accesses, which is what nvcc
 * emitted for an __align__(32) double4 there anyway.
 *
 * The load asm blocks are deliberately NOT volatile and carry no memory clobber.
 * The matrix is read-only for the lifetime of these kernels, so the compiler
 * stays free to hoist and batch them -- that freedom is what keeps the two-stage
 * A/B prefetch pipeline overlapping. A volatile block would pin every load to
 * its program point and serialise exactly what the pipeline exists to hide.
 *
 * Alignment: callers guarantee base_row, in_slice and slice_size are all
 * multiples of W, and slice offsets are multiples of slice_size, so a W-wide
 * access is naturally W*8-byte aligned. That is the same invariant the existing
 * __align__(32) double4 path already relies on.
 *
 * The column paths load .u32, so this header requires 32-bit local_int_t.
 * Callers guard with #ifndef INDEX_64, i.e. wherever columns are 32-bit.
 */

#ifndef LDG_LOADS_CUH
#define LDG_LOADS_CUH

#include "Geometry.hpp"

namespace ldgload
{

struct __align__(32) double4_32a
{
    double x, y, z, w;
};

// ------------------------------------------------ default policy (caching)

template <int W>
__device__ __forceinline__ void LoadColsWb(const local_int_t* __restrict__ p, int (&c)[W])
{
    if constexpr (W == 1)
        c[0] = (int) p[0];
    else if constexpr (W == 2)
    {
        const int2 t = *reinterpret_cast<const int2*>(p);
        c[0] = t.x;
        c[1] = t.y;
    }
    else if constexpr (W == 4)
    {
        const int4 t = *reinterpret_cast<const int4*>(p);
        c[0] = t.x;
        c[1] = t.y;
        c[2] = t.z;
        c[3] = t.w;
    }
    else if constexpr (W == 8)
    {
        const int4 a = *reinterpret_cast<const int4*>(p);
        const int4 b = *reinterpret_cast<const int4*>(p + 4);
        c[0] = a.x;
        c[1] = a.y;
        c[2] = a.z;
        c[3] = a.w;
        c[4] = b.x;
        c[5] = b.y;
        c[6] = b.z;
        c[7] = b.w;
    }
}

template <int W>
__device__ __forceinline__ void LoadValsWb(const double* __restrict__ p, double (&v)[W])
{
    if constexpr (W == 1)
        v[0] = p[0];
    else if constexpr (W == 2)
    {
        const double2 t = *reinterpret_cast<const double2*>(p);
        v[0] = t.x;
        v[1] = t.y;
    }
    else if constexpr (W == 4)
    {
        const double4_32a t = *reinterpret_cast<const double4_32a*>(p);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
    }
    else if constexpr (W == 8)
    {
        const double4_32a a = *reinterpret_cast<const double4_32a*>(p);
        const double4_32a b = *reinterpret_cast<const double4_32a*>(p + 4);
        v[0] = a.x;
        v[1] = a.y;
        v[2] = a.z;
        v[3] = a.w;
        v[4] = b.x;
        v[5] = b.y;
        v[6] = b.z;
        v[7] = b.w;
    }
}

// ------------------------------------------------- streaming policy (.cs)

template <int W>
__device__ __forceinline__ void LoadColsCs(const local_int_t* __restrict__ p, int (&c)[W])
{
    static_assert(sizeof(local_int_t) == 4, "streaming column loads assume 32-bit local_int_t");
    if constexpr (W == 1)
        asm("ld.global.nc.cs.u32 %0, [%1];" : "=r"(c[0]) : "l"(p));
    else if constexpr (W == 2)
        asm("ld.global.nc.cs.v2.u32 {%0,%1}, [%2];" : "=r"(c[0]), "=r"(c[1]) : "l"(p));
    else if constexpr (W == 4)
        asm("ld.global.nc.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
            : "l"(p));
    else if constexpr (W == 8)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        asm("ld.global.nc.cs.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]), "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
            : "l"(p));
#else
        asm("ld.global.nc.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
            : "l"(p));
        asm("ld.global.nc.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
            : "l"(p + 4));
#endif
    }
}

template <int W>
__device__ __forceinline__ void LoadValsCs(const double* __restrict__ p, double (&v)[W])
{
    if constexpr (W == 1)
        asm("ld.global.nc.cs.f64 %0, [%1];" : "=d"(v[0]) : "l"(p));
    else if constexpr (W == 2)
        asm("ld.global.nc.cs.v2.f64 {%0,%1}, [%2];" : "=d"(v[0]), "=d"(v[1]) : "l"(p));
    else if constexpr (W == 4)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        asm("ld.global.nc.cs.v4.f64 {%0,%1,%2,%3}, [%4];"
            : "=d"(v[0]), "=d"(v[1]), "=d"(v[2]), "=d"(v[3])
            : "l"(p));
#else
        asm("ld.global.nc.cs.v2.f64 {%0,%1}, [%2];" : "=d"(v[0]), "=d"(v[1]) : "l"(p));
        asm("ld.global.nc.cs.v2.f64 {%0,%1}, [%2];" : "=d"(v[2]), "=d"(v[3]) : "l"(p + 2));
#endif
    }
    else if constexpr (W == 8)
    {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        asm("ld.global.nc.cs.v4.f64 {%0,%1,%2,%3}, [%4];"
            : "=d"(v[0]), "=d"(v[1]), "=d"(v[2]), "=d"(v[3])
            : "l"(p));
        asm("ld.global.nc.cs.v4.f64 {%0,%1,%2,%3}, [%4];"
            : "=d"(v[4]), "=d"(v[5]), "=d"(v[6]), "=d"(v[7])
            : "l"(p + 4));
#else
        asm("ld.global.nc.cs.v2.f64 {%0,%1}, [%2];" : "=d"(v[0]), "=d"(v[1]) : "l"(p));
        asm("ld.global.nc.cs.v2.f64 {%0,%1}, [%2];" : "=d"(v[2]), "=d"(v[3]) : "l"(p + 2));
        asm("ld.global.nc.cs.v2.f64 {%0,%1}, [%2];" : "=d"(v[4]), "=d"(v[5]) : "l"(p + 4));
        asm("ld.global.nc.cs.v2.f64 {%0,%1}, [%2];" : "=d"(v[6]), "=d"(v[7]) : "l"(p + 6));
#endif
    }
}

// ----------------------------------------------------------- policy select

template <int W, bool STREAM>
__device__ __forceinline__ void LoadCols(const local_int_t* __restrict__ p, int (&c)[W])
{
    if constexpr (STREAM)
        LoadColsCs<W>(p, c);
    else
        LoadColsWb<W>(p, c);
}

template <int W, bool STREAM>
__device__ __forceinline__ void LoadVals(const double* __restrict__ p, double (&v)[W])
{
    if constexpr (STREAM)
        LoadValsCs<W>(p, v);
    else
        LoadValsWb<W>(p, v);
}

// -------------------------------------------------- scalar vector element

// Streaming load of one read-only double (rhs, diag): touched once per call, so
// it should not displace the vector being gathered from. .nc is safe here -- the
// kernel never writes these -- and keeps the read-only path the caching version
// got from __restrict__.
template <bool STREAM>
__device__ __forceinline__ double LoadScalarRo(const double* __restrict__ p)
{
    if constexpr (STREAM)
    {
        double v;
        asm("ld.global.nc.cs.f64 %0, [%1];" : "=d"(v) : "l"(p));
        return v;
    }
    else
        return *p;
}

// Streaming load of a double this kernel also WRITES (y under beta != 0).
// Deliberately no .nc: the non-coherent path is only valid for data not written
// during the kernel, and y is read-modify-written right here.
template <bool STREAM>
__device__ __forceinline__ double LoadScalarRw(const double* p)
{
    if constexpr (STREAM)
    {
        double v;
        asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(v) : "l"(p) : "memory");
        return v;
    }
    else
        return *p;
}

// volatile + memory clobber here is deliberate and ~free: these are the trailing
// writes of the kernel, so pinning them costs no overlap, and it keeps the store
// ordered against the read-modify-write of y under beta != 0.
template <bool STREAM>
__device__ __forceinline__ void StoreScalar(double* p, double v)
{
    if constexpr (STREAM)
        asm volatile("st.global.cs.f64 [%0], %1;" : : "l"(p), "d"(v) : "memory");
    else
        *p = v;
}

} // namespace ldgload

#endif // LDG_LOADS_CUH
