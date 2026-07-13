
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
 @file IndexMode.hpp

 Runtime-selectable index-width support for the GPU Sliced-ELL operator.

 A single binary can choose, at run time (via the --mi flag), how wide the
 Sliced-ELL slice-offset and column-index device arrays are, without a
 recompile. This module is the ONE place that maps an IndexMode to concrete
 element types, byte sizes and cuSPARSE index-type enums, and provides the
 single dispatch helper that turns a runtime mode into compile-time template
 parameters. Keep all mode->type decisions here so the rest of the codebase
 never open-codes a switch over IndexMode.
 */

#ifndef INDEXMODE_HPP
#define INDEXMODE_HPP

#include <cstddef>

#ifdef USE_CUDA
#include <cusparse.h>
#endif

/*!
  Index-width mode for the GPU Sliced-ELL operator arrays.

  The two independent widths are the slice-offset width and the column-index
  width. Only the combinations that are useful in practice are exposed:
   - I32_I32: 32-bit offsets, 32-bit columns (legacy behavior; default).
   - I64_I32: 64-bit offsets, 32-bit columns (mixed indexing).
   - I64_I64: 64-bit offsets, 64-bit columns.
 */
enum class IndexMode : int
{
    I32_I32 = 0,
    I64_I32 = 1,
    I64_I64 = 2
};

//! Concrete C++ element types selected at runtime. We use `int` (4 bytes) and
//! `long long` (8 bytes) to match HPCG's local_int_t conventions and the
//! device atomicAdd overloads in CudaKernels.cu.
using idx32_t = int;
using idx64_t = long long;

/*!
  Parse an integer (e.g. from the --mi command-line flag) into an IndexMode.
  Unknown values fall back to the default legacy mode (I32_I32).
 */
inline IndexMode indexModeFromInt(int value)
{
    switch (value)
    {
    case 1: return IndexMode::I64_I32;
    case 2: return IndexMode::I64_I64;
    default: return IndexMode::I32_I32;
    }
}

//! Human-readable label for banners/logs.
inline const char* toString(IndexMode mode)
{
    switch (mode)
    {
    case IndexMode::I64_I32: return "int64 offsets / int32 columns (mixed)";
    case IndexMode::I64_I64: return "int64 offsets / int64 columns";
    case IndexMode::I32_I32:
    default: return "int32 offsets / int32 columns";
    }
}

//! True when the slice-offset arrays use a 64-bit element type.
inline bool offsetsAre64(IndexMode mode)
{
    return mode == IndexMode::I64_I32 || mode == IndexMode::I64_I64;
}

//! True when the column-index arrays use a 64-bit element type.
inline bool columnsAre64(IndexMode mode)
{
    return mode == IndexMode::I64_I64;
}

//! Byte size of one slice-offset element for the given mode.
inline size_t offsetIndexBytes(IndexMode mode)
{
    return offsetsAre64(mode) ? sizeof(idx64_t) : sizeof(idx32_t);
}

//! Byte size of one column-index element for the given mode.
inline size_t columnIndexBytes(IndexMode mode)
{
    return columnsAre64(mode) ? sizeof(idx64_t) : sizeof(idx32_t);
}

//! Advance a raw device pointer by `elems` elements of `elemBytes` each.
//! Centralizes the void* pointer arithmetic so call sites stay type-clean.
inline void* byteOffset(void* base, size_t elems, size_t elemBytes)
{
    return static_cast<char*>(base) + elems * elemBytes;
}

#ifdef USE_CUDA
//! cuSPARSE index type for the slice-offset arrays.
inline cusparseIndexType_t offsetCusparseIndexType(IndexMode mode)
{
    return offsetsAre64(mode) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
}

//! cuSPARSE index type for the column-index arrays.
inline cusparseIndexType_t columnCusparseIndexType(IndexMode mode)
{
    return columnsAre64(mode) ? CUSPARSE_INDEX_64I : CUSPARSE_INDEX_32I;
}
#endif

/*!
  Turn a runtime IndexMode into compile-time template parameters.

  Invokes `fn` with two value-initialized tag arguments whose types are the
  concrete offset and column element types for the mode. Use a generic lambda
  and recover the types via decltype, e.g.:

  @code
  dispatchIndexMode(mode, [&](auto offTag, auto colTag) {
      using OffsetT = decltype(offTag);
      using ColT    = decltype(colTag);
      launchKernel<OffsetT, ColT>(...);
  });
  @endcode

  This is the single point where the mode->type switch lives.
 */
template <class Fn>
inline void dispatchIndexMode(IndexMode mode, Fn&& fn)
{
    switch (mode)
    {
    case IndexMode::I32_I32: fn(idx32_t{}, idx32_t{}); break;
    case IndexMode::I64_I32: fn(idx64_t{}, idx32_t{}); break;
    case IndexMode::I64_I64: fn(idx64_t{}, idx64_t{}); break;
    }
}

//! Global runtime-selected index mode (defined in src/main.cpp), mirroring the
//! pattern used by P2P_Mode / Use_Compression.
extern IndexMode Index_Mode;

#endif // INDEXMODE_HPP
