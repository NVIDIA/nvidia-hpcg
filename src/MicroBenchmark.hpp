#ifndef MICROBENCHMARK_HPP
#define MICROBENCHMARK_HPP

#include "SparseMatrix.hpp"
#include "Vector.hpp"

#if defined(USE_CUDA) && defined(HPCG_MICRO_BENCHMARK)

/*!
  Runs SpMV and/or SpSV micro benchmarks on the given matrix.
  Controlled by environment variables:
    MICRO_BENCHMARK_SPMV=1   Enable SpMV benchmark
    MICRO_BENCHMARK_SPSV=1   Enable SpSV benchmark
    MICRO_BENCHMARK_MODE     0=General, 1=Lower, 2=Upper, -1=all (default)

  @return true if any benchmark ran (caller should exit early), false otherwise.
*/
bool RunMicroBenchmarks(const SparseMatrix& A, Vector& b, Vector& x, int rank);

#endif // USE_CUDA && HPCG_MICRO_BENCHMARK

#endif // MICROBENCHMARK_HPP
