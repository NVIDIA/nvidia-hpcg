#include "MicroBenchmark.hpp"

#if defined(USE_CUDA) && defined(HPCG_MICRO_BENCHMARK)

#include <cstdlib>
#include <iostream>

#include "Cuda.hpp"
#include "CudaKernels.hpp"

bool RunMicroBenchmarks(const SparseMatrix& A, Vector& b, Vector& x, int rank)
{
    bool run_spmv = false;
    bool run_spsv = false;

    // Case selection: -1 = run all (default), 0 = General, 1 = Lower, 2 = Upper
    int mode = -1;

    const char* env_mode = std::getenv("MICRO_BENCHMARK_MODE");
    if (env_mode) {
        mode = std::atoi(env_mode);
        if (mode < 0 || mode > 2) {
            if (rank == 0) {
                std::cout << "Warning: Invalid MICRO_BENCHMARK_MODE value (" << mode
                          << "). Valid values are 0 (General), 1 (Lower), 2 (Upper). Using default (run all)." << std::endl;
            }
            mode = -1;
        }
    }

    const char* env_spmv = std::getenv("MICRO_BENCHMARK_SPMV");
    if (env_spmv && std::atoi(env_spmv) != 0)
        run_spmv = true;

    const char* env_spsv = std::getenv("MICRO_BENCHMARK_SPSV");
    if (env_spsv && std::atoi(env_spsv) != 0)
        run_spsv = true;

    if (!run_spmv && !run_spsv)
        return false;

    // Print summary header
    if (rank == 0) {
        std::cout << "\n========== Running Micro Benchmarks ==========" << std::endl;
        if (run_spmv) {
            std::cout << "SPMV Benchmark: ENABLED" << std::endl;
            if (mode == -1)
                std::cout << "  - Mode: All cases (General, Lower, Upper)" << std::endl;
            else if (mode == 0)
                std::cout << "  - Mode: General case only" << std::endl;
            else if (mode == 1)
                std::cout << "  - Mode: Lower triangular only" << std::endl;
            else if (mode == 2)
                std::cout << "  - Mode: Upper triangular only" << std::endl;
        }
        if (run_spsv) {
            std::cout << "SPSV Benchmark: ENABLED (no General case)" << std::endl;
            if (mode == -1)
                std::cout << "  - Mode: Both cases (Lower, Upper)" << std::endl;
            else if (mode == 0)
                std::cout << "  - Mode: General not applicable for SpSV, running both" << std::endl;
            else if (mode == 1)
                std::cout << "  - Mode: Lower triangular only" << std::endl;
            else if (mode == 2)
                std::cout << "  - Mode: Upper triangular only" << std::endl;
        }
        std::cout << "Set MICRO_BENCHMARK_SPMV=1 or MICRO_BENCHMARK_SPSV=1 to enable" << std::endl;
        std::cout << "Set MICRO_BENCHMARK_MODE to 0 (General), 1 (Lower), or 2 (Upper)" << std::endl;
        std::cout << "===============================================\n" << std::endl;
    }

    const int warmupRuns = 3;
    const int numberOfRuns = 10;

    // ==================== SpMV Benchmark ====================
    if (run_spmv) {
        double flops_per_spmv = 2.0 * A.localNumberOfNonzeros;
        double one = 1.0, zero = 0.0;

        std::cout << "========== SPMV Configuration ==========" << std::endl;
        std::cout << "MV_UNROLL: " << g_config.MV_UNROLL << std::endl;
        std::cout << "MV_BLOCK_SIZE: " << g_config.MV_BLOCK_SIZE << std::endl;
        std::cout << "VECTOR_WIDTH: " << g_config.VECTOR_WIDTH << std::endl;
        std::cout << "USE_TMA_MV: " << (g_config.USE_TMA_MV ? "ENABLED" : "DISABLED") << std::endl;
        std::cout << "Local Non-zeros: " << A.localNumberOfNonzeros << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << std::endl;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // General (mode == -1 or mode == 0)
        if (mode == -1 || mode == 0) {
            for (int i = 0; i < warmupRuns; ++i)
                mv_sell(General, A, one, zero, b.values_d, x.values_d);
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            for (int i = 0; i < numberOfRuns; ++i)
                mv_sell(General, A, one, zero, b.values_d, x.values_d);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            double t = ms / numberOfRuns;
            double gflops = (flops_per_spmv / (t * 1e-3)) / 1e9;
            std::cout << "General SPMV time (ms): " << t << " | GFLOPS: " << gflops << std::endl;
        }

        // Lower (mode == -1 or mode == 1)
        if (mode == -1 || mode == 1) {
            for (int i = 0; i < warmupRuns; ++i)
                mv_sell(Forward, A, one, zero, b.values_d, x.values_d);
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            for (int i = 0; i < numberOfRuns; ++i)
                mv_sell(Forward, A, one, zero, b.values_d, x.values_d);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            double t = ms / numberOfRuns;
            double gflops = (flops_per_spmv / (t * 1e-3)) / 1e9;
            std::cout << "Lower SPMV time (ms): " << t << " | GFLOPS: " << gflops << std::endl;
        }

        // Upper (mode == -1 or mode == 2)
        if (mode == -1 || mode == 2) {
            for (int i = 0; i < warmupRuns; ++i)
                mv_sell(Backward, A, one, zero, b.values_d, x.values_d);
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            for (int i = 0; i < numberOfRuns; ++i)
                mv_sell(Backward, A, one, zero, b.values_d, x.values_d);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            double t = ms / numberOfRuns;
            double gflops = (flops_per_spmv / (t * 1e-3)) / 1e9;
            std::cout << "Upper SPMV time (ms): " << t << " | GFLOPS: " << gflops << std::endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaGetLastError();
    }

    // ==================== SpSV Benchmark ====================
    if (run_spsv) {
        double flops_per_spsv = 2.0 * A.localNumberOfNonzeros;

        std::cout << "========== SPSV Configuration ==========" << std::endl;
        std::cout << "SV_UNROLL: " << g_config.SV_UNROLL << std::endl;
        std::cout << "SV_BLOCK_SIZE: " << g_config.SV_BLOCK_SIZE << std::endl;
        std::cout << "VECTOR_WIDTH: " << g_config.VECTOR_WIDTH << std::endl;
        std::cout << "USE_TMA_SV: " << (g_config.USE_TMA_SV ? "ENABLED" : "DISABLED") << std::endl;
        std::cout << "Local Non-zeros: " << A.localNumberOfNonzeros << std::endl;
        std::cout << "==========================================" << std::endl;
        std::cout << std::endl;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Lower (mode == -1 or mode == 0 or mode == 1) — same pattern as SpMV / Upper SpSV: timed sv_sell only
        if (mode == -1 || mode == 0 || mode == 1) {
            for (int i = 0; i < warmupRuns; ++i)
                sv_sell(Forward, A, b.values_d, x.values_d);
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            for (int i = 0; i < numberOfRuns; ++i)
                sv_sell(Forward, A, b.values_d, x.values_d);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            double t = ms / numberOfRuns;
            double gflops = (flops_per_spsv / (t * 1e-3)) / 1e9;
            std::cout << "Lower SpSV time (ms): " << t << " | GFLOPS: " << gflops << std::endl;
        }

        // Upper (mode == -1 or mode == 0 or mode == 2)
        if (mode == -1 || mode == 0 || mode == 2) {
            for (int i = 0; i < warmupRuns; ++i)
                sv_sell(Backward, A, b.values_d, x.values_d);
            cudaStreamSynchronize(stream);

            cudaEventRecord(start, stream);
            for (int i = 0; i < numberOfRuns; ++i)
                sv_sell(Backward, A, b.values_d, x.values_d);
            cudaEventRecord(stop, stream);
            cudaStreamSynchronize(stream);

            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            double t = ms / numberOfRuns;
            double gflops = (flops_per_spsv / (t * 1e-3)) / 1e9;
            std::cout << "Upper SpSV time (ms): " << t << " | GFLOPS: " << gflops << std::endl;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaGetLastError();
    }

    return true;
}

#endif // USE_CUDA && HPCG_MICRO_BENCHMARK
