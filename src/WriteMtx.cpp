#include "WriteMtx.hpp"

#ifdef HPCG_WRITE_MTX

#include <algorithm>
#include <cstdio>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

static void WriteOneSellToMtx(const char* fname,
                               local_int_t nrow, local_int_t slice_size,
                               local_int_t num_slices,
                               const std::vector<slice_ptr_t>& off,
                               const std::vector<local_int_t>& cols,
                               const std::vector<double>& vals)
{
    slice_ptr_t nnz = 0;
    for (local_int_t s = 0; s < num_slices; s++) {
        local_int_t mrl = (local_int_t) ((off[s + 1] - off[s]) / slice_size);
        local_int_t rows_in_slice = std::min(slice_size, nrow - s * slice_size);
        for (local_int_t r = 0; r < rows_in_slice; r++)
            for (local_int_t k = 0; k < mrl; k++)
                if (cols[off[s] + r + k * slice_size] >= 0) nnz++;
    }

    FILE* fp = fopen(fname, "w");
    if (!fp) {
        printf("WriteMtx: failed to open %s\n", fname);
        return;
    }

    fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(fp, "%lld %lld %lld\n",
            (long long)nrow, (long long)nrow, (long long)nnz);

    for (local_int_t s = 0; s < num_slices; s++) {
        slice_ptr_t base = off[s];
        local_int_t mrl = (local_int_t) ((off[s + 1] - base) / slice_size);
        local_int_t rows_in_slice = std::min(slice_size, nrow - s * slice_size);
        for (local_int_t r = 0; r < rows_in_slice; r++) {
            for (local_int_t k = 0; k < mrl; k++) {
                slice_ptr_t idx = base + r + (slice_ptr_t) k * slice_size;
                if (cols[idx] >= 0)
                    fprintf(fp, "%lld %lld %.15g\n",
                            (long long)(s * slice_size + r + 1),
                            (long long)(cols[idx] + 1),
                            vals[idx]);
            }
        }
    }

    fclose(fp);
    printf("Wrote %s (%lld x %lld, %lld nnz)\n",
           fname, (long long)nrow, (long long)nrow, (long long)nnz);
}

void WriteSellToMtx(const SparseMatrix& A, const char* fileA, const char* fileL)
{
    local_int_t nrow = A.localNumberOfRows;
    local_int_t ss   = A.slice_size;
    local_int_t num_slices = (nrow + ss - 1) / ss;

#ifdef USE_CUDA
    std::vector<slice_ptr_t> h_aOff(num_slices + 1), h_lOff(num_slices + 1);
    cudaMemcpy(h_aOff.data(), A.sellASliceMrl,
               (num_slices + 1) * sizeof(slice_ptr_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lOff.data(), A.sellLSliceMrl,
               (num_slices + 1) * sizeof(slice_ptr_t), cudaMemcpyDeviceToHost);

    slice_ptr_t a_total = h_aOff[num_slices];
    slice_ptr_t l_total = h_lOff[num_slices];

    std::vector<local_int_t> h_aCols(a_total), h_lCols(l_total);
    std::vector<double>      h_aVals(a_total), h_lVals(l_total);

    cudaMemcpy(h_aCols.data(), A.sellAPermColumns,
               a_total * sizeof(local_int_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_aVals.data(), A.sellAPermValues,
               a_total * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lCols.data(), A.sellLPermColumns,
               l_total * sizeof(local_int_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lVals.data(), A.sellLPermValues,
               l_total * sizeof(double), cudaMemcpyDeviceToHost);

    WriteOneSellToMtx(fileA, nrow, ss, num_slices, h_aOff, h_aCols, h_aVals);
    WriteOneSellToMtx(fileL, nrow, ss, num_slices, h_lOff, h_lCols, h_lVals);
#else
    std::vector<slice_ptr_t> h_aOff(A.sellASliceMrl, A.sellASliceMrl + num_slices + 1);
    std::vector<slice_ptr_t> h_lOff(A.sellLSliceMrl, A.sellLSliceMrl + num_slices + 1);

    slice_ptr_t a_total = h_aOff[num_slices];
    slice_ptr_t l_total = h_lOff[num_slices];

    std::vector<local_int_t> h_aCols(A.sellAPermColumns, A.sellAPermColumns + a_total);
    std::vector<double>      h_aVals(A.sellAPermValues,  A.sellAPermValues  + a_total);
    std::vector<local_int_t> h_lCols(A.sellLPermColumns, A.sellLPermColumns + l_total);
    std::vector<double>      h_lVals(A.sellLPermValues,  A.sellLPermValues  + l_total);

    WriteOneSellToMtx(fileA, nrow, ss, num_slices, h_aOff, h_aCols, h_aVals);
    WriteOneSellToMtx(fileL, nrow, ss, num_slices, h_lOff, h_lCols, h_lVals);
#endif
}

#endif // HPCG_WRITE_MTX
