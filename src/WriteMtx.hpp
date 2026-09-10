#ifndef WRITEMTX_HPP
#define WRITEMTX_HPP

#include "SparseMatrix.hpp"

#ifdef HPCG_WRITE_MTX
void WriteSellToMtx(const SparseMatrix& A, const char* fileA, const char* fileL);
#endif

#endif // WRITEMTX_HPP
