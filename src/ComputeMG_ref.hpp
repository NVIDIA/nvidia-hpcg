
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

#ifndef COMPUTEMG_REF_HPP
#define COMPUTEMG_REF_HPP
#include "SparseMatrix.hpp"
#include "Vector.hpp"

// The use of CPU and GPU Sparse Matrix is intended to resolve
// the linked list structures for MG coarse levels (A->Ac)
// There is no change of th erefernce code

int ComputeMG_ref(const SparseMatrix& A, const Vector& r, Vector& x);

#endif // COMPUTEMG_REF_HPP
