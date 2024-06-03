
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

#ifndef CG_REF_HPP
#define CG_REF_HPP

#include "CGData.hpp"
#include "SparseMatrix.hpp"
#include "Vector.hpp"

// The use of CPU and GPU Sparse Matrix is intended to resolve
// the linked list structures for MG coarse levels
// There is no change of th erefernce code

int CG_ref(const SparseMatrix& A, CGData& data, const Vector& b, Vector& x, const int max_iter, const double tolerance,
    int& niters, double& normr, double& normr0, double* times, bool doPreconditioning, int flag);

// this function will compute the Conjugate Gradient iterations.
// geom - Domain and processor topology information
// A - Matrix
// b - constant
// x - used for return value
// max_iter - how many times we iterate
// tolerance - Stopping tolerance for preconditioned iterations.
// niters - number of iterations performed
// normr - computed residual norm
// normr0 - Original residual
// times - array of timing information
// doPreconditioning - bool to specify whether or not symmetric GS will be applied.

#endif // CG_REF_HPP
