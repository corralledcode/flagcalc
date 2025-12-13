//
// Created by peterglenn on 5/12/25.
//

#ifndef CUDAENGINE_CUH
#define CUDAENGINE_CUH

#ifdef FLAGCALC_CUDA

#include <cuda_runtime.h>

#include "cuda.cuh"

void CUDAevalwithcriterion( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const CUDAfcptr start, const unsigned int sz );

void CUDAevalwithcriterion( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const unsigned int sz );

void CUDAevalwithcriterionfast( bool* crit, CUDAvalms* out, CUDAextendedcontext& Cec, const CUDAfcptr start,
    const unsigned int dimm, const unsigned int sz );

void CUDAcountpathsbetweenwrapper(int* out, int walklength, const bool* adjmatrix, const int dim );


void CUDAcomputeneighborslistwrapper( graphtype* g, neighborstype* ns );

void CUDAcomputeneighborslistenmassewrapper( std::vector<graphtype*>& gs, std::vector<neighborstype*>& ns );

void CUDAverticesconnectedmatrixwrapper( const graphtype* g, const neighborstype* ns, bool* out );

#endif // FLAGCALC_CUDA

#endif //CUDAENGINE_CUH
