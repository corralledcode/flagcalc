//
// Created by peterglenn on 5/12/25.
//

#ifndef CUDAENGINE_CUH
#define CUDAENGINE_CUH


void CUDAevalwithcriterion( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const CUDAfcptr start, const uint sz );

void CUDAevalwithcriterion( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const uint sz );

void CUDAevalwithcriterionfast( bool* crit, CUDAvalms* out, CUDAextendedcontext& Cec, const CUDAfcptr start,
    const uint dimm, const uint sz );



#endif //CUDAENGINE_CUH
