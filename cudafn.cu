//
// Created by peterglenn on 5/10/25.
//

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#include "cuda.cuh"

#include "ameas.h"

#include <cstring>;



__device__ inline int CUDAtotient( int n)
{
    int phi = n > 1 ? n : 1;
    int max = (int)(sqrt( (double)n )) + 1;
    for (int p = 2; p <= max; p++)
    {
        if (n % p == 0)
        {
            phi -= phi / p;
            while (n % p == 0)
                n /= p;
        }
    }
    if (n > 1)
        phi -= phi / n;
    return phi;
}

// Function to compute Stirling numbers of
// the second kind S(n, k) with memoization
__device__ inline int CUDAstirlinginternal(int n, int k) {

    // Base cases
    if (n == 0 && k == 0) return 1;
    if (k == 0 || n == 0) return 0;
    if (n == k) return 1;
    if (k == 1) return 1;


    // Recursive formula
    return k * CUDAstirlinginternal(n - 1, k) + CUDAstirlinginternal(n - 1, k - 1);
}

// Function to calculate the total number of
// ways to partition a set of `n` elements
__device__ inline int CUDAbellNumberinternal(int n) {

    int result = 0;

    // Sum up Stirling numbers S(n, k) for all
    // k from 1 to n
    for (int k = 1; k <= n; ++k) {
        result += CUDAstirlinginternal(n, k);
    }
    return result;
}


__device__ inline long int CUDAnchoosekinternal( const long int n, const long int k )
{if (k == 0) return 1;
    return (n* CUDAnchoosekinternal(n-1,k-1))/k; }
__device__ inline long int CUDAmod( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{long int a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtdiscrete).v.iv;
    long int b = CUDAlookupnamedvariable(Cec,Cnvptr,1,mtdiscrete).v.iv;
    return a % b;}
__device__ inline double CUDAlog( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return log(a);}
__device__ inline double CUDAsin( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return sin(a);}
__device__ inline double CUDAcos( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return cos(a);}
__device__ inline double CUDAtan( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return tan(a);}
__device__ inline long int CUDAfloor( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return floorf(a);}
__device__ inline long int CUDAceil( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return ceilf(a);}
__device__ inline double CUDAgamma( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return tgamma(a);}
__device__ inline long int CUDAnchoosek( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{long int a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtdiscrete).v.iv;
    long int b = CUDAlookupnamedvariable(Cec,Cnvptr,1,mtdiscrete).v.iv;
    return CUDAnchoosekinternal(a,b);}
__device__ inline double CUDAexp( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv;return exp(a);}
__device__ inline bool CUDAisinf( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return isinf(a);}
__device__ inline double CUDAabs( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return abs(a);}
__device__ inline long int CUDAstirling( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{long int a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtdiscrete).v.iv;
    long int b = CUDAlookupnamedvariable(Cec,Cnvptr,1,mtdiscrete).v.iv;
    return CUDAstirlinginternal(a,b);}
__device__ inline double CUDAbell( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return CUDAbellNumberinternal(a);}
__device__ inline double CUDAsqrt( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{double a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return sqrt(a);}
__device__ inline long int CUDAphi( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr)
{long int a = CUDAlookupnamedvariable(Cec,Cnvptr,0,mtcontinuous).v.dv; return CUDAtotient(a);}

inline CUDAliteral CUDAlogfn = { false, 0, mtcontinuous,-1,{ .fncontinuous = &CUDAlog} };
inline CUDAliteral CUDAsinfn = {  false, 1, mtcontinuous,-1,{ .fncontinuous = &CUDAsin}};
inline CUDAliteral CUDAcosfn = {  false, 2, mtcontinuous, -1, { .fncontinuous = &CUDAcos}};
inline CUDAliteral CUDAtanfn = {   false, 3, mtcontinuous, -1,{ .fncontinuous = &CUDAtan}};
inline CUDAliteral CUDAfloorfn = {   false, 4, mtdiscrete, -1,{ .fndiscrete = &CUDAfloor}};
inline CUDAliteral CUDAceilfn = {   false, 5, mtdiscrete, -1,{ .fndiscrete = &CUDAceil}};
inline CUDAliteral CUDAgammafn = {   false, 6, mtcontinuous, -1,{ .fncontinuous = &CUDAgamma}};
inline CUDAliteral CUDAnchoosekfn = {   false, 7, mtdiscrete, -1,{ .fndiscrete = &CUDAnchoosek}};
inline CUDAliteral CUDAexpfn = {  false, 8, mtcontinuous, -1,{ .fncontinuous = &CUDAexp}};
inline CUDAliteral CUDAisinffn = { false, 9, mtbool, -1,{ .fnbool = &CUDAisinf}};
inline CUDAliteral CUDAabsfn = {  false, 10, mtcontinuous, -1,{ .fncontinuous = &CUDAabs}};
inline CUDAliteral CUDAmodfn = {  false, 11, mtdiscrete, -1,{ .fndiscrete = &CUDAmod}};
inline CUDAliteral CUDAstirlingfn = {   false, 12, mtdiscrete, -1,{ .fndiscrete = &CUDAstirling}};
inline CUDAliteral CUDAbellfn = { false, 13, mtcontinuous, -1,{ .fncontinuous = &CUDAbell}};
inline CUDAliteral CUDAsqrtfn = {  false, 14, mtcontinuous, -1,{ .fncontinuous = &CUDAsqrt}};
inline CUDAliteral CUDAphifn = {  false, 15, mtdiscrete, -1,{ .fndiscrete = &CUDAphi}};

__device__ inline long int CUDAsizetally( const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr )
{long int cnt = 0; CUDAvalms v = CUDAlookupnamedvariable(Cec, Cnvptr, 0, mtset);
    for (int i = 0; i < v.v.seti.sz; ++i)
        cnt += (*(bool**)&Cec.CUDAvalsarray)[v.v.seti.ptr + i * sizeof(bool)] ? 1 : 0;
    return cnt;}

inline CUDAliteral CUDAsizetallyfn = {false,14, mtdiscrete, -1, { .fndiscrete = &CUDAsizetally}};

inline std::map<std::string,std::pair<CUDAliteral,std::vector<CUDAvalms>>> global_CUDAfnptrs
    {{"log", {CUDAlogfn,{{.t = mtcontinuous}}}},
     {"sin", {CUDAsinfn,{{.t = mtcontinuous}}}},
     {"cos", {CUDAcosfn,{{.t = mtcontinuous}}}},
     {"tan", {CUDAtanfn,{{.t = mtcontinuous}}}},
     {"floor", {CUDAfloorfn,{{.t = mtcontinuous}}}},
     {"ceil", {CUDAceilfn,{{.t = mtcontinuous}}}},
     {"gamma", {CUDAgammafn,{{.t = mtcontinuous}}}},
     {"nchoosek", {CUDAnchoosekfn,{{.t = mtdiscrete}, {.t = mtdiscrete}}}},
     {"exp", {CUDAexpfn,{{.t = mtcontinuous}}}},
     {"isinf", {CUDAisinffn,{{.t = mtcontinuous}}}},
     {"abs", {CUDAabsfn,{{.t = mtcontinuous}}}},
     {"mod",{CUDAmodfn,{{.t = mtdiscrete},{.t = mtdiscrete}}}},
     {"stirling", {CUDAstirlingfn,{{.t = mtdiscrete},{.t = mtdiscrete}}}},
     {"bell",{CUDAbellfn,{{.t = mtdiscrete}}}},
     {"sqrt",{CUDAsqrtfn,{{.t = mtcontinuous}}}},
     {"phi",{CUDAphifn,{{.t = mtdiscrete}}}},
     {"st",{CUDAsizetallyfn,{{.t = mtset}}}}};

__device__ inline void populateCUDAfnarraysingle( CUDAliteral* lits, const int litssize ) // absolutely bizarre, the second argument must be int not uint
{
    for (int i = 0; i < litssize; ++i)
    {
        auto lit = lits[i];
        switch (lit.l)
        {
        case 0: lit.function.fncontinuous = CUDAlog; break;
        case 1: lit.function.fncontinuous = CUDAsin; break;
        case 2: lit.function.fncontinuous = CUDAcos; break;
        case 3: lit.function.fncontinuous = CUDAtan; break;
        case 4: lit.function.fndiscrete = CUDAfloor; break;
        case 5: lit.function.fndiscrete = CUDAceil; break;
        case 6: lit.function.fncontinuous = CUDAgamma; break;
        case 7: lit.function.fndiscrete = CUDAnchoosek; break;
        case 8: lit.function.fncontinuous = CUDAexp; break;
        case 9: lit.function.fnbool = CUDAisinf; break;
        case 10: lit.function.fncontinuous = CUDAabs; break;
        case 11: lit.function.fndiscrete = CUDAmod; break;
        case 12: lit.function.fndiscrete = CUDAstirling; break;
        case 13: lit.function.fncontinuous = CUDAbell; break;
        case 14: lit.function.fncontinuous = CUDAsqrt; break;
        case 15: lit.function.fndiscrete = CUDAphi; break;
        case 16: lit.function.fndiscrete = CUDAsizetally; break;
        default: lit.function.fndiscrete = CUDAfloor; break;
        }
        lits[i] = lit;
    }
}

__global__ inline void populateCUDAfnarray( CUDAextendedcontext* Cecs, const uint sz )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= sz)
        return;
    if (sz == 1)
        populateCUDAfnarraysingle(Cecs->CUDAliteralarray,Cecs->CUDAliteralarraysize);
    else
        populateCUDAfnarraysingle(Cecs[index].CUDAliteralarray,Cecs[index].CUDAliteralarraysize);
}
