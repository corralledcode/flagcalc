//
// Created by peterglenn on 5/4/25.
//

// #include "config.h"

#ifdef FLAGCALC_CUDA

#include <iostream>
#include <vector>

#include <cuda_runtime.h>

// #include "cuda.cuh"
// #include "cudafn.cu"
#include "cudagraph.cu"

#include "ameas.h"

#include <cstring>;


__device__ inline bool CUDAto_mtbool( const CUDAvalms v )
{
    switch (v.t)
    {
    case mtbool:
            return v.v.bv;
    case mtdiscrete:
            return v.v.iv ? true : false;
    case mtcontinuous:
            return v.v.dv ? true : false;
    default:
        // std::cout << "Not yet implemented to_mtbool type\n";
        return false;
    }
}

__device__ inline LONGINT CUDAto_mtdiscrete( const CUDAvalms v )
{
    switch (v.t)
    {
    case mtbool:
        return v.v.bv ? 1 : 0;
    case mtdiscrete:
        return v.v.iv;
    case mtcontinuous:
        return (int)v.v.dv;
    default:
        // std::cout << "Not yet implemented to_mtbool type\n";
        return 0;
    }
}

__device__ inline double CUDAto_mtcontinuous( const CUDAvalms v )
{
    switch (v.t)
    {
    case mtbool:
        return (double)(v.v.bv ? 1 : 0);
    case mtdiscrete:
        return (double)v.v.iv;
    case mtcontinuous:
        return v.v.dv;
    default:
        // std::cout << "Not yet implemented to_mtbool type\n";
        return 0;
    }
}

__device__ inline CUDAseti newset( const CUDAextendedcontext& Cec, const CUDAvalsptr& Cvptr, const measuretype& mt, const unsigned int& sz, void* data )
{
    memcpy(((void**)&Cec.CUDAvalsarray)[Cvptr],data,sz);
    CUDAseti res;
    res.ptr = Cvptr;
    res.sz = sz;
    res.st = mt;
    delete data;
    return res;
}

__device__ inline CUDAseti CUDAto_mtset( const CUDAextendedcontext& Cec, const CUDAvalsptr& Cvptr, const measuretype& mt, const unsigned int& sz, const CUDAvalms& v )
{
    switch (v.t)
    {
    case mttuple:
    case mtset:
        return v.v.seti;
    case mtdiscrete:
        {
            bool* data = new bool[v.v.iv+1];
            memset(data, false, (v.v.iv+1)*sizeof(bool));
            data[v.v.iv] = true;
            return newset(Cec,Cvptr,v.t,v.v.iv+1,data);
        }
    case mtcontinuous:
        {
            double* data = new double[1];
            data[0] = v.v.dv;
            return newset(Cec,Cvptr,v.t,1,data);
        }
    case mtbool:
        {
            bool* data = new bool[1];
            data[0] = v.v.bv;
            return newset(Cec,Cvptr,v.t,1,data);
        }
    default:
        return newset(Cec,Cvptr,v.t,0,nullptr);
    }
}

__device__ inline CUDAvalms CUDAvalmsto_specified( const CUDAextendedcontext& Cec, const CUDAvalms v, const measuretype mt )
{
    CUDAvalms res;
    res.t = mt;
    switch (mt)
    {
        case mtdiscrete: {res.v.iv = CUDAto_mtdiscrete(v); break;}
        case mtcontinuous: {res.v.dv = CUDAto_mtcontinuous(v); break;}
        case mtbool: {res.v.bv = CUDAto_mtbool(v); break;}
        case mtset:
        case mttuple: {res.v.seti = CUDAto_mtset(Cec,v.v.seti.ptr,v.v.seti.st,v.v.seti.sz,v); break;}
    }
    return res;
}

__device__ inline LONGINT CUDAsizetally( const CUDAextendedcontext& Cec, const CUDAvalms* args )
{LONGINT cnt = 0; CUDAvalms v = args[0];
    for (int i = 0; i < v.v.seti.sz; ++i)
        cnt += (*(bool**)&Cec.CUDAvalsarray)[v.v.seti.ptr + i * sizeof(bool)] ? 1 : 0;
    return cnt;}

__device__ inline bool CUDAac( const CUDAextendedcontext& Cec, const CUDAvalms* args )
{ // vertices adjacent
    LONGINT a = CUDAto_mtdiscrete(args[0]);
    LONGINT b = CUDAto_mtdiscrete(args[1]);
    return Cec.g.adjacencymatrix[a*Cec.g.dim + b];}

__device__ inline bool CUDAconnvc( const CUDAextendedcontext& Cec, const CUDAvalms* args )
{ // vertices are connected
    LONGINT a = CUDAto_mtdiscrete(args[0]);
    LONGINT b = CUDAto_mtdiscrete(args[1]);
    return CUDApathsbetweenmin( &Cec.g, &Cec.ns, a, b, 1 ) > 0;
}


__device__ inline int CUDAtotient( LONGINT n)
{
    LONGINT phi = n > 1 ? n : 1;
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


__device__ inline LONGINT CUDAnchoosekinternal( const LONGINT n, const LONGINT k )
{if (k == 0) return 1;
    return (n* CUDAnchoosekinternal(n-1,k-1))/k; }
__device__ inline LONGINT CUDAmod( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{LONGINT a = CUDAto_mtdiscrete(args[0]);
    LONGINT b = CUDAto_mtdiscrete(args[1]);
    return a % b;}
__device__ inline double CUDAlog( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return log(a);}
__device__ inline double CUDAsin( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return sin(a);}
__device__ inline double CUDAcos( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return cos(a);}
__device__ inline double CUDAtan( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return tan(a);}
__device__ inline LONGINT CUDAfloor( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return floorf(a);}
__device__ inline LONGINT CUDAceil( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return ceilf(a);}
__device__ inline double CUDAgamma( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return tgamma(a);}
__device__ inline LONGINT CUDAnchoosek( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{LONGINT a = CUDAto_mtdiscrete(args[0]);
    LONGINT b = CUDAto_mtdiscrete(args[1]);
    return CUDAnchoosekinternal(a,b);}
__device__ inline double CUDAexp( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return exp(a);}
__device__ inline bool CUDAisinf( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return isinf(a);}
__device__ inline double CUDAabs( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return abs(a);}
__device__ inline LONGINT CUDAstirling( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{LONGINT a = CUDAto_mtdiscrete(args[0]);
    LONGINT b = CUDAto_mtdiscrete(args[1]);
    return CUDAstirlinginternal(a,b);}
__device__ inline double CUDAbell( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return CUDAbellNumberinternal(a);}
__device__ inline double CUDAsqrt( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return sqrt(a);}
__device__ inline LONGINT CUDAphi( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{LONGINT a = CUDAto_mtcontinuous(args[0]); return CUDAtotient(a);}

__device__ inline double CUDAasin( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return asin(a);}
__device__ inline double CUDAacos( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return acos(a);}
__device__ inline double CUDAatan( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return atan(a);}
__device__ inline double CUDAatan2( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); double b = CUDAto_mtcontinuous(args[1]); return atan2(a,b);}
__device__ inline double CUDApi( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{return std::numbers::pi;}
__device__ inline double CUDAlog2( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return log2(a);}
__device__ inline double CUDAlog10( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); return log10(a);}
__device__ inline double CUDAlogb( const CUDAextendedcontext& Cec, const CUDAvalms* args)
{double a = CUDAto_mtcontinuous(args[0]); double b = CUDAto_mtcontinuous(args[1]); return log(a)/log(b);}


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

inline CUDAliteral CUDAasinfn = { false, 16, mtcontinuous, -1,{ .fncontinuous = &CUDAasin}};
inline CUDAliteral CUDAacosfn = { false, 17, mtcontinuous, -1,{ .fncontinuous = &CUDAacos}};
inline CUDAliteral CUDAatanfn = { false, 18, mtcontinuous, -1,{ .fncontinuous = &CUDAatan}};
inline CUDAliteral CUDAatan2fn = { false, 19, mtcontinuous, -1,{ .fncontinuous = &CUDAatan2}};
inline CUDAliteral CUDApifn = { false, 20, mtcontinuous, -1,{ .fncontinuous = &CUDApi}};
inline CUDAliteral CUDAlog2fn = { false, 21, mtcontinuous, -1,{ .fncontinuous = &CUDAlog2}};
inline CUDAliteral CUDAlog10fn = { false, 22, mtcontinuous, -1,{ .fncontinuous = &CUDAlog10}};
inline CUDAliteral CUDAlogbfn = { false, 23, mtcontinuous, -1,{ .fncontinuous = &CUDAlogb}};




inline CUDAliteral CUDAsizetallyfn = {false,16, mtdiscrete, -1, { .fndiscrete = &CUDAsizetally}};
inline CUDAliteral CUDAacfn = {false,17, mtbool, -1, { .fnbool = &CUDAac}};
inline CUDAliteral CUDAconnvcfn = {false,18, mtbool, -1, { .fnbool = &CUDAconnvc}};

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

    {"asin",{CUDAasinfn,{{.t = mtcontinuous}}}},
    {"acos",{CUDAacosfn,{{.t = mtcontinuous}}}},
    {"atan",{CUDAatanfn,{{.t = mtcontinuous}}}},
    {"atan2",{CUDAatan2fn,{{.t = mtcontinuous}, {.t = mtcontinuous}}}},
    {"pi",{CUDApifn,{{}}}},
    {"log2",{CUDAlog2fn,{{.t = mtcontinuous}}}},
    {"log10",{CUDAlog10fn,{{.t = mtcontinuous}}}},
    {"logb",{CUDAlogbfn,{{.t = mtcontinuous},{.t = mtcontinuous}}}},

     {"st",{CUDAsizetallyfn,{{.t = mtset}}}},
     {"ac",{CUDAacfn,{{.t = mtdiscrete}, {.t = mtdiscrete}}}},
     {"connvc",{CUDAconnvcfn,{{.t = mtdiscrete}, {.t = mtdiscrete}}}}};

__device__ inline void populateCUDAfnarraydevice( CUDAliteral* lits, const unsigned int litssize ) // absolutely bizarre, the second argument must be int not unsigned int
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
        case 16: lit.function.fncontinuous = CUDAasin; break;
        case 17: lit.function.fncontinuous = CUDAacos; break;
        case 18: lit.function.fncontinuous = CUDAatan; break;
        case 19: lit.function.fncontinuous = CUDAatan2; break;
        case 20: lit.function.fncontinuous = CUDApi; break;
        case 21: lit.function.fncontinuous = CUDAlog2; break;
        case 22: lit.function.fncontinuous = CUDAlog10; break;
        case 23: lit.function.fncontinuous = CUDAlogb; break;
        case 24: lit.function.fndiscrete = CUDAsizetally; break;
        case 25: lit.function.fnbool = CUDAac; break;
        case 26: lit.function.fnbool = CUDAconnvc; break;
        default: lit.function.fndiscrete = CUDAfloor; break;
        }
        lits[i] = lit;
    }
}

__global__ inline void populateCUDAfnarray( CUDAextendedcontext* Cecs, const unsigned int sz )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= sz || sz == 0)
        return;
    for (int i = 0; i < Cecs[0].CUDAliteralarraysize; ++i)
    {
        populateCUDAfnarraydevice( Cecs[index].CUDAliteralarray, Cecs[index].CUDAliteralarraysize );
    }

    // populateCUDAfnarraydevice(Cecs[index].CUDAliteralarray,Cecs[index].CUDAliteralarraysize);
}

__global__ inline void populateCUDAfnarraysingle( CUDAextendedcontext* Cec )
{
    populateCUDAfnarraydevice(Cec->CUDAliteralarray,Cec->CUDAliteralarraysize);
}


__device__ CUDAvalms CUDAevalinternal( CUDAextendedcontext& Cec, const CUDAfcptr Cfcin );


__device__ inline CUDAvalms CUDAlookupnamedvariable(const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr, unsigned int count, const measuretype& mt)
{
    CUDAvalms res;
    CUDAnamedvariableptr ptr = Cnvptr;
    while (count > 0)
    {
        count--;
        ptr = Cec.namedvararray[ptr].next;
    }
    res.t = mt;
    res = CUDAvalmsto_specified(Cec,Cec.namedvararray[ptr].ufc.v,mt);
    return res;
}


inline bool CUDAfindfunction( std::vector<CUDAnamedvariable>& Cnvv,
    std::map<std::string,std::pair<CUDAliteral,std::vector<CUDAvalms>>> fnptrs,
    std::string nm, CUDAliteral& Cl)
{

    bool found = false;
    for (auto fnptr : fnptrs)
    {
        if (fnptr.first == nm)
        {
            // stringlookup.strings.push_back(nm);
            // stringlookup.indices.push_back()

            found = true;
            Cl = fnptr.second.first;
            Cl.inputvariabletypesptr = -1;
            for (auto v : fnptr.second.second)
            {
                CUDAnamedvariable nv;
                nv.ufc.v = v;
                nv.next = Cnvv.size()+1;
                if (Cl.inputvariabletypesptr < 0)
                    Cl.inputvariabletypesptr = Cnvv.size();
                Cnvv.push_back(nv);
            }
            if (Cnvv[Cnvv.size()-1].next >= Cnvv.size())
                Cnvv[Cnvv.size()-1].next = -1;
            break;
        }
    }
    return found;
}



void mttranslatetoCUDAset( setitr* seti, CUDAvalms& nv, CUDAdataspaces& Cdss )
{
    unsigned int typesize;
    if (setitrint* cast = dynamic_cast<setitrint*>(seti))
    {
        typesize = sizeof(bool);
        nv.v.seti.st = mtdiscrete;
        nv.v.seti.sz = cast->maxint+1;
        CUDAdataspace d;
        d.data = cast->elts;
        d.sz = sizeof(bool)*(nv.v.seti.sz);
        d.szfactor = typesize;
        // std::cout << "d.data[0] == " << (*(bool**)&d.data)[0] << ", " << (*(bool**)&d.data)[1] << std::endl;
        Cdss.Csv.push_back(d);
    } else if (setitrint2dsymmetric* cast2d = dynamic_cast<setitrint2dsymmetric*>(seti))
    {
        typesize = sizeof(bool);
        nv.v.seti.st = mtdiscrete; // for now, but ultimately mtset;
        nv.v.seti.sz = cast2d->itrint->maxint+1;
        CUDAdataspace d;
        d.data = cast2d->itrint->elts;
        d.sz = sizeof(bool)*(nv.v.seti.sz);
        d.szfactor = typesize;
        Cdss.Csv.push_back(d);
    }
    else if (setitrsubset* castss = dynamic_cast<setitrsubset*>(seti))
    {
        if (setitrint* castparent = dynamic_cast<setitrint*>(castss->superset->parent))
        {
            typesize = sizeof(bool);
            nv.v.seti.st = mtdiscrete;
            nv.v.seti.sz = castparent->maxint+1;
            bool* tempb = (bool*)malloc( sizeof(bool) * nv.v.seti.sz );
            for (int i = 0; i < nv.v.seti.sz; i++)
            {
                int j = castparent->totality[i].v.iv;
                tempb[j] = castss->itrint->elts[i];
            }
            CUDAdataspace d;
            d.data = tempb;
            d.sz = sizeof(bool)*(nv.v.seti.sz);
            d.szfactor = typesize;
        d.needsdelete = true;
            Cdss.Csv.push_back(d);
        } else if (setitrint2dsymmetric* cast2dparent = dynamic_cast<setitrint2dsymmetric*>(castss->superset->parent))
        {
            typesize = sizeof(bool);
            nv.v.seti.st = mtdiscrete;
            nv.v.seti.sz = cast2dparent->itrint->maxint+1;
            bool* tempb = (bool*)malloc( sizeof(bool) * nv.v.seti.sz );
            for (int i = 0; i < nv.v.seti.sz; i++)
            {
                int j = cast2dparent->totality[i].v.iv;
                tempb[j] = castss->itrint->elts[i];
            }
            CUDAdataspace d;
            d.data = tempb;
            d.sz = sizeof(bool)*(nv.v.seti.sz);
            d.szfactor = typesize;
            d.needsdelete = true;
            Cdss.Csv.push_back(d);
        } else
        {
            std::cout << "No CUDA support for this seti type\n";
            exit(1);
        }
    } else
    {
        std::cout << "No CUDA support for this seti type\n";
        exit(1);
    }
    nv.t = mtset;
    nv.v.seti.ptr = Cdss.totalsz;
    Cdss.totalsz += nv.v.seti.sz*typesize;
}


unsigned int CUDAprognosticatespaceneeded( CUDAdataspaces& Cdss, CUDAfcptr& fctop)
{
    if (fctop < 0)
    {
        return 0;
    }
    auto Cfc = &Cdss.Cfcv[fctop];

    switch (Cfc->fo)
    {
        case formulaoperator::fovariable:
            {
                // Cfc->v.v = Cdss.Ccv[Cdss.Cnvv[Cfc->namedvar].l].ufc.v;
                // Cdss.Cnvv[Cfc->namedvar].
                if (Cdss.Ccv[Cdss.Cnvv[Cfc->namedvar].l].ufc.v.t == mtset)
                // if (Cfc->v.v.t == mtset)
                {
                    // Cfc->v.v.v.seti = CUDAec.CUDAcontext[CUDAec.namedvararray[Cfc->namedvar].l].ufc.v.v.seti;
                    return Cdss.Ccv[Cdss.Cnvv[Cfc->namedvar].l].ufc.v.v.seti.sz;
                    // return Cfc->v.v.v.seti.sz;
                }
                return 0;
            }
        case formulaoperator::founion:
        case formulaoperator::fodupeunion:
        case formulaoperator::fointersection:
        case formulaoperator::fosetminus:
        case formulaoperator::fosetxor:
            {

                // CUDAvalms* Cv = &Cdss.Cnvv[Cfc->namedvar].ufc.v;

                // Cfc->v.v.v.seti.ptr = Cdss.totalsz;
                // Cfc->v.v.t = mtset;
                // Cfc->v.v.v.seti.st = mtdiscrete; // to do...
                CUDAdataspace d;
                d.szfactor = sizeof(bool);
                int d1 = CUDAprognosticatespaceneeded( Cdss, Cfc->fcright);
                int d2 = CUDAprognosticatespaceneeded( Cdss, Cfc->fcleft);
                d.sz = std::max(d1,d2); // treating all arrays as boolean membership lists
                CUDAprognosticatespaceneeded( Cdss, Cfc->criterion );
                d.data = nullptr;
                CUDAnamedvariable Cv;
                Cv.ufc.v.v.seti.ptr = Cdss.totalsz;
                Cv.ufc.v.t = mtset;
                Cv.ufc.v.v.seti.st = mtdiscrete;
                Cv.ufc.v.v.seti.sz = d.sz;
                Cdss.Cnvv.push_back(Cv);
                Cfc->namedvar = Cdss.Cnvv.size() - 1;
                Cdss.Csv.push_back(d);
                Cdss.totalsz += d.sz * d.szfactor;
                // Cfc->v.v.v.seti.sz = d.sz;
                return d.sz;
            }
        default:
            {
                CUDAprognosticatespaceneeded(Cdss, Cfc->fcleft );
                CUDAprognosticatespaceneeded(Cdss, Cfc->fcright );
                auto ptr = Cfc->namedvar;
                while (ptr >= 0)
                {
                    if (Cdss.Cnvv[ptr].t == CUDAnamedvariabletype::cnvfcvalue )
                        CUDAprognosticatespaceneeded(Cdss, Cdss.Cnvv[ptr].ufc.fcv );
                    ptr = Cdss.Cnvv[ptr].next;
                }
                return 0;
            }

    }
}


void flattencontextforCUDA( const namedparams& context, const unsigned int offset, CUDAdataspaces* Cdss, CUDAdataspaces& CdssNEW )
{
    if (Cdss)
    {
        CdssNEW.Ccv = Cdss->Ccv;
    }
    else CdssNEW.Ccv.clear();
        // Cdss->copyCUDAdataspaces(CdssNEW);

    const unsigned int startidx = offset;
    int idx = 0;
    CdssNEW.Ccv.resize(context.size());
    while (idx+startidx < context.size())
    {
        CUDAnamedvariable nv;
        stringlookup.strings.push_back(context[idx+startidx].first);
        stringlookup.indices.push_back(idx+startidx); // the index into context
        nv.t = CUDAnamedvariabletype::cnvvalue;

        nv.l = startidx+idx;
        nv.bound = true;

        nv.ufc.v.t = context[idx+startidx].second.t;
        switch (context[idx+startidx].second.t)
        {
        case mtbool:
            {
                nv.ufc.v.v.bv = context[idx+startidx].second.v.bv;
                break;
            }
        case mtdiscrete:
            {
                nv.ufc.v.v.iv = context[idx+startidx].second.v.iv;
                break;
            }
        case mtcontinuous:
            {
                nv.ufc.v.v.dv = context[idx+startidx].second.v.dv;
                break;
            }
        case mtset:
            {
                mttranslatetoCUDAset(context[idx+startidx].second.seti,nv.ufc.v,CdssNEW);
                break;
            }
        default:
            {
                std::cout << "No CUDA support for this type\n";
                exit(1);
            }
        }

        CdssNEW.Ccv[startidx+idx] = nv;
#ifdef CUDADEBUG
        std::cout << "Check t: " << CdssNEW.Ccv[idx+startidx].ufc.v.t << "\n";
        #endif

        ++idx;
    }

    // Assume fcarray already populated; now "prognosticate" what space is needed for future set/tuple computations


    // futureCUDAec.CUDAvalsarraysize = 0;

    // CUDAprognosticatespaceneeded( futureCUDAec, futureCUDAec.fctop, Cdss );

    // ... the above commented out three lines are now done in math.cu


#ifdef CUDADEBUG
    for (int i = 0; i < CdssNEW.Cfcv.size(); ++i)
    {
        CUDAfc tempfc = CdssNEW.Cfcv[i];
        std::cout << i << " (" << int(tempfc.fo) << "): " << tempfc.criterion << ", " << tempfc.fcleft << ", " << tempfc.fcright
            << "; " << (int)tempfc.v.v.t << ", seti.ptr == " << tempfc.v.v.v.seti.ptr << std::endl;
    }
#endif


    // ... populate CUDAvalsarray

    // Cdss.populateCdssNEW(CdssNEW);

#ifdef CUDADEBUG
    std::cout << "namedvararraysize == " << CdssNEW.Cnvv.size() << "\n";
/*
    for (int i = 0; i < CdssNEW.Cnvv.size(); ++i)
    {
        auto nv = CdssNEW.Cnvv[i];
        // if (nv.ufc.v.t == mtset)
        if (CdssNEW.Ccv[nv.l].ufc.v.t == mtset)
        {
            std::cout << "namedvararray: i == " << i << ", bound == " << nv.bound << ", l == " << nv.l << ", " << "[";
            // for (int j = 0; j < CdssNEW.Ccv[nv.l].ufc.v.v.seti.sz; ++j)
            // {
                // std::cout << (*(bool**)&CdssNEW.CUDAvalsarray)[nv.ufc.v.v.seti.ptr] << ",";
            // }
            std::cout << "]\n";
        } else
        std::cout << "namedvararray: i == " << i << ", bound == " << nv.bound << ", l == " << nv.l << ", nm == "
                << ", context[l].ufc.v.v.iv == " << CdssNEW.Ccv[nv.l].ufc.v.v.iv << "\n";
    }
    for (int i = 0; i < CdssNEW.Ccv.size(); ++i)
    {
        auto c = CdssNEW.Ccv[i];
        std::cout << "flattenContext: " << i << ", " << ": " << c.ufc.v.t << std::endl;
    }*/
#endif

    // Cdss.mergeCUDAdataspaces(CdssNEW);
}

CUDAfcptr addCfc( const formulaclass* fc, CUDAdataspaces& Cdss )
{
    if (!fc)
        return -1;

    CUDAfc Cfc;
    Cfc.fo = fc->fo;

    for (int i = 0; i < fc->boundvariables.size(); i++)
    {
        if (fc->boundvariables[i]->CUDAfastidx >= 0)
        {
            CUDAnamedvariable Cnv;
            Cnv.t = CUDAnamedvariabletype::cnvcudafast;
            Cnv.ufc.CUDAfast = fc->boundvariables[i]->CUDAfastidx;
            Cnv.next = -1;
            Cdss.Cnames.push_back( {fc->boundvariables[i]->name,Cdss.Cnvv.size()});
            Cdss.Cnvv.push_back(Cnv);
        }
    }

    Cfc.criterion = addCfc(fc->criterion,Cdss);
    Cfc.fcleft = addCfc(fc->fcleft,Cdss);
    Cfc.fcright = addCfc(fc->fcright,Cdss);

    if (fc->fo == formulaoperator::foconstant || fc->fo == formulaoperator::fovariable)
    {
        // Cfc.v.v = fc->v.v;   // bizarre and not working. Instead do it manually
        Cfc.v.v.t = fc->v.v.t;
        if (fc->v.v.t == mtbool)
            Cfc.v.v.v.bv = fc->v.v.v.bv;
        else if (fc->v.v.t == mtdiscrete)
            Cfc.v.v.v.iv = fc->v.v.v.iv;
        else if (fc->v.v.t == mtcontinuous)
            Cfc.v.v.v.dv = fc->v.v.v.dv;
        else if (fc->v.v.t == mtset)
        {
            mttranslatetoCUDAset(fc->v.v.seti,Cfc.v.v,Cdss);
        } else if (fc->v.v.t == mttuple)
            std::cout << "No support yet for tuples within GPU\n";
    }
    if (fc->fo == formulaoperator::fovariable)
    {
        CUDAnamedvariable Cnv;
        if (fc->v.vs.l < 0)
        {
            stringlookup.strings.push_back(fc->v.vs.name);
            stringlookup.indices.push_back(Cdss.Cnvv.size());
            // assigncharpointertostring(Cnv.nm, fc->v.vs.name);
            // Cnv.nm = fc->v.vs.name;
            std::cout << "No support in GPU for non bound variable\n";
            exit(1);
        } else
        {
            Cnv.l = fc->v.vs.l;
            Cnv.bound = true;
        }
        bool found = false;
        for (int i = 0; i < Cdss.Cnames.size() && !found; i++)
            if (fc->v.vs.name == Cdss.Cnames[i].first)
                if (Cdss.Cnvv[Cdss.Cnames[i].second].t == CUDAnamedvariabletype::cnvcudafast)
                {
                    Cnv.t = CUDAnamedvariabletype::cnvcudafast;
                    Cnv.ufc.CUDAfast = Cdss.Cnvv[i].ufc.CUDAfast;
                    found = true;
                } else
                {
                    Cnv.t = Cdss.Cnvv[Cdss.Cnames[i].second].t;
                    Cnv.l = Cdss.Cnvv[Cdss.Cnames[i].second].l;
                    Cnv.ufc.v = Cfc.v.v;
                    found = true;
                }
        if (!found) // hokey way to say that the variable is a cudafast variable
        {
            Cnv.t = CUDAnamedvariabletype::cnvvalue;
            Cnv.ufc.v = Cfc.v.v;
        }
        Cnv.next = -1;
        Cdss.Cnvv.push_back(Cnv);
        Cfc.namedvar = Cdss.Cnvv.size()-1;
    } else
    {
        CUDAnamedvariableptr oldptr = -1;
        Cfc.namedvar = oldptr;/*
        for (int i = 0; i < fc->boundvariables.size(); i++)
        {
            CUDAnamedvariable Cnv;
            if (fc->boundvariables[i]->CUDAfastidx >= 0)
            {
                Cnv.t = CUDAnamedvariabletype::cnvcudafast;
                Cnv.ufc.CUDAfast = fc->boundvariables[i]->CUDAfastidx;
                Cnv.l = fc->v.vs.l;
                Cnv.next = -1;
                Cdss.Cnvv.push_back(Cnv);
            }
            // Cnv.nm = fc->boundvariables[i]->name;
            if (fc->boundvariables[i]->superset)
            {
                Cnv.ufc.superset = addCfc(CUDAfcv,fc->boundvariables[i]->superset, Cnvv, Cdss, Clv);
                Cnv.t = CUDAnamedvariabletype::cnvsuperset;
            } else if (fc->boundvariables[i]->alias)
            {
                Cnv.ufc.alias = addCfc(fc->boundvariables[i]->alias, Cdss);
                Cnv.t = CUDAnamedvariabletype::cnvalias;

            } else
            {
                Cnv.ufc.fcv = addCfc( CUDAfcv, fc->boundvariables[i]->value, Cnvv, Cdss, Clv);
                Cnv.t = CUDAnamedvariabletype::cnvfcvalue;
            }
            CUDAnamedvariableptr ptr = Cdss.Cnvv.size();
            Cnv.next = -1;
            Cdss.Cnvv.push_back(Cnv);
            if (i == 0)
                Cfc.namedvar = ptr;
            else
                Cdss.Cnvv[oldptr].next = ptr;
            oldptr = ptr;
        }*/
    }
    if (fc->fo == formulaoperator::fofunction || fc->fo == formulaoperator::foliteral)
    {
        bool found = false;
        // int i = 0;
        // while (!found && i++ < Clv.size())
            // found = fc->v.lit.lname == Clv[i].nm || fc->v.fns.nm == Clv[i].nm;
        // int i = stringlookup.lookup(fc->v.lit.lname);
        // if (i < 0)
            // i = stringlookup.lookup(fc->v.fns.nm);
        // if (i >= 0)
            // Cfc.literal = i;
        // else
        // {
            CUDAliteral Cl;
            std::string nm = fc->v.fns.nm;
            if (!CUDAfindfunction(Cdss.Cnvv, global_CUDAfnptrs, nm, Cl))
            {
                nm = fc->v.lit.lname;
                if (!CUDAfindfunction(Cdss.Cnvv, global_CUDAfnptrs, nm, Cl))
                {
                    // ... add other search domains...
                    std::cout << "CUDA not yet aware of function named " << fc->v.lit.lname << " or " << fc->v.fns.nm << std::endl;
                    exit(1);
                }
            }

            Cl.bound = true;
            // Cl.l = Clv.size(); // not to be confused, two different indices
            Cfc.literal = Cdss.Clv.size();
            stringlookup.strings.push_back(nm);
            stringlookup.indices.push_back(Cfc.literal);
            Cdss.Clv.push_back(Cl);
        // }

        CUDAnamedvariableptr oldptr = -1;
        CUDAnamedvariableptr ptr;
        for (int i = 0; i < fc->v.fns.ps.size(); ++i)
        {
            CUDAnamedvariable Cnv;
            Cnv.ufc.fcv = addCfc(fc->v.fns.ps[i],Cdss);
            Cnv.t = CUDAnamedvariabletype::cnvfcvalue;
            ptr = Cdss.Cnvv.size();
            if (oldptr == -1)
                Cfc.namedvar = ptr;
            else
                Cdss.Cnvv[oldptr].next = ptr;
            oldptr = ptr;
            Cdss.Cnvv.push_back(Cnv);
        }

        for (int i = 0; i < fc->v.lit.ps.size(); ++i)
        {
            CUDAnamedvariable Cnv;
            Cnv.ufc.fcv = addCfc(fc->v.lit.ps[i],Cdss);
            Cnv.t = CUDAnamedvariabletype::cnvfcvalue;
            ptr = Cdss.Cnvv.size();
            if (oldptr == -1)
                Cfc.namedvar = ptr;
            else
                Cdss.Cnvv[oldptr].next = ptr;
            oldptr = ptr;
            Cdss.Cnvv.push_back(Cnv);
        }
        if (oldptr >= 0)
            Cdss.Cnvv[Cdss.Cnvv.size()-1].next = -1;
        else
            Cfc.namedvar = -1;
    }

    Cdss.Cfcv.push_back(Cfc);
    CUDAfcptr res = Cdss.Cfcv.size()-1;
    return res;
}

void flattenformulaclassforCUDA( const formulaclass* fc, CUDAdataspaces& Cdss )
{

    Cdss.fctop = addCfc( fc, Cdss );
    // for (auto tempfc : Cfcv)
    // {
        // CUDAec.CUDAfcarray[i++] = tempfc;
#ifdef CUDADEBUG
        // std::cout << i-1 << " (" << int(tempfc.fo) << "): " << tempfc.criterion << ", " << tempfc.fcleft << ", " << tempfc.fcright << std::endl;
        // if (tempfc.namedvar != -1 && Cnvv[tempfc.namedvar].t == CUDAnamedvariabletype::cnvfcvalue)
        // {
            // std::cout << "   (args): ";
            // int nvptr = tempfc.namedvar;
            // while (nvptr != -1)
            // {
                // std::cout << Cnvv[nvptr].ufc.fcv << ", ";
                // nvptr = Cnvv[nvptr].next;
            // }
            // std::cout << std::endl;
        // }
#endif
    // }

    // ... populate CUDAvalsvector

    // CUDAec.CUDAvalsarraysize = 0;

    // Cdss.populateCUDAec(CUDAec);

    // CUDAec.namedvararraysize = Cnvv.size();
    // CUDAec.namedvararray = new CUDAnamedvariable[CUDAec.namedvararraysize];
    // CUDAnamedvariableptr j = 0;
    // for (auto tempvals : Cnvv)
        // CUDAec.namedvararray[j++] = tempvals;

    // CUDAec.CUDAliteralarraysize = Clv.size();
    // CUDAec.CUDAliteralarray = new CUDAliteral[CUDAec.CUDAliteralarraysize];
    // j = 0;
    // for (auto templiteral : Clv)
        // CUDAec.CUDAliteralarray[j++] = templiteral;
}

__device__ CUDAvalms CUDAevalinternal( CUDAextendedcontext& Cec, const CUDAfcptr Cfcin )
{

    if (Cfcin < 0)
    {
        // std::cout << "Error: abstract empty method called\n";
        // exit(1);
        CUDAvalms res;
        res.t = mtbool;
        res.v.bv = false;
        return res;
    }

    const auto Cfc = Cec.CUDAfcarray[Cfcin];

    switch (Cfc.fo)
    {
    case formulaoperator::foliteral:
    case formulaoperator::fofunction:
        {
            CUDAvalms res;
            CUDAnamedvariableptr j = Cec.CUDAliteralarray[Cfc.literal].inputvariabletypesptr;
            CUDAnamedvariableptr k = Cfc.namedvar;
            unsigned int argcnt = j >= 0;
            measuretype t0, t1;
            CUDAvalms v[2]; // the plan is use two "registers" as a best guess as to how many args are needed
            if (argcnt > 0)
            {
                t0 = Cec.namedvararray[j].ufc.v.t;
                v[0] = CUDAvalmsto_specified(Cec,CUDAevalinternal(Cec, Cec.namedvararray[k].ufc.fcv),t0);
                j = Cec.namedvararray[j].next;
                k = Cec.namedvararray[k].next;
                if (k >= 0)
                {
                    t1 = Cec.namedvararray[j].ufc.v.t;
                    v[1] = CUDAvalmsto_specified(Cec,CUDAevalinternal(Cec, Cec.namedvararray[k].ufc.fcv),t1);
                    argcnt = 2;
                    j = Cec.namedvararray[j].next;
                    k = Cec.namedvararray[k].next;
                }
            }
            while (j >= 0)
            {
                argcnt++;
                j = Cec.namedvararray[j].next;
            }
            CUDAvalms* args;
            bool needtodelete = false;
            if (argcnt > 2)
            {
                args = new CUDAvalms[argcnt];
                needtodelete = true;
                int l = 2;
                while (j >= 0 && k >= 0)
                {
                    measuretype t = Cec.namedvararray[j].ufc.v.t;
                    auto v = CUDAvalmsto_specified(Cec,CUDAevalinternal(Cec, Cec.namedvararray[k].ufc.fcv),t);
                    args[l++] = v;
                    j = Cec.namedvararray[j].next;
                    k = Cec.namedvararray[k].next;
                }
            }
            else
                args = v;
            res.t = Cec.CUDAliteralarray[Cfc.literal].t;
            switch (res.t)
            {
            case measuretype::mtcontinuous:
                {
                    res.v.dv = Cec.CUDAliteralarray[Cfc.literal].function.fncontinuous(Cec,args);
                    break;
                }
            case measuretype::mtdiscrete:
                {
                    res.v.iv = (Cec.CUDAliteralarray[Cfc.literal].function.fndiscrete)(Cec,args);
                    break;
                }
            case measuretype::mtbool:
                {
                    res.v.bv = Cec.CUDAliteralarray[Cfc.literal].function.fnbool(Cec,args);
                    break;
                }
            case measuretype::mtset:
                {
                    res.v.seti = Cec.CUDAliteralarray[Cfc.literal].function.fnset(Cec,args);
                    break;
                }
            }
            if (needtodelete)
                delete args;
            return res;
        }
    case formulaoperator::fone:
    case formulaoperator::foe:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            if (v1.t == mtbool)
                res.v.bv = v1.v.bv == CUDAto_mtbool(v2);
            else if (v1.t == mtdiscrete)
                res.v.bv = v1.v.iv == CUDAto_mtdiscrete(v2);
            else if (v1.t == mtcontinuous)
                res.v.bv = abs(v1.v.dv - CUDAto_mtcontinuous(v2)) < ABSCUTOFF;
            else if ((v1.t == mtset) && (v2.t == mtset) && (v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
            {
                res.v.bv = true;
                if (v1.v.seti.sz < v2.v.seti.sz)
                {
                    int i;
                    for (i = 0; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && ((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)] ==
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                    for (; i < v2.v.seti.sz && res.v.bv; ++i)
                        res.v .bv = res.v.bv && !((*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                } else
                {
                    int i;
                    for (i = 0; (i < v2.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && ((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)] ==
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                    for (; i < v1.v.seti.sz && res.v.bv; ++i)
                        res.v.bv = res.v.bv && !((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]);
                }
            } else
                res.v.bv = false;
            if (Cfc.fo == formulaoperator::fone)
                res.v.bv = !res.v.bv;
            return res; // add all remaining cases
        }
    case formulaoperator::folt:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            if (v1.t == mtbool)
                res.v.bv = v1.v.bv < CUDAto_mtbool(v2);
            else if (v1.t == mtdiscrete)
                res.v.bv = v1.v.iv < CUDAto_mtdiscrete(v2);
            else if (v1.t == mtcontinuous)
                res.v.bv = v1.v.dv < CUDAto_mtcontinuous(v2);
            else if ((v1.t == mtset) && (v2.t == mtset) && (v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
            { // for now just treat lt and lte the same when it comes to sets
                res.v.bv = true;
                if (v1.v.seti.sz < v2.v.seti.sz)
                {
                    int i;
                    for (i = 0; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                } else
                {
                    int i;
                    for (i = 0; (i < v2.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                    for (; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && !((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]);
                }
            } else
                res.v.bv = false;
            return res; // add all remaining cases
        }
    case formulaoperator::fogt:
        {
            auto v2 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v1 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            if (v1.t == mtbool)
                res.v.bv = v1.v.bv < CUDAto_mtbool(v2);
            else if (v1.t == mtdiscrete)
                res.v.bv = v1.v.iv < CUDAto_mtdiscrete(v2);
            else if (v1.t == mtcontinuous)
                res.v.bv = v1.v.dv < CUDAto_mtcontinuous(v2);
            else if ((v1.t == mtset) && (v2.t == mtset) && (v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
            { // for now just treat gt and gte the same when it comes to sets
                res.v.bv = true;
                if (v1.v.seti.sz < v2.v.seti.sz)
                {
                    int i;
                    for (i = 0; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                } else
                {
                    int i;
                    for (i = 0; (i < v2.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                    for (; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && !((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]);
                }
            } else
                res.v.bv = false;
            return res; // add all remaining cases
        }
    case formulaoperator::fogte:
        {
            auto v2 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v1 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            if (v1.t == mtbool)
                res.v.bv = v1.v.bv <= CUDAto_mtbool(v2);
            else if (v1.t == mtdiscrete)
                res.v.bv = v1.v.iv <= CUDAto_mtdiscrete(v2);
            else if (v1.t == mtcontinuous)
                res.v.bv = v1.v.dv <= CUDAto_mtcontinuous(v2);
            else if ((v1.t == mtset) && (v2.t == mtset) && (v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
            {
                res.v.bv = true;
                if (v1.v.seti.sz < v2.v.seti.sz)
                {
                    int i;
                    for (i = 0; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                } else
                {
                    int i;
                    for (i = 0; (i < v2.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                    for (; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && !((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]);
                }
            } else
                res.v.bv = false;
            return res; // add all remaining cases
        }

    case formulaoperator::folte:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            if (v1.t == mtbool)
                res.v.bv = v1.v.bv <= CUDAto_mtbool(v2);
            else if (v1.t == mtdiscrete)
                res.v.bv = v1.v.iv <= CUDAto_mtdiscrete(v2);
            else if (v1.t == mtcontinuous)
                res.v.bv = v1.v.dv <= CUDAto_mtcontinuous(v2);
            else if ((v1.t == mtset) && (v2.t == mtset) && (v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
            {
                res.v.bv = true;
                if (v1.v.seti.sz < v2.v.seti.sz)
                {
                    int i;
                    for (i = 0; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                } else
                {
                    int i;
                    for (i = 0; (i < v2.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && (!((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]) ||
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]);
                    for (; (i < v1.v.seti.sz) && res.v.bv; ++i)
                        res.v.bv = res.v.bv && !((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]);
                }
            } else
                res.v.bv = false;
            return res; // add all remaining cases
        }

    case formulaoperator::foplus:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = (v1.t == mtcontinuous) ? mtcontinuous : ((v2.t == mtcontinuous) ? mtcontinuous : v1.t);
            if (res.t == mtbool)
                res.v.bv = CUDAto_mtbool(v1) + CUDAto_mtbool(v2);
            else if (res.t == mtdiscrete)
                res.v.iv = CUDAto_mtdiscrete(v1) + CUDAto_mtdiscrete(v2);
            else if (res.t == mtcontinuous)
                res.v.dv = CUDAto_mtcontinuous(v1) + CUDAto_mtcontinuous(v2);
            return res;
        }
    case formulaoperator::fominus:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = (v1.t == mtcontinuous) ? mtcontinuous : (v2.t == mtcontinuous ? mtcontinuous : v1.t);
            if (res.t == mtbool)
                res.v.bv = CUDAto_mtbool(v1) - CUDAto_mtbool(v2);
            else if (res.t == mtdiscrete)
                res.v.iv = CUDAto_mtdiscrete(v1) - CUDAto_mtdiscrete(v2);
            else if (res.t == mtcontinuous)
                res.v.dv = CUDAto_mtcontinuous(v1) - CUDAto_mtcontinuous(v2);
            return res;
        }
    case formulaoperator::fotimes:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            if (v1.t == mtbool && v2.t == mtbool)
                res.t = mtbool;
            else
                res.t = v1.t == mtcontinuous ? mtcontinuous : (v2.t == mtcontinuous ? mtcontinuous : mtdiscrete);
            if (res.t == mtbool)
                res.v.bv = CUDAto_mtbool(v1) * CUDAto_mtbool(v2);
            else if (res.t == mtdiscrete)
                res.v.iv = CUDAto_mtdiscrete(v1) * CUDAto_mtdiscrete(v2);
            else if (res.t == mtcontinuous)
                res.v.dv = CUDAto_mtcontinuous(v1) * CUDAto_mtcontinuous(v2);
            return res;
        }
    case formulaoperator::fodivide:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtcontinuous;
            res.v.dv = CUDAto_mtcontinuous(v1) / CUDAto_mtcontinuous(v2);
            return res;
        }
    case formulaoperator::foand:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            res.v.bv = CUDAto_mtbool(v1) && CUDAto_mtbool(v2);
            // ... add support for set types
            return res;
        }
    case formulaoperator::foor:
        {
            CUDAvalms res;
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            res.t = mtbool;
            res.v.bv = CUDAto_mtbool(v1) || CUDAto_mtbool(v2);
            // ... add support for set types
            return res;
        }
    case formulaoperator::foiff:
        {
            CUDAvalms res;
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            res.t = mtbool;
            res.v.bv = CUDAto_mtbool(v1) == CUDAto_mtbool(v2);
            // ... add support for set types
            return res;
        }
    case formulaoperator::foimplies:
        {
            CUDAvalms res;
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            res.t = mtbool;
            res.v.bv = (!CUDAto_mtbool(v1)) || CUDAto_mtbool(v2);
            // ... add support for set types
            return res;
        }
    case formulaoperator::foif:
        {
            CUDAvalms res;
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            res.t = mtbool;
            res.v.bv = CUDAto_mtbool(v1) || (!CUDAto_mtbool(v2));
            // ... add support for set types
            return res;
        }
    case formulaoperator::fonot:
        {
            auto v = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            res.v.bv = !CUDAto_mtbool(v);
            // ... add support for set types
            return res;
        }
    case formulaoperator::foconstant:
        {
            CUDAvalms res;
            res = Cfc.v.v;
            return res;
        }
    case formulaoperator::fovariable:
        {
            CUDAvalms res;
            if (Cec.namedvararray[Cfc.namedvar].t == CUDAnamedvariabletype::cnvcudafast)
            {
                res.t = mtdiscrete;
                res.v.iv = Cec.fastn[Cec.namedvararray[Cfc.namedvar].ufc.CUDAfast];
            } else
                res = Cec.CUDAcontext[Cec.namedvararray[Cfc.namedvar].l].ufc.v;
            return res;
        }
    case formulaoperator::fovariablederef:
        {
            // ... to do
            CUDAvalms res;
            res = Cec.CUDAcontext[Cec.namedvararray[Cfc.namedvar].l].ufc.v;
            return res;
        }
    case formulaoperator::founion:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtset;
            res.v = Cec.namedvararray[Cfc.namedvar].ufc.v.v;
            // res.v.seti.ptr = Cfc.v.v.v.seti.ptr;
            // res.v.seti.st = Cfc.v.v.v.seti.st;
            if ((v1.t == mtset) && (v2.t == mtset))
                if ((v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
                {
                    if (v1.v.seti.sz < v2.v.seti.sz)
                    {
                        res.v.seti.sz = v2.v.seti.sz;
                        // memcpy(((bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr],((bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr],v2.v.seti.sz*sizeof(bool));
                        for (auto i = 0; i < v2.v.seti.sz; ++i)
                            (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] =
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)];
                        for (auto i = 0; i < v1.v.seti.sz; ++i)
                            if ((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)])
                                (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] = true;
                    } else {
                        res.v.seti.sz = v1.v.seti.sz;
                        for (auto i = 0; i < v1.v.seti.sz; ++i)
                            (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] =
                                (*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)];
                        // memcpy(((bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr],((bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr],v1.v.seti.sz*sizeof(bool));
                        for (auto i = 0; i < v2.v.seti.sz; ++i)
                            if ((*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)])
                                (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] = true;
                    }
                }
            return res;
        }
    case formulaoperator::fointersection:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtset;
            res.v = Cec.namedvararray[Cfc.namedvar].ufc.v.v;
            // res.v.seti.ptr = Cfc.v.v.v.seti.ptr;
            // res.v.seti.st = Cfc.v.v.v.seti.st;
            if ((v1.t == mtset) && (v2.t == mtset))
                if ((v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
                {
                    if (v1.v.seti.sz > v2.v.seti.sz)
                        res.v.seti.sz = v2.v.seti.sz;
                    else
                        res.v.seti.sz = v1.v.seti.sz;
                    // memcpy(((bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr],((bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr],v2.v.seti.sz*sizeof(bool));
                    for (auto i = 0; i < res.v.seti.sz; ++i)
                        (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] =
                            (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]
                            &&((*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]);
                } else
                    res.v.seti.sz = Cfc.v.v.v.seti.sz;
            return res;
        }
    case formulaoperator::fosetxor:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtset;
            res.v = Cec.namedvararray[Cfc.namedvar].ufc.v.v;
            // res.v.seti.ptr = Cfc.v.v.v.seti.ptr;
            // res.v.seti.st = Cfc.v.v.v.seti.st;
            // res.v.seti.sz = Cfc.v.v.v.seti.sz;
            if ((v1.t == mtset) && (v2.t == mtset))
                if ((v1.v.seti.st == v2.v.seti.st) && (v1.v.seti.st == mtdiscrete))
                {
                    if (v1.v.seti.sz < v2.v.seti.sz)
                    {
                        res.v.seti.sz = v2.v.seti.sz;
                        // memcpy(((bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr],((bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr],v2.v.seti.sz*sizeof(bool));
                        for (auto i = 0; i < v2.v.seti.sz; ++i)
                            (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] =
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)];
                        for (auto i = 0; i < v1.v.seti.sz; ++i)
                            (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] =
                                (*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)]
                                != (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)];
                    } else {
                        res.v.seti.sz = v1.v.seti.sz;
                        // memcpy(((bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr],((bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr],v2.v.seti.sz*sizeof(bool));
                        for (auto i = 0; i < v1.v.seti.sz; ++i)
                            (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] =
                                (*(bool**)&Cec.CUDAvalsarray)[v1.v.seti.ptr + i * sizeof(bool)];
                        for (auto i = 0; i < v2.v.seti.sz; ++i)
                            (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)] =
                                (*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + i * sizeof(bool)]
                                != (*(bool**)&Cec.CUDAvalsarray)[res.v.seti.ptr + i * sizeof(bool)];
                    }
                }
            return res;
        }
    case formulaoperator::foelt:
        {
            auto v1 = CUDAevalinternal(Cec,Cfc.fcleft);
            auto v2 = CUDAevalinternal(Cec,Cfc.fcright);
            CUDAvalms res;
            res.t = mtbool;
            if ((v2.t == mtset)) // && (v1.t == v2.v.seti.st))
                switch (v1.t)
                {
                case mtdiscrete:
                    res.v.bv = ((*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + v1.v.iv*sizeof(bool)]);
                    break;
                case mtcontinuous:
                    res.v.bv = ((*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + CUDAto_mtdiscrete(v1)*sizeof(bool)]);
                    break;
                case mtset:
                    // ... to do
                    res.v.bv = ((*(bool**)&Cec.CUDAvalsarray)[v2.v.seti.ptr + v1.v.iv*sizeof(bool)]);
                    break;
                }
            return res;
        }
        /*
    case formulaoperator::foqexists:
        {



            for (int i = 0; i < Cec.CUDAcontextsize; ++i)
            {
                if (Cec.CUDAcontext[i].ufc.v.t == mtset)
                    if (Cec.CUDAcontext[i].ufc.v.v.seti.st == mtdiscrete)
                        for (auto j = 0; j < Cec.CUDAcontext[i].ufc.v.v.seti.sz; ++j)
                        {
                            void* v = Cec.CUDAvalsarray + Cec.CUDAcontext[i].ufc.v.v.seti.ptr + sizeof(bool)*j;
                            std::cout << *static_cast<bool*>(v) << " ";
                        }
                else
                    std::cout << Cec.CUDAcontext[i].ufc.v.v.iv << ", ";
                std::cout << std::endl;
            }

            auto res = CUDAevalinternal(Cec,Cfc.fcright);
            return res;
        }*/
    default:
        {
            CUDAvalms res;
            res.v.bv = false;
            res.t = mtbool;
            return res;

            // std::cout << "No support yet for this formulaoperator\n";
            // exit(1);
        }

        // ...

    }

}


#endif FLAGCALC_CUDA