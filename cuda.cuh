//
// Created by peterglenn on 5/4/25.
//

#ifndef CUDA_CUH
#define CUDA_CUH

#include "config.h"

#ifdef FLAGCALC_CUDA

#include "math.h"

// #define CUDADEBUG
// #define CUDADEBUG2
#define GPUQUANTFASTDIM 3
// #define CUDAFORCOMPUTENEIGHBORSLIST
// #define CUDAFORCOMPUTENEIGHBORSLISTENMASSE


struct CUDAvalms;

struct CUDAgraph {
    int dim;
    bool* adjacencymatrix;
};


struct CUDAneighbors
{
    CUDAgraph* g;
    int* neighborslist;
    int* degrees;
    int maxdegree;
    int* nonneighborslist;
};


using CUDAvalmsptr = int;

using CUDAvalsptr = int;

struct CUDAseti
{
    measuretype st;
    CUDAvalsptr ptr;
    uint sz;
};

union CUDAvals
{
    long int iv;
    bool bv;
    double dv;
    CUDAseti seti;
};

struct CUDAvalms
{
    measuretype t = mtbool;
    CUDAvals v;

    /*
    CUDAvalms operator=(const valms& a1)
    {
        CUDAvalms a2;
        a2.t = a1.t;
        switch (a1.t) {
        case measuretype::mtcontinuous:
            a2.v.dv = a1.v.dv;
            break;
        case measuretype::mtdiscrete:
            a2.v.iv = a1.v.iv;
            break;
        case measuretype::mtbool:
            a2.v.bv = a1.v.bv;
            break;
        default:
            std::cout << "Not yet implemented operator = between CUDAvalms and valms type\n";
            break;
        }
        return a2;
    }*/

};


struct CUDAfv {
    CUDAvalms v;
    // litstruct lit;
    // fnstruct fns;
    // std::vector<qclass*> qcs;
    // formulaclass* criterion;
    // variablestruct vs;
    // setstruct ss;
    // bool subgraph;
};



struct CUDAfc;

using CUDAfcptr = int;

enum class CUDAnamedvariabletype  {cnvfcvalue, cnvsuperset, cnvalias, cnvvalue, cnvcudafast};

union CUDAnamedvariables
{
    CUDAfcptr fcv;
    CUDAfcptr superset;
    CUDAfcptr alias;
    CUDAvalms v;
    int CUDAfast;
};

using CUDAnamedvariableptr = int;

inline class stringlookupclass
{
public:
    std::vector<std::string> strings;
    std::vector<int> indices;
    int lookup(std::string s)
    {
        int i = 0;
        bool found = false;
        while (i < strings.size())
            if (strings[i] == s)
            {
                found = true;
                return indices[i];
            } else i++;
        return -1;
    }
} stringlookup {};

struct CUDAnamedvariable
{
    // stringptr nm = -1;

    bool bound = false;
    uint l = 0;

    CUDAnamedvariabletype t = CUDAnamedvariabletype::cnvvalue;
    CUDAnamedvariables ufc {};
    CUDAnamedvariableptr next {};

};

struct CUDAextendedcontext;

union CUDAfunction
{
    bool (*fnbool)(const CUDAextendedcontext&, const CUDAvalms*);
    long int (*fndiscrete)(const CUDAextendedcontext&, const CUDAvalms*);
    double (*fncontinuous)(const CUDAextendedcontext&, const CUDAvalms*);
    CUDAseti (*fnset)(const CUDAextendedcontext&, const CUDAvalms*);
    CUDAseti (*fntuple)(const CUDAextendedcontext&, const CUDAvalms*);
};

struct CUDAliteral
{
    // stringptr nm = -1;
    bool bound = false;
    uint l = 0;

    measuretype t = measuretype::mtcontinuous;
    CUDAnamedvariableptr inputvariabletypesptr = -1;

    CUDAfunction function;

};

using CUDAliteralptr = int;

struct CUDAfc
{
    CUDAfcptr fcleft;
    CUDAfcptr fcright;
    CUDAfcptr criterion;
    CUDAfv v;
    formulaoperator fo;
    CUDAnamedvariableptr namedvar;
    CUDAliteralptr literal; // includes fns.fn
};

using CUDAvdimn = long int[GPUQUANTFASTDIM];

struct CUDAextendedcontext
{

    CUDAfc* CUDAfcarray;
    uint CUDAfcarraysize;
    CUDAfcptr fctop;

    CUDAnamedvariable* namedvararray;
    uint namedvararraysize;

    void* CUDAvalsarray;
    uint CUDAvalsarraysize;

    CUDAnamedvariable* CUDAcontext;
    uint CUDAcontextsize;
    CUDAnamedvariableptr contextoffset;

    CUDAliteral* CUDAliteralarray;
    uint CUDAliteralarraysize;

    CUDAvdimn fastn;
    uint numfastn;

    CUDAgraph g;
    CUDAneighbors ns;
};


struct CUDAdataspace
{
    void* data;
    bool needsdelete = false;
    uint sz;
    uint szfactor = sizeof(bool);
};

class CUDAdataspaces
{
public:
    std::vector<std::pair<std::string,CUDAnamedvariableptr>> Cnames {};
    std::vector<CUDAdataspace> Csv {};
    std::vector<CUDAnamedvariable> Cnvv {};
    std::vector<CUDAfc> Cfcv {};
    CUDAfcptr fctop;
    std::vector<CUDAliteral> Clv {};
    std::vector<CUDAnamedvariable> Ccv {}; // CUDAcontext

    graphtype* g;
    neighborstype* ns;

    uint totalsz = 0;

    void clear()
    {
        for (auto p : Csv)
        {
            if (p.needsdelete)
                delete p.data;
        }
        Csv.clear();
        totalsz = 0;
        Cnvv.clear();
        Cfcv.clear();
        Clv.clear();
        Ccv.clear();
    }

    bool checktotalsz()
    {
        uint temptotalsz = 0;
        for (auto Cds : Csv)
            temptotalsz += Cds.sz * Cds.szfactor;
        return temptotalsz == totalsz;
    }

    void mergeCUDAdataspaces( CUDAdataspaces& Cdstarget )
    {
        for (auto Cds : Csv)
            Cdstarget.Csv.insert(Cdstarget.Csv.begin(),Cds);
        Cdstarget.totalsz += totalsz;
        for (auto Cnv : Cnvv)
            Cdstarget.Cnvv.insert( Cdstarget.Cnvv.begin(), Cnv );
        for (auto Cfc : Cfcv)
            Cdstarget.Cfcv.insert(Cdstarget.Cfcv.begin(), Cfc );
        Cdstarget.fctop = fctop;
        for (auto Cl : Clv)
            Cdstarget.Clv.insert( Cdstarget.Clv.begin(), Cl );
        for (auto Cc : Ccv)
            Cdstarget.Ccv.insert( Cdstarget.Ccv.begin(), Cc );
        Cdstarget.g = g;
    }

    void copyCUDAdataspaces( CUDAdataspaces& Cdstarget )
    {
        Cdstarget.clear();
        for (auto Cds : Csv)
            Cdstarget.Csv.push_back(Cds);
        Cdstarget.totalsz = totalsz;
        for (auto Cnv : Cnvv)
            Cdstarget.Cnvv.push_back( Cnv );
        for (auto Cfc : Cfcv)
            Cdstarget.Cfcv.push_back(Cfc );
        Cdstarget.fctop = fctop;
        for (auto Cl : Clv)
            Cdstarget.Clv.push_back( Cl );
        for (auto Cc : Ccv)
            Cdstarget.Ccv.push_back( Cc );
        Cdstarget.g = g;
        Cdstarget.ns = ns;
    }

    void populateCUDAecvolatileonly( CUDAextendedcontext& Cec )
    {
        if (!checktotalsz())
        {
            std::cout << "Error in CUDAdataspace bookkeeping\n";
            exit(1);
        }
        auto oldvals = Cec.CUDAvalsarray;
        auto oldsz = Cec.CUDAvalsarraysize;
        Cec.CUDAvalsarraysize = totalsz + oldsz;
        Cec.CUDAvalsarray = new CUDAdataspace[totalsz + oldsz];
        memcpy(Cec.CUDAvalsarray, oldvals, oldsz);
        CUDAvalsptr i = oldsz;
        for (auto v : Csv)
        {
            void* ptr = Cec.CUDAvalsarray + i;
            if (v.data)
                memcpy(ptr,v.data,v.sz*v.szfactor);
            i += v.sz * v.szfactor;
        }
        if (oldsz)
            delete [] oldvals;

        //

        if (Cec.CUDAcontextsize > 0)
            delete[] Cec.CUDAcontext;

        Cec.CUDAcontextsize = Ccv.size();
        Cec.CUDAcontext = new CUDAnamedvariable[Cec.CUDAcontextsize];
        i = 0;
        for (auto v : Ccv)
            Cec.CUDAcontext[i++] = v;

        //

        if (Cec.namedvararraysize > 0)
            delete[] Cec.namedvararray;

        Cec.namedvararraysize = Cnvv.size();
        Cec.namedvararray = new CUDAnamedvariable[Cec.namedvararraysize];
        i = 0;
        for (auto v : Cnvv)
            Cec.namedvararray[i++] = v;

    }


    void populateCUDAec( CUDAextendedcontext& Cec )
    {
        populateCUDAecvolatileonly( Cec );

        //
        int i;

        if (Cec.CUDAfcarraysize > 0)
            delete[] Cec.CUDAfcarray;
        Cec.CUDAfcarraysize = Cfcv.size();
        Cec.CUDAfcarray = new CUDAfc[Cec.CUDAfcarraysize];
        i = 0;
        for (auto fc : Cfcv)
            Cec.CUDAfcarray[i++] = fc;
        Cec.fctop = fctop;

        //

        if (Cec.CUDAliteralarraysize > 0)
            delete[] Cec.CUDAliteralarray;
        Cec.CUDAliteralarraysize = Clv.size();
        Cec.CUDAliteralarray = new CUDAliteral[Cec.CUDAliteralarraysize];
        i = 0;
        for (auto l : Clv)
            Cec.CUDAliteralarray[i++] = l;

        //

        Cec.g.adjacencymatrix = g->adjacencymatrix;
        Cec.g.dim = g->dim;
        Cec.ns.maxdegree = ns->maxdegree;
        Cec.ns.neighborslist = ns->neighborslist;
        Cec.ns.nonneighborslist = ns->nonneighborslist;
        Cec.ns.degrees = ns->degrees;
    }
};


void flattencontextforCUDA( const namedparams& context, const uint offset, CUDAdataspaces* Cdss, CUDAdataspaces& CdssNEW );

uint CUDAprognosticatespaceneeded( CUDAdataspaces& Cdss, CUDAfcptr& fctop );

void flattenformulaclassforCUDA( const formulaclass* fc, CUDAdataspaces& Cdss );

__device__ CUDAvalms CUDAevalinternal( CUDAextendedcontext& Cec, const CUDAfcptr Cfcin );



inline CUDAvalms to_mtbool( const CUDAvalms v )
{
    CUDAvalms res;
    res.t = mtbool;
    switch (v.t)
    {
    case mtbool:
        {
            res.v.bv = v.v.bv;
            break;
        }
    case mtdiscrete:
        {
            res.v.bv = (bool)v.v.iv;
            break;
        }
    case mtcontinuous:
        {
            res.v.bv = (bool)v.v.dv;
            break;
        }
    default:
        std::cout << "Not yet implemented to_mtbool type\n";
        res.v.bv = false;
        break;
    }
    return res;
}



inline CUDAvalms to_mtdiscrete( const CUDAvalms v )
{
    CUDAvalms res;
    res.t = mtdiscrete;
    switch (v.t)
    {
    case mtbool:
        res.v.iv = v.v.bv ? 1 : 0;
        break;
    case mtdiscrete:
        res.v.iv = v.v.iv;
        break;
    case mtcontinuous:
        res.v.iv = v.v.dv;
        break;
    default:
        std::cout << "Not yet implemented to_mtdiscrete type\n";
        break;
    }
    return res;
}



inline CUDAvalms to_mtcontinuous( const CUDAvalms v )
{
    CUDAvalms res;
    res.t = mtcontinuous;
    switch (v.t)
    {
    case mtbool:
        res.v.dv = v.v.bv;
        break;
    case mtdiscrete:
        res.v.dv = v.v.iv;
        break;
    case mtcontinuous:
        res.v.dv = v.v.dv;
        break;
    default:
        std::cout << "Not yet implemented to_mtcontinuous type\n";
        break;
    }
    return res;
}

inline CUDAvalms to_mtset( const CUDAvalms v )
{
    CUDAvalms res;
    res.t = mtset;
    switch (v.t)
    {
    case mtbool:
        {
            res.v.bv = v.v.bv;
            res.v.seti.st = mtbool;
            res.v.seti.sz = 1;
            std::cout << "No support for forming sets within a GPU call\n";
            break;
        }
    case mtdiscrete:
        {
            res.v.dv = v.v.bv;
            res.v.seti.st = mtdiscrete;
            res.v.seti.sz = 1;
            std::cout << "No support for forming sets within a GPU call\n";
            res.v.dv = v.v.iv;
            break;
        }
    case mtcontinuous:
        {
            res.v.dv = v.v.bv;
            res.v.seti.st = mtcontinuous;
            res.v.seti.sz = 1;
            std::cout << "No support for forming sets within a GPU call\n";
            break;
        }
    case mtset:
        {
            res.v.seti.sz = v.v.seti.sz;
            res.v.seti.st = v.v.seti.st;
            res.v.seti.ptr = v.v.seti.ptr;
            break;
        }
    case mttuple:
        {
            if (res.t != mttuple)
                std::cout << "No support for converting tuple to set within a GPU call\n";
            res.v.seti.sz = v.v.seti.sz;
            res.v.seti.st = v.v.seti.st;
            res.v.seti.ptr = v.v.seti.ptr;
            break;
        }
    default:
        {
            std::cout << "Not yet implemented to_mtcontinuous type\n";
            break;
        }
    }
    return res;
}


/*
 *valms operator=(const CUDAvalms& a1)
{
    valms a2;
    a2.t = a1.t;
    switch (a1.t) {
    case measuretype::mtcontinuous:
        a2.v.dv = a1.v.dv;
        break;
    case measuretype::mtdiscrete:
        a2.v.iv = a1.v.iv;
        break;
    case measuretype::mtbool:
        a2.v.bv = a1.v.bv;
        break;
    default:
        std::cout << "Not yet implemented operator = between CUDAvalms and valms type\n";
        break;
    }
    return a2;
}
*/

inline bool operator==(const CUDAvalms& a1, const valms& a2)
{
    if (a1.t != a2.t)
        return false;
    switch (a1.t) {
    case measuretype::mtcontinuous:
        return a2.v.dv == a1.v.dv;
    case measuretype::mtdiscrete:
        return a2.v.iv == a1.v.iv;
    case measuretype::mtbool:
        return a2.v.bv == a1.v.bv;
    default:
        std::cout << "Not yet implemented operator == between CUDAvalms and valms type\n";
        return false;
    }
}

inline valms CUDAtovalms(const CUDAvalms& a1)
{
    valms a2;
    a2.t = a1.t;
    switch (a1.t) {
    case measuretype::mtcontinuous:
        a2.v.dv = a1.v.dv;
        break;
    case measuretype::mtdiscrete:
        a2.v.iv = a1.v.iv;
        break;
    case measuretype::mtbool:
        a2.v.bv = a1.v.bv;
        break;
    default:
        std::cout << "Not yet implemented operator = between CUDAvalms and valms type\n";
        break;
    }
    return a2;
}


inline bool operator==(const valms& a1, const CUDAvalms& a2)
{
    if (a1.t != a2.t)
        return false;
    switch (a1.t) {
    case measuretype::mtcontinuous:
        return a2.v.dv == a1.v.dv;
    case measuretype::mtdiscrete:
        return a2.v.iv == a1.v.iv;
    case measuretype::mtbool:
        return a2.v.bv == a1.v.bv;
    default:
        std::cout << "Not yet implemented operator == between CUDAvalms and valms type\n";
        return false;
    }
}

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

__device__ inline long int CUDAto_mtdiscrete( const CUDAvalms v )
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

__device__ inline CUDAseti newset( const CUDAextendedcontext& Cec, const CUDAvalsptr& Cvptr, const measuretype& mt, const uint& sz, void* data )
{
    memcpy(((void**)&Cec.CUDAvalsarray)[Cvptr],data,sz);
    CUDAseti res;
    res.ptr = Cvptr;
    res.sz = sz;
    res.st = mt;
    delete data;
    return res;
}

__device__ inline CUDAseti CUDAto_mtset( const CUDAextendedcontext& Cec, const CUDAvalsptr& Cvptr, const measuretype& mt, const uint& sz, const CUDAvalms& v )
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

__device__ inline CUDAvalms CUDAlookupnamedvariable(const CUDAextendedcontext& Cec, const CUDAnamedvariableptr& Cnvptr, uint count, const measuretype& mt)
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

#endif // FLAGCALC_CUDA

#endif // CUDA_CUH