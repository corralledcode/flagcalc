//
// Created by peterglenn on 12/12/25.
//

#include "config.h"

#ifdef FLAGCALC_CUDA

#include "ameas.h"
#include "graphs.h"
#include <unordered_set>
#include "cudaengine.cuh"

inline void CUDAverticesconnectedmatrix( bool* out, const graphtype* g, const neighborstype* ns )
{

    const int dim = g->dim;
    // auto out = (bool*)malloc(dim*dim*sizeof(bool));
    memcpy(out,g->adjacencymatrix,dim*dim*sizeof(bool));
    for (int i = 0; i < dim; ++i)
        out[i*dim + i] = true;
    CUDAverticesconnectedmatrixwrapper(g,ns,out);
}


class CUDAnwalksbetweentuple : public set
{
public:
    CUDAnwalksbetweentuple( mrecords* recin ) :
        set( recin, "CUDAnwalksbetweenp", "CUDA array of walk counts of given length between each vertex pair")
    {
        valms v;
        v.t = mtdiscrete;
        nps.push_back(std::pair{"walk length",v});
        bindnamedparams();
    };
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        int dim = ns->g->dim;
        int* out;
        out = new int[dim*dim];
        CUDAcountpathsbetweenwrapper(out,ps[0].v.iv,ns->g->adjacencymatrix,dim);
        return new setitrtuple<int>(dim*dim,out);
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class CUDAConnvtuple : public set
{
public:
    CUDAConnvtuple( mrecords* recin ) :
        set( recin, "CUDAConnv", "CUDA boolean matrix of connected vertices") {};
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        const int dim = g->dim;
        bool* out = new bool[dim*dim];
        CUDAverticesconnectedmatrix( out, g, ns );
        std::vector<setitrtuple<bool>*> tuples {};
        tuples.resize(dim);
        for (int i = 0; i < dim; ++i)
            tuples[i] = new setitrtuple<bool>(dim,&out[i*dim]);
        auto res = new setitrtuple2d<bool>(tuples);
        return res;
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

#endif // FLAGCALC_CUDA