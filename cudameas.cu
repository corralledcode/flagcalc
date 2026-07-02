//
// Created by peterglenn on 12/12/25.
//

// #include "config.h"

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

inline void CUDAverticesconnectedmatrixoptimized( bool* out, const graphtype* g, const neighborstype* ns )
{

    // const int dim = g->dim;
    // auto out = (bool*)malloc(dim*dim*sizeof(bool));
    // memcpy(out,g->adjacencymatrix,dim*dim*sizeof(bool));
    // for (int i = 0; i < dim; ++i)
        // out[i*dim + i] = true;
    // CUDAverticesconnectedmatrixwrapper(g,ns,out);
}

inline void CUDAconnectedpartition( const graphtype* g, const neighborstype* ns, std::vector<bool*>& outv )
{
    int num_vertices = g->dim;
    std::vector<int> src {};
    std::vector<int> dst {};
    for (int i = 0; i < num_vertices; ++i)
        for (int j = 0; j < num_vertices; ++j)
            if (g->adjacencymatrix[i*num_vertices+j])
            {
                src.push_back(i);
                dst.push_back(j);
            }

    std::vector<int> parent;
    find_connected_components(src, dst, num_vertices, parent);

/* Commented out at least six times slower than what follows the comment
    std::vector<bool> used;
    used.resize(num_vertices);
    outv.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
    {
        used[i] = false;
        // outv[i] = new bool[num_vertices];
        // memset(outv[i],false,num_vertices*sizeof(bool));
    }
    for (int i = 0; i < num_vertices; ++i)
    {
        if (!used[parent[i]])
        {
            outv[parent[i]] = new bool[num_vertices];
            memset(outv[parent[i]],false,num_vertices*sizeof(bool));
            used[parent[i]] = true;
        }
        outv[parent[i]][i] = true;
    }
    for (int i = 0; i < outv.size(); )
    {
        if (!used[i])
        {
            outv.erase(outv.begin()+i);
            used.erase(used.begin()+i);
        } else
            ++i;
    }
    */


    std::vector<int> lookup;
    lookup.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i)
        lookup[i] = -1;
    int j = 0;
    for (int i = 0; i < num_vertices; ++i)
    {
        if (lookup[parent[i]] == -1)
        {
            lookup[parent[i]] = j;
            ++j;
        }
    }

    outv.resize(j);

    for (int i = 0; i < j; ++i)
    {
        outv[i] = new bool[num_vertices];
        memset(outv[i],false,sizeof(bool)*num_vertices);

        // std::cout << "vertex " << i << ": parent " << parent[i] << std::endl;
    }
    for (int i = 0; i < num_vertices; ++i)
        outv[lookup[parent[i]]][i] = true;

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

        std::vector<setitrtuple<int>*> tuples {};
        tuples.resize(dim);
        for (int i = 0; i < dim; ++i)
            tuples[i] = new setitrtuple<int>(dim,&out[i*dim]);
        auto res = new setitrtuple2d<int>(tuples);

        return res;
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

class CUDAConnset : public set
{
public:
    CUDAConnset( mrecords* recin ) :
        set( recin, "CUDAConns", "Set of connected components") {};
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        const int dim = g->dim;

        std::vector<bool*> outv;

        CUDAconnectedpartition(g,ns,outv);

        std::vector<valms> res {};
        for (auto elt : outv)
        {
            valms v;
            v.t = mtset;
            v.seti = new setitrint(g->dim-1,elt);
            res.push_back(v);
        }
        return new setitrmodeone(res);
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};


#endif // FLAGCALC_CUDA