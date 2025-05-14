//
// Created by peterglenn on 5/13/25.
//

#ifndef CUDAGRAPH_CUH
#define CUDAGRAPH_CUH
#include <iostream>

#include "cuda.cuh"
#include "cudaengine.cuh"

using CUDAvertextype = int;


__device__ inline void CUDAcomputeneighborslistsingle(const CUDAgraph* g, CUDAneighbors* ns, const int index)
{
    if (index < g->dim)
    {
        int degree = 0;
        for (int v2 = 0; v2 < g->dim; ++v2)
        {
            if (g->adjacencymatrix[index*g->dim+v2])
            {
                ns->neighborslist[index*g->dim+degree] = v2;
                ++degree;
            }
        }
        ns->degrees[index] = degree;;
        int nondegree = 0;
        for (int i = 0; i < g->dim; ++i) {
            if (!g->adjacencymatrix[index*(g->dim) + i] && (index != i)) {
                ns->nonneighborslist[index*(g->dim) + nondegree] = i;
                ++nondegree;
            }
        }
    }
}


__global__ inline void CUDAcomputeneighborslist( const CUDAgraph* g, CUDAneighbors* ns)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    CUDAcomputeneighborslistsingle(g,ns,index);

}


__device__ inline CUDAneighbors* newCUDAneighbors(CUDAgraph& g)
{
    auto ns = new CUDAneighbors;
    ns->g = &g;
    if (ns->g == nullptr) {
        // std::cout << "Can't create a neighbor object with a null graph\n";
        ns->maxdegree=0;
        return ns;
    }
    ns->neighborslist = (CUDAvertextype*)malloc(g.dim * g.dim * sizeof(CUDAvertextype));
    ns->nonneighborslist = (CUDAvertextype*)malloc(g.dim * g.dim * sizeof(CUDAvertextype));
    ns->degrees = (int*)malloc(g.dim * sizeof(int) );
    for (int n = 0; n < g.dim; ++n)
        CUDAcomputeneighborslistsingle(&g,ns,n);
    ns->maxdegree = -1;
    for (int n = 0; n < g.dim; ++n)
        ns->maxdegree = (ns->degrees[n] > ns->maxdegree) ? ns->degrees[n] : ns->maxdegree;
    return ns;
}

inline void hostnewCUDAneighbors(CUDAgraph* d_g, CUDAneighbors* d_ns, int dim )
{
    CUDAneighbors h_ns;
    h_ns.g = d_g;

    // if (h_ns.g == nullptr) {
        // std::cout << "Can't create a neighbor object with a null graph\n";
        // d_ns->maxdegree=0;
    // }
    h_ns.maxdegree = -1;
    cudaMalloc((void**)&h_ns.neighborslist,dim*dim*sizeof(CUDAvertextype));
    cudaMalloc((void**)&h_ns.nonneighborslist,dim*dim*sizeof(CUDAvertextype) );
    cudaMalloc((void**)&h_ns.degrees,dim*sizeof(int) );

    cudaMalloc(&d_ns,sizeof(CUDAneighbors));



    cudaMemcpy(d_ns,&h_ns,sizeof(CUDAneighbors),cudaMemcpyHostToDevice);
    // for (int n = 0; n < g.dim; ++n)
    //     CUDAcomputeneighborslist(g,*d_ns,n);
}

inline void hostdeleteCUDAneighbors(CUDAneighbors* ns)
{
    free(ns->neighborslist);
    free(ns->nonneighborslist);
    free(ns->degrees);
    free(ns);
}


__device__ inline void deleteCUDAneighbors(CUDAneighbors& ns)
{
    free(ns.neighborslist);
    free(ns.nonneighborslist);
    free(ns.degrees);
}

__device__ inline void CUDAcopygraph( const CUDAgraph* g1, CUDAgraph* g2 ) {
    if (g1->dim != g2->dim) {
        // std::cout << "Mismatched dimensions in copygraph\n";
        return;
    }
    for (int i = 0; i < g1->dim * g1->dim; ++i)
        g2->adjacencymatrix[i] = g1->adjacencymatrix[i];
}


__device__ int CUDApathsbetweenmin( const CUDAgraph* g, const CUDAneighbors* ns, CUDAvertextype v1, CUDAvertextype v2, int min );




#endif //CUDAGRAPH_CUH
