//
// Created by peterglenn on 5/13/25.
//

#include "cudagraph.cuh"

__device__ int CUDApathsbetweenmin( const CUDAgraph* g, const CUDAneighbors* ns, CUDAvertextype v1, CUDAvertextype v2, int min )
{
    if (v1 == v2 || min <= 0)
        return 1;
    auto g2 = new CUDAgraph;
    g2->dim = g->dim;
    g2->adjacencymatrix = (bool*)malloc(g->dim * g->dim * sizeof(bool));
    // bool res = false;
    int res = 0;
    for (int i = 0; i < ns->degrees[v1]; ++i)
    {
        CUDAcopygraph( g, g2 );
        CUDAvertextype v = ns->neighborslist[v1*g->dim+i];
        for (int j = 0; j < ns->degrees[v1]; ++j)
        {
            CUDAvertextype v3 = ns->neighborslist[v1*g->dim+j];
            g2->adjacencymatrix[v1*g->dim+v3] = false;
            g2->adjacencymatrix[v3*g->dim+v1] = false;
        }
        auto ns2 = newCUDAneighbors(*g2);
        res += CUDApathsbetweenmin(g2,ns2,v,v2,min);
        deleteCUDAneighbors(*ns2);
        if (res >= min)
            break;
    }
    delete g2->adjacencymatrix;
    delete g2;
    return res;
}

__device__ inline int CUDApathsbetweencount( const CUDAgraph* g, const CUDAneighbors* ns, const CUDAvertextype v1, const CUDAvertextype v2 )
{
    // CUDApathsbetweenmin(g,ns,v1,v2,-1);
    return -1;
}
