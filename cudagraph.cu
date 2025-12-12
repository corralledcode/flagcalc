//
// Created by peterglenn on 5/13/25.
//

#include "config.h"

#ifdef FLAGCALC_CUDA

#include "cudagraph.cuh"



__global__ inline void CUDAneighborcount( int* out, bool* adjmatrix, int dim )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < dim)
    {
        out[index] = 0;
        for (int j = index+1; j < dim; ++j)
            out[index] += adjmatrix[index * dim + j];
    }
}


__global__ inline void CUDAsquarematrixmultiply( int* out, int* in1, int* in2, const int dim )
{
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    // int col = blockIdx.x * blockDim.x + threadIdx.x;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int row = index / dim;
    int col = index % dim;

    int sum = 0;

    if (row < dim && col < dim)
    {
        for (int i = 0; i < dim; ++i)
            sum += in1[row * dim + i] * in2[i * dim + col];
        out[row * dim + col] = sum;
    }
}


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

__global__ inline void CUDAverticesconnectedmatrix( bool* out, bool* adj, const int dim)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    CUDAvertextype row = index / dim;
    CUDAvertextype col = index % dim;

    if (index < dim*dim)
    {
        int cnt = 0;
        for (int i = 0; i < dim && !cnt; ++i)
            cnt += out[row*dim + i] && adj[i*dim + col];
        out[row*dim + col] = out[row*dim + col] || cnt;
    }
}

#endif // FLAGCALC_CUDA