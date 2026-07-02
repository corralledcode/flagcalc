//
// Created by peterglenn on 5/13/25.
//

// #include "config.h"

#ifdef FLAGCALC_CUDA



#ifdef FLAGCALC_CUDA


#include <device_launch_parameters.h>
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

__device__ inline void CUDAcopygraph( const CUDAgraph* g1, CUDAgraph* g2 )
{
    if (g1->dim != g2->dim) {
        // std::cout << "Mismatched dimensions in copygraph\n";
        return;
    }
    for (int i = 0; i < g1->dim * g1->dim; ++i)
        g2->adjacencymatrix[i] = g1->adjacencymatrix[i];
}

__device__ int CUDApathsbetweenmin( const CUDAgraph* g, const CUDAneighbors* ns, CUDAvertextype v1, CUDAvertextype v2, int min );
__global__ inline void CUDAneighborcount( int* out, bool* adjmatrix, int dim );
__global__ inline void CUDAsquarematrixmultiply( int* out, int* in1, int* in2, const int dim );
__global__ inline void CUDAverticesconnectedmatrix( bool* out, bool* adj, const int dim);



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
            cnt = out[row*dim + i] && adj[i*dim + col];
        out[row*dim + col] = out[row*dim + col] || cnt;
    }
}




// Step 1: Hooking - representative parents of adjacent vertices are linked together (this is the Shiloach-Vishkin algorithm)
__global__ void cc_hook(const int* src, const int* dst, int* parent, int num_edges, bool* d_changed) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_id < num_edges) {
        int u = src[edge_id];
        int v = dst[edge_id];

        int p_u = parent[u];
        int p_v = parent[v];

        // Hook the higher component ID to the lower component ID to avoid cycles
        if (p_u < p_v) {
            atomicMin(&parent[p_v], p_u);
            *d_changed = true;
        } else if (p_v < p_u) {
            atomicMin(&parent[p_u], p_v);
            *d_changed = true;
        }
    }
}

// Step 2: Pointer Jumping - Shortens the paths to the root component representative (this is the Shiloach-Vishkin algorithm)
__global__ void cc_pointer_jump(int* parent, int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v < num_vertices) {
        int p = parent[v];
        // Jump directly to the grandparent
        while (p != parent[p]) {
            p = parent[p];
        }
        parent[v] = p;
    }
}



// Host Driver Function (this is the Shiloach-Vishkin algorithm)
void find_connected_components(const std::vector<int>& h_src, const std::vector<int>& h_dst, int num_vertices,
    std::vector<int>& h_parent) {
    int num_edges = h_src.size();

    // Allocate Device Memory
    int *d_src, *d_dst, *d_parent;
    bool *d_changed;
    cudaMalloc(&d_src, num_edges * sizeof(int));
    cudaMalloc(&d_dst, num_edges * sizeof(int));
    cudaMalloc(&d_parent, num_vertices * sizeof(int));
    cudaMalloc(&d_changed, sizeof(bool));

    // Initialize Parent array where every node is its own parent: parent[i] = i
    // std::vector<int> h_parent(num_vertices);
    h_parent.resize(num_vertices);
    for (int i = 0; i < num_vertices; ++i) h_parent[i] = i;

    // Copy Data to Device
    cudaMemcpy(d_src, h_src.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_parent, h_parent.data(), num_vertices * sizeof(int), cudaMemcpyHostToDevice);

    // Execution Configuration
    int threads_per_block = 256;
    int edge_blocks = (num_edges + threads_per_block - 1) / threads_per_block;
    int vertex_blocks = (num_vertices + threads_per_block - 1) / threads_per_block;

    bool h_changed = true;

    // Main Iteration Loop
    while (h_changed) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        // Run Hooking Kernel over all edges
        cc_hook<<<edge_blocks, threads_per_block>>>(d_src, d_dst, d_parent, num_edges, d_changed);
        cudaDeviceSynchronize();

        // Run Pointer Jumping Kernel over all vertices
        cc_pointer_jump<<<vertex_blocks, threads_per_block>>>(d_parent, num_vertices);
        cudaDeviceSynchronize();

        // Check if any parents were modified during this iteration
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // Retrieve Final Component IDs
    cudaMemcpy(h_parent.data(), d_parent, num_vertices * sizeof(int), cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_src); cudaFree(d_dst); cudaFree(d_parent); cudaFree(d_changed);
}







#endif // ENABLE_CUDA

#endif // FLAGCALC_CUDA