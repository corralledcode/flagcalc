//
// Created by peterglenn on 5/12/25.
//

#include "config.h"

#ifdef FLAGCALC_CUDA

#include "cudaengine.cuh"
#include "cudafn.cu"
// #include "cudagraph.cuh"

#include <cuda_runtime.h>


__global__ void wrapCUDAeval( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const CUDAfcptr start, const uint sz )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sz)
    {
        crit[index] = true;
        out[index] = CUDAevalinternal(Cecs[index],start);
    }
}

__global__ void wrapCUDAevalcriterion( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const CUDAfcptr start, const uint sz )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < sz)
    {
        crit[index] = CUDAto_mtbool(CUDAevalinternal(Cecs[index],Cecs[index].CUDAfcarray[start].criterion));
        if (crit[index] == true)
            out[index] = CUDAevalinternal(Cecs[index],start);
    }
}

__global__ void wrapCUDAevalfast( bool* crit, CUDAvalms* out, CUDAextendedcontext& Cec,
    const CUDAfcptr start, const uint dimm, const uint sz )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    CUDAvdimn v;
    int i = 0;
    int r = index;
    while (i < GPUQUANTFASTDIM)
    {
        v[i] = r % dimm;
        r /= dimm;
        ++i;
    }
    if (index < sz)
    {
        CUDAextendedcontext Cecplus = Cec;
        for (int j = 0; j < Cecplus.numfastn; ++j)
            Cecplus.fastn[j] = v[j];
        crit[index] = true;
        out[index] = CUDAevalinternal(Cecplus,start);
    }
}

__global__ void wrapCUDAevalcriterionfast( bool* crit, CUDAvalms* out, CUDAextendedcontext& Cec,
    const CUDAfcptr start, const uint dimm, const uint sz )
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    CUDAvdimn v;
    int i = 0;
    int r = index;
    while (i < GPUQUANTFASTDIM)
    {
        v[i] = r % dimm;
        r /= dimm;
        ++i;
    }
    if (index < sz)
    {
        CUDAextendedcontext Cecplus = Cec;
        for (int j = 0; j < Cecplus.numfastn; ++j)
            Cecplus.fastn[j] = v[j];
        crit[index] = CUDAto_mtbool(CUDAevalinternal(Cecplus,Cec.CUDAfcarray[start].criterion));
        if (crit[index] == true)
            out[index] = CUDAevalinternal(Cecplus,start);
    }
}


__global__ void CUDAcomputeneighborslistenmasse( const CUDAgraph* gs, CUDAneighbors* ns, const long int sz)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < sz)
    {
        for (int i = 0; i < gs[index].dim; ++i)
            CUDAcomputeneighborslistsingle(&gs[index],&ns[index],i);
    }
}


void CUDAevalwithcriterionfast( bool* crit, CUDAvalms* out, CUDAextendedcontext& Cec, const CUDAfcptr start,
    const uint dimm, const uint sz )
{
#ifdef CUDADEBUG2
    auto starttime = std::chrono::high_resolution_clock::now();
#endif

    CUDAextendedcontext h_Cec = Cec;
    CUDAextendedcontext* d_Cec;

    cudaMalloc((void**)&h_Cec.CUDAfcarray, Cec.CUDAfcarraysize*sizeof(CUDAfc));
    cudaMalloc((void**)&h_Cec.namedvararray, Cec.namedvararraysize*sizeof(CUDAnamedvariable));
    cudaMalloc((void**)&h_Cec.CUDAvalsarray, Cec.CUDAvalsarraysize);
    cudaMalloc((void**)&h_Cec.CUDAcontext, Cec.CUDAcontextsize*sizeof(CUDAnamedvariable));
    cudaMalloc((void**)&h_Cec.CUDAliteralarray, Cec.CUDAliteralarraysize*sizeof(CUDAliteral));
    cudaMalloc((void**)&h_Cec.fastn,sizeof(CUDAvdimn));
    cudaMalloc((void**)&h_Cec.g.adjacencymatrix, Cec.g.dim*Cec.g.dim*sizeof(bool));
    cudaMalloc((void**)&h_Cec.ns.neighborslist, Cec.g.dim*Cec.g.dim*sizeof(vertextype));
    cudaMalloc((void**)&h_Cec.ns.nonneighborslist, Cec.g.dim*Cec.g.dim*sizeof(vertextype));
    cudaMalloc((void**)&h_Cec.ns.degrees, Cec.g.dim*sizeof(int));

    cudaMemcpy(h_Cec.CUDAfcarray, Cec.CUDAfcarray,Cec.CUDAfcarraysize*sizeof(CUDAfc),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.namedvararray, Cec.namedvararray,Cec.namedvararraysize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.CUDAvalsarray, Cec.CUDAvalsarray,Cec.CUDAvalsarraysize,cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.CUDAcontext, Cec.CUDAcontext,Cec.CUDAcontextsize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.CUDAliteralarray, Cec.CUDAliteralarray,Cec.CUDAliteralarraysize*sizeof(CUDAliteral),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.g.adjacencymatrix, Cec.g.adjacencymatrix,Cec.g.dim*Cec.g.dim*sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.ns.neighborslist, Cec.ns.neighborslist,Cec.g.dim*Cec.g.dim*sizeof(CUDAvertextype),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.ns.nonneighborslist, Cec.ns.nonneighborslist,Cec.g.dim*Cec.g.dim*sizeof(CUDAvertextype),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.ns.degrees, Cec.ns.degrees,Cec.g.dim*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_Cec,sizeof(CUDAextendedcontext));

    cudaMemcpy(d_Cec, &h_Cec,sizeof(CUDAextendedcontext),cudaMemcpyHostToDevice);

    CUDAvalms *d_out;
    bool *d_crit;

    cudaMalloc((void**)&d_out, sz*sizeof(CUDAvalms));
    cudaMalloc((void**)&d_crit, sz*sizeof(bool));

    int blockSize = 1024;
    int numBlocks = (sz + blockSize - 1) / blockSize;


#ifdef CUDADEBUG2
    auto starttime2 = std::chrono::high_resolution_clock::now();
#endif

    size_t pValue;
    cudaDeviceGetLimit(&pValue,cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize,2048);
    cudaDeviceGetLimit(&pValue,cudaLimitStackSize);

#ifdef CUDADEBUG2

    std::cout << "cudaDeviceGetLimit(cudaLimitStackSize) == " << pValue << std::endl;

#endif

    populateCUDAfnarraysingle<<<numBlocks,blockSize>>>(d_Cec);

    if (sz > 0)
        if (Cec.CUDAfcarray[start].criterion >= 0)
            wrapCUDAevalcriterionfast<<<numBlocks,blockSize>>>(d_crit, d_out, *d_Cec, start, dimm, sz);
        else
            wrapCUDAevalfast<<<numBlocks,blockSize>>>(d_crit, d_out, *d_Cec, start, dimm, sz);

    cudaDeviceSynchronize();
#ifdef CUDADEBUG2

    auto stoptime2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stoptime2 - starttime2);

    std::cout << "CUDA runtime excluding cudaMalloc and cudaMemcpy: " << duration2.count() << " microseconds" << std::endl;
#endif
    cudaMemcpy( crit, d_crit, sz * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy( out, d_out, sz * sizeof(CUDAvalms), cudaMemcpyDeviceToHost);

    cudaFree(h_Cec.CUDAfcarray);
    cudaFree(h_Cec.namedvararray);
    cudaFree(h_Cec.CUDAvalsarray);
    cudaFree(h_Cec.CUDAcontext);
    cudaFree(h_Cec.CUDAliteralarray);
    cudaFree(h_Cec.fastn);
    cudaFree(h_Cec.g.adjacencymatrix);
    cudaFree(h_Cec.ns.neighborslist);
    cudaFree(h_Cec.ns.nonneighborslist);
    cudaFree(h_Cec.ns.degrees);

    cudaFree(d_Cec);
    cudaFree(d_out);
    cudaFree(d_crit);

#ifdef CUDADEBUG2
    auto stoptime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stoptime - starttime);

    std::cout << "CUDA runtime including cudaMalloc and cudaMemcpy: " << duration.count() << " microseconds" << std::endl;
#endif


#ifdef CUDADEBUG
    for (int i = 0; i < sz; ++i)
        std::cout << "OUT: " << i << ": " << crit[i] << ".. " << out[i].v.bv << "(type " << out[i].t << ")" << "; ";
    std::cout << std::endl;
#endif


}


void CUDAevalwithcriterion( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const CUDAfcptr start, const uint sz )
{
    auto starttime = std::chrono::high_resolution_clock::now();

    auto h_Cecs = new CUDAextendedcontext[sz];
    CUDAextendedcontext h_Cec;
    CUDAextendedcontext* d_Cecs;
    cudaMalloc((void**)&d_Cecs,sz*sizeof(CUDAextendedcontext));

    if (sz == 0)
        return;

    h_Cec = Cecs[0];

    cudaMalloc((void**)&h_Cec.CUDAfcarray,Cecs[0].CUDAfcarraysize*sizeof(CUDAfc));
    cudaMalloc((void**)&h_Cec.CUDAliteralarray, Cecs[0].CUDAliteralarraysize*sizeof(CUDAliteral));
    cudaMalloc((void**)&h_Cec.g.adjacencymatrix, Cecs[0].g.dim*Cecs[0].g.dim*sizeof(bool));
    cudaMalloc((void**)&h_Cec.ns.neighborslist, Cecs[0].g.dim*Cecs[0].g.dim*sizeof(vertextype));
    cudaMalloc((void**)&h_Cec.ns.nonneighborslist, Cecs[0].g.dim*Cecs[0].g.dim*sizeof(vertextype));
    cudaMalloc((void**)&h_Cec.ns.degrees, Cecs[0].g.dim*sizeof(int));

    cudaMemcpy(h_Cec.CUDAfcarray, Cecs[0].CUDAfcarray,Cecs[0].CUDAfcarraysize*sizeof(CUDAfc),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.CUDAliteralarray, Cecs[0].CUDAliteralarray,Cecs[0].CUDAliteralarraysize*sizeof(CUDAliteral),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.g.adjacencymatrix, Cecs[0].g.adjacencymatrix,Cecs[0].g.dim*Cecs[0].g.dim*sizeof(bool),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.ns.neighborslist, Cecs[0].ns.neighborslist,Cecs[0].g.dim*Cecs[0].g.dim*sizeof(vertextype),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.ns.nonneighborslist, Cecs[0].ns.nonneighborslist,Cecs[0].g.dim*Cecs[0].g.dim*sizeof(vertextype),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.ns.degrees, Cecs[0].ns.degrees,Cecs[0].g.dim*sizeof(int),cudaMemcpyHostToDevice);

    for (int i = 0; i < sz; ++i)
    {
        // h_Cecs[i] = Cecs[i];

        h_Cecs[i] = h_Cec;

        cudaMalloc((void**)&h_Cecs[i].namedvararray, Cecs[i].namedvararraysize*sizeof(CUDAnamedvariable));
        cudaMalloc((void**)&h_Cecs[i].CUDAvalsarray,Cecs[i].CUDAvalsarraysize);
        cudaMalloc((void**)&h_Cecs[i].CUDAcontext, Cecs[i].CUDAcontextsize*sizeof(CUDAnamedvariable));

        cudaMemcpy(h_Cecs[i].namedvararray, Cecs[i].namedvararray,Cecs[i].namedvararraysize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);
        cudaMemcpy(h_Cecs[i].CUDAvalsarray, Cecs[i].CUDAvalsarray,Cecs[i].CUDAvalsarraysize,cudaMemcpyHostToDevice);
        cudaMemcpy(h_Cecs[i].CUDAcontext, Cecs[i].CUDAcontext,Cecs[i].CUDAcontextsize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);

        // cudaMemcpy(&h_Cecs[i].CUDAfcarray, h_Cec.CUDAfcarray, sizeof(CUDAfc*), cudaMemcpyHostToDevice);
        // cudaMemcpy(&h_Cecs[i].CUDAliteralarray, h_Cec.CUDAliteralarray, sizeof(CUDAliteral*), cudaMemcpyHostToDevice);
        // cudaMemcpy(&h_Cecs[i].g.adjacencymatrix, h_Cec.g.adjacencymatrix, sizeof(bool*), cudaMemcpyHostToDevice);

#ifdef CUDADEBUG
        for (int j = 0; j < Cecs[i].CUDAfcarraysize; ++j)
        {
            auto tempfc = Cecs[i].CUDAfcarray[j];
            std::cout << "CUDAeval: " << j << " (" << int(tempfc.fo) << "): " << tempfc.criterion << ", " << tempfc.fcleft << ", " << tempfc.fcright << std::endl;
        }
#endif
    }

    cudaMemcpy(d_Cecs, h_Cecs,sz*sizeof(CUDAextendedcontext),cudaMemcpyHostToDevice);

    CUDAvalms *d_out;
    bool *d_crit;

    cudaMalloc((void**)&d_out, sz*sizeof(CUDAvalms));
    cudaMalloc((void**)&d_crit, sz*sizeof(bool));

    int blockSize = 1024;
    int numBlocks = (sz + blockSize - 1) / blockSize;


#ifdef CUDADEBUG2
    auto starttime2 = std::chrono::high_resolution_clock::now();
#endif

    size_t pValue;
    cudaDeviceGetLimit(&pValue,cudaLimitStackSize);
    cudaDeviceSetLimit(cudaLimitStackSize,4096);
    cudaDeviceGetLimit(&pValue,cudaLimitStackSize);

#ifdef CUDADEBUG2

    std::cout << "cudaDeviceGetLimit(cudaLimitStackSize) == " << pValue << std::endl;

#endif


    if (sz > 0)
    {
        populateCUDAfnarray<<<numBlocks,blockSize>>>(d_Cecs,sz);
        if (Cecs[0].CUDAfcarray[start].criterion >= 0)
            wrapCUDAevalcriterion<<<numBlocks,blockSize>>>(d_crit, d_out, d_Cecs, start, sz);
        else
            wrapCUDAeval<<<numBlocks,blockSize>>>(d_crit, d_out, d_Cecs, start, sz);
    }

    cudaDeviceSynchronize();
#ifdef CUDADEBUG2

    auto stoptime2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stoptime2 - starttime2);

    std::cout << "CUDA runtime excluding cudaMalloc and cudaMemcpy: " << duration2.count() << " microseconds" << std::endl;
#endif
    cudaMemcpy( out, d_out, sz * sizeof(CUDAvalms), cudaMemcpyDeviceToHost);
    cudaMemcpy( crit, d_crit, sz * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(h_Cec.CUDAfcarray);
    cudaFree(h_Cec.CUDAliteralarray);
    cudaFree(h_Cec.g.adjacencymatrix);
    cudaFree(h_Cec.ns.neighborslist);
    cudaFree(h_Cec.ns.nonneighborslist);
    cudaFree(h_Cec.ns.degrees);
    for (auto i = 0; i < sz; ++i)
    {
        cudaFree(h_Cecs[i].namedvararray);
        cudaFree(h_Cecs[i].CUDAvalsarray);
        cudaFree(h_Cecs[i].CUDAcontext);
    }
    cudaFree(d_out);
    cudaFree(d_crit);

    // delete d_Cecs;  // segfaults

    delete h_Cecs;

#ifdef CUDADEBUG2
    auto stoptime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stoptime - starttime);

    std::cout << "CUDA runtime including cudaMalloc and cudaMemcpy: " << duration.count() << " microseconds" << std::endl;
#endif


#ifdef CUDADEBUG
    for (int i = 0; i < sz; ++i)
        std::cout << "OUT: " << i << ": " << crit[i] << ".. " << out[i].v.iv << "(type " << out[i].t << ")" << "; ";
    std::cout << std::endl;
#endif
}

void CUDAevalwithcriterion( bool* crit, CUDAvalms* out, CUDAextendedcontext* Cecs, const uint sz )
{
    CUDAevalwithcriterion(crit, out, Cecs, Cecs[0].fctop, sz);
}

void CUDAcountpathsbetweenwrapper(int* out, int walklength, const bool* adjmatrix, const int dim )
{
    int* d_out;
    int* d_in1;
    int* d_in2;

    int in1[dim * dim];
    int in2[dim * dim];

    memset(in1,0,sizeof(int) * dim * dim);
    for (int i = 0; i < dim; ++i)
        in1[i*dim+i] = 1;
    for (int i = 0; i < dim*dim; ++i)
        in2[i] = adjmatrix[i];

#ifdef CUDADEBUG2

    auto starttime = std::chrono::high_resolution_clock::now();
#endif

    cudaMalloc(&d_out,dim*dim*sizeof(int));
    cudaMalloc(&d_in1,dim*dim*sizeof(int));
    cudaMalloc(&d_in2,dim*dim*sizeof(int));
    cudaMemcpy(d_in1,in1,dim*dim*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2,in2,dim*dim*sizeof(int),cudaMemcpyHostToDevice);

    // Define the grid and block dimensions
    int blockSize = dim;
    int numBlocks = dim; // (blockSize + dim + 1)/blockSize;

    // dim3 blockSize = (dim,dim);
    // dim3 numBlocks  = ((dim+blockSize.x - 1)/blockSize.x, (dim+blockSize.y - 1)/blockSize.y);

#ifdef CUDADEBUG2

    auto starttime2 = std::chrono::high_resolution_clock::now();

#endif

    // Launch the kernel
    for (int i = 0; i < walklength; ++i)
    {
        CUDAsquarematrixmultiply<<<numBlocks, blockSize>>>(d_out, d_in1, d_in2, dim);
        cudaMemcpy(d_in1,d_out,dim*dim*sizeof(int),cudaMemcpyDeviceToDevice);
    }

    cudaDeviceSynchronize();

#ifdef CUDADEBUG2
    auto stoptime2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stoptime2 - starttime2);

    std::cout << "CUDA runtime excluding cudaMalloc and cudaMemcpy: " << duration2.count() << " microseconds" << std::endl;
#endif

    // Copy the result back to the host
    cudaMemcpy(out, d_out, dim * dim * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    cudaFree(d_in1);
    cudaFree(d_in2);

#ifdef CUDADEBUG2
    auto stoptime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stoptime - starttime);

    std::cout << "CUDA runtime including cudaMalloc and cudaMemcpy: " << duration.count() << " microseconds" << std::endl;
#endif

};


void CUDAcomputeneighborslistwrapper( graphtype* g, neighborstype* ns )
{

    if (g->dim == 0)
        return;

    CUDAgraph* d_g;
    CUDAgraph h_g;
    h_g.dim = g->dim;

    CUDAneighbors h_ns;

    cudaMalloc((void**)&h_g.adjacencymatrix,g->dim*g->dim*sizeof(bool));
    cudaMalloc((void**)&h_ns.neighborslist,g->dim*g->dim*sizeof(CUDAvertextype));
    cudaMalloc((void**)&h_ns.nonneighborslist,g->dim*g->dim*sizeof(CUDAvertextype));
    cudaMalloc((void**)&h_ns.degrees,g->dim*sizeof(int));

    cudaMemcpy(h_g.adjacencymatrix,g->adjacencymatrix,sizeof(bool)*g->dim*g->dim,cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_g, sizeof(CUDAgraph));
    cudaMemcpy(d_g, &h_g,sizeof(CUDAgraph),cudaMemcpyHostToDevice);

    CUDAneighbors* d_ns;

    cudaMalloc((void**)&d_ns,sizeof(CUDAneighbors));
    cudaMemcpy(d_ns,&h_ns,sizeof(CUDAneighbors),cudaMemcpyHostToDevice);

    // hostnewCUDAneighbors(d_g, d_ns, g->dim);

    int blockSize = 1024;
    int numBlocks = (g->dim + blockSize - 1) / blockSize;


#ifdef CUDADEBUG2
    auto starttime2 = std::chrono::high_resolution_clock::now();
#endif

    // size_t pValue;
    // cudaDeviceGetLimit(&pValue,cudaLimitStackSize);
    // cudaDeviceSetLimit(cudaLimitStackSize,2048);
    // cudaDeviceGetLimit(&pValue,cudaLimitStackSize);

#ifdef CUDADEBUG2

    std::cout << "cudaDeviceGetLimit(cudaLimitStackSize) == " << pValue << std::endl;

#endif


    CUDAcomputeneighborslist<<<numBlocks,blockSize>>>(d_g,d_ns);

    cudaDeviceSynchronize();
#ifdef CUDADEBUG2

    auto stoptime2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stoptime2 - starttime2);

    std::cout << "CUDA runtime excluding cudaMalloc and cudaMemcpy: " << duration2.count() << " microseconds" << std::endl;
#endif
    cudaMemcpy(&h_ns,d_ns,sizeof(CUDAneighbors), cudaMemcpyDeviceToHost);
    cudaMemcpy( ns->degrees, h_ns.degrees, g->dim * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy( ns->neighborslist, h_ns.neighborslist, g->dim*g->dim * sizeof(CUDAvertextype), cudaMemcpyDeviceToHost);
    cudaMemcpy( ns->nonneighborslist, h_ns.nonneighborslist, g->dim*g->dim * sizeof(CUDAvertextype), cudaMemcpyDeviceToHost);

    // osadjacencymatrix(std::cout, g);
    // osneighbors(std::cout, ns);

    ns->g = g;
    ns->maxdegree = -1;
    for (int i = 0; i < g->dim; ++i)
        ns->maxdegree = ns->degrees[i] > ns->maxdegree ? ns->degrees[i] : ns->maxdegree;
    // std::cout << std::endl;

    // for (auto l = 0; l < g->dim; ++l)
    // {
        // std::cout << "l == " << l << ": ";
        // for (auto k = 0; k < (g->dim - ns->degrees[l]); ++k)
            // std::cout << "engine " << k << ": " << ns->nonneighborslist[g->dim*l + k] << ".. ";
        // std::cout << std::endl;
    // }

    // for (auto l = 0; l < g->dim; ++l)
    // {
        // std::cout << "l2 == " << l << ": ";
        // for (auto k = 0; k < ns->degrees[l]; ++k)
            // std::cout << "engine " << k << ": " << ns->neighborslist[g->dim*l + k] << ".. ";
        // std::cout << std::endl;
    // }

    cudaFree(&d_ns->neighborslist);
    cudaFree(&d_ns->nonneighborslist);
    cudaFree(&d_ns->degrees);
    cudaFree(&d_g->adjacencymatrix);
    cudaFree(d_ns);
    cudaFree(d_g);


}


void CUDAcomputeneighborslistenmassewrapper( std::vector<graphtype*>& gv, std::vector<neighborstype*>& nsv )
{

    const int sz = gv.size();

    auto h_gs = new CUDAgraph[sz];

    auto h_ns = new CUDAneighbors[sz];

    for (int i = 0; i < sz; ++i)
    {
        int dim = gv[i]->dim;
        h_gs[i].dim = dim;
        cudaMalloc((void**)&h_gs[i].adjacencymatrix,dim*dim*sizeof(bool));
        cudaMalloc((void**)&h_ns[i].neighborslist,dim*dim*sizeof(CUDAvertextype));
        cudaMalloc((void**)&h_ns[i].nonneighborslist,dim*dim*sizeof(CUDAvertextype));
        cudaMalloc((void**)&h_ns[i].degrees,dim*sizeof(int));

        cudaMemcpy(h_gs[i].adjacencymatrix,gv[i]->adjacencymatrix,sizeof(bool)*dim*dim,cudaMemcpyHostToDevice);
    }

    CUDAgraph* d_gs;
    CUDAneighbors* d_ns;

    cudaMalloc((void**)&d_gs, sz*sizeof(CUDAgraph));
    cudaMemcpy(d_gs, h_gs,sz*sizeof(CUDAgraph),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_ns,sz*sizeof(CUDAneighbors));
    cudaMemcpy(d_ns,h_ns,sz*sizeof(CUDAneighbors),cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (sz + blockSize - 1) / blockSize;


#ifdef CUDADEBUG2
    auto starttime2 = std::chrono::high_resolution_clock::now();
#endif

    // size_t pValue;
    // cudaDeviceGetLimit(&pValue,cudaLimitStackSize);
    // cudaDeviceSetLimit(cudaLimitStackSize,2048);
    // cudaDeviceGetLimit(&pValue,cudaLimitStackSize);

#ifdef CUDADEBUG2

    std::cout << "cudaDeviceGetLimit(cudaLimitStackSize) == " << pValue << std::endl;

#endif


    CUDAcomputeneighborslistenmasse<<<numBlocks,blockSize>>>(d_gs,d_ns,sz);

    cudaDeviceSynchronize();
#ifdef CUDADEBUG2

    auto stoptime2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stoptime2 - starttime2);

    std::cout << "CUDA runtime excluding cudaMalloc and cudaMemcpy: " << duration2.count() << " microseconds" << std::endl;
#endif


    cudaMemcpy(h_ns,d_ns,sz*sizeof(CUDAneighbors), cudaMemcpyDeviceToHost);

    for (int i = 0; i < sz; ++i)
    {
        long int dim = gv[i]->dim;

        nsv[i] = new neighbors(gv[i],false); // tell it not to compute the neighborslist

        cudaMemcpy( nsv[i]->degrees, h_ns[i].degrees, dim * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy( nsv[i]->neighborslist, h_ns[i].neighborslist, dim*dim * sizeof(CUDAvertextype), cudaMemcpyDeviceToHost);
        cudaMemcpy( nsv[i]->nonneighborslist, h_ns[i].nonneighborslist, dim*dim * sizeof(CUDAvertextype), cudaMemcpyDeviceToHost);

        nsv[i]->g = gv[i];
        nsv[i]->maxdegree = -1;
        for (int j = 0; j < dim; ++j)
            nsv[i]->maxdegree = nsv[i]->degrees[j] > nsv[i]->maxdegree ? nsv[i]->degrees[j] : nsv[i]->maxdegree;
        // std::cout << nsv[i]->maxdegree;

        // osadjacencymatrix(std::cout, gv[i]);
        // osneighbors(std::cout, nsv[i]);

        cudaFree( &d_ns[i].neighborslist );
        cudaFree( &d_ns[i].nonneighborslist );
        cudaFree( &d_ns[i].degrees );
        cudaFree( &d_gs[i].adjacencymatrix );

    }

    cudaFree( d_ns );

    delete[] h_ns;
    // std::cout << std::endl;

    // for (auto l = 0; l < g->dim; ++l)
    // {
        // std::cout << "l == " << l << ": ";
        // for (auto k = 0; k < (g->dim - ns->degrees[l]); ++k)
            // std::cout << "engine " << k << ": " << ns->nonneighborslist[g->dim*l + k] << ".. ";
        // std::cout << std::endl;
    // }

    // for (auto l = 0; l < g->dim; ++l)
    // {
        // std::cout << "l2 == " << l << ": ";
        // for (auto k = 0; k < ns->degrees[l]; ++k)
            // std::cout << "engine " << k << ": " << ns->neighborslist[g->dim*l + k] << ".. ";
        // std::cout << std::endl;
    // }

}


void CUDAverticesconnectedmatrixwrapper( const graphtype* g, const neighborstype* ns, bool* out )
{

    const int dim = g->dim;
    const int sz = dim*dim;

    bool* d_out;

    cudaMalloc((void**)&d_out,dim*dim*sizeof(bool));

    // for (int i = 0; i < dim; ++i)
    // {
        // cudaMemcpy(&d_partitions[i*sz],g->adjacencymatrix,dim*dim*sizeof(bool),cudaMemcpyHostToDevice);
    // }
    // cudaMemcpy(d_indices,ns->neighborslist,dim*sizeof(int),cudaMemcpyHostToDevice);

    cudaMemcpy(d_out,out,dim*dim*sizeof(bool),cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlocks = (sz + blockSize - 1) / blockSize;

#ifdef CUDADEBUG2
    auto starttime2 = std::chrono::high_resolution_clock::now();
#endif

    // size_t pValue;
    // cudaDeviceGetLimit(&pValue,cudaLimitStackSize);
    // cudaDeviceSetLimit(cudaLimitStackSize,2048);
    // cudaDeviceGetLimit(&pValue,cudaLimitStackSize);

#ifdef CUDADEBUG2

    std::cout << "cudaDeviceGetLimit(cudaLimitStackSize) == " << pValue << std::endl;

#endif

    for (int i = 0; i < ceil(log(dim)/log(2)); ++i)
        CUDAverticesconnectedmatrix<<<numBlocks,blockSize>>>(d_out, d_out, dim);

    cudaDeviceSynchronize();
#ifdef CUDADEBUG2

    auto stoptime2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stoptime2 - starttime2);

    std::cout << "CUDA runtime excluding cudaMalloc and cudaMemcpy: " << duration2.count() << " microseconds" << std::endl;
#endif

    cudaMemcpy(out,d_out,dim*dim*sizeof(bool), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < dim; ++i)
    // {
        // for (int j = 0; j < sz; ++j)
            // std::cout << h_partitions[i*sz + j] << " ";
        // std::cout << std::endl;
    // }

    cudaFree( d_out );

    // delete[] h_ns;



}

#endif // FLAGCALC_CUDA