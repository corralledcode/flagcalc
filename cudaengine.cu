//
// Created by peterglenn on 5/12/25.
//

#include "cuda.cuh"
#include "cudaengine.cuh"
#include "cudafn.cu"

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

void CUDAevalwithcriterionfast( bool* crit, CUDAvalms* out, CUDAextendedcontext& Cec, const CUDAfcptr start,
    const uint dimm, const uint sz )
{
    auto starttime = std::chrono::high_resolution_clock::now();

    CUDAextendedcontext h_Cec = Cec;
    CUDAextendedcontext* d_Cec;

    cudaMalloc((void**)&h_Cec.CUDAfcarray, Cec.CUDAfcarraysize*sizeof(CUDAfc));
    cudaMalloc((void**)&h_Cec.namedvararray, Cec.namedvararraysize*sizeof(CUDAnamedvariable));
    cudaMalloc((void**)&h_Cec.CUDAvalsarray, Cec.CUDAvalsarraysize);
    cudaMalloc((void**)&h_Cec.CUDAcontext, Cec.CUDAcontextsize*sizeof(CUDAnamedvariable));
    cudaMalloc((void**)&h_Cec.CUDAliteralarray, Cec.CUDAliteralarraysize*sizeof(CUDAliteral));
    cudaMalloc((void**)&h_Cec.fastn,sizeof(CUDAvdimn));

    cudaMemcpy(h_Cec.CUDAfcarray, Cec.CUDAfcarray,Cec.CUDAfcarraysize*sizeof(CUDAfc),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.namedvararray, Cec.namedvararray,Cec.namedvararraysize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.CUDAvalsarray, Cec.CUDAvalsarray,Cec.CUDAvalsarraysize,cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.CUDAcontext, Cec.CUDAcontext,Cec.CUDAcontextsize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);
    cudaMemcpy(h_Cec.CUDAliteralarray, Cec.CUDAliteralarray,Cec.CUDAliteralarraysize*sizeof(CUDAliteral),cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_Cec,sizeof(CUDAextendedcontext));

    cudaMemcpy(d_Cec, &h_Cec,sizeof(CUDAextendedcontext),cudaMemcpyHostToDevice);

    CUDAvalms *d_out;
    bool *d_crit;

    cudaMalloc((void**)&d_out, sz*sizeof(CUDAvalms));
    cudaMalloc((void**)&d_crit, sz*sizeof(bool));

    int blockSize = 256;
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

    CUDAextendedcontext h_Cecs[sz];
    CUDAextendedcontext* d_Cecs;
    cudaMalloc((void**)&d_Cecs,sz*sizeof(CUDAextendedcontext));

    for (int i = 0; i < sz; ++i)
    {
        h_Cecs[i] = Cecs[i];

        cudaMalloc((void**)&h_Cecs[i].CUDAfcarray,Cecs[i].CUDAfcarraysize*sizeof(CUDAfc));
        cudaMalloc((void**)&h_Cecs[i].namedvararray, Cecs[i].namedvararraysize*sizeof(CUDAnamedvariable));
        cudaMalloc((void**)&h_Cecs[i].CUDAvalsarray,Cecs[i].CUDAvalsarraysize);
        cudaMalloc((void**)&h_Cecs[i].CUDAcontext, Cecs[i].CUDAcontextsize*sizeof(CUDAnamedvariable));
        cudaMalloc((void**)&h_Cecs[i].CUDAliteralarray, Cecs[i].CUDAliteralarraysize*sizeof(CUDAliteral));
        // cudaMalloc((void**)&h_Cecs[i].fastn,sizeof(CUDAvdimn));

        cudaMemcpy(h_Cecs[i].CUDAfcarray, Cecs[i].CUDAfcarray,Cecs[i].CUDAfcarraysize*sizeof(CUDAfc),cudaMemcpyHostToDevice);
        cudaMemcpy(h_Cecs[i].namedvararray, Cecs[i].namedvararray,Cecs[i].namedvararraysize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);
        cudaMemcpy(h_Cecs[i].CUDAvalsarray, Cecs[i].CUDAvalsarray,Cecs[i].CUDAvalsarraysize,cudaMemcpyHostToDevice);
        cudaMemcpy(h_Cecs[i].CUDAcontext, Cecs[i].CUDAcontext,Cecs[i].CUDAcontextsize*sizeof(CUDAnamedvariable),cudaMemcpyHostToDevice);
        cudaMemcpy(h_Cecs[i].CUDAliteralarray, Cecs[i].CUDAliteralarray,Cecs[i].CUDAliteralarraysize*sizeof(CUDAliteral),cudaMemcpyHostToDevice);
        // cudaMemcpy(h_Cecs[i].fastn, Cecs[i].fastn,sizeof(CUDAvdimn),cudaMemcpyHostToDevice);

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
    cudaDeviceSetLimit(cudaLimitStackSize,2048);
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

    for (auto i = 0; i < sz; ++i)
    {
        cudaFree(h_Cecs[i].CUDAfcarray);
        cudaFree(h_Cecs[i].namedvararray);
        cudaFree(h_Cecs[i].CUDAvalsarray);
        cudaFree(h_Cecs[i].CUDAcontext);
        cudaFree(h_Cecs[i].CUDAliteralarray);
    }
    cudaFree(d_out);
    cudaFree(d_crit);
    cudaFree(h_Cecs);

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

