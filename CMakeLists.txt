cmake_minimum_required(VERSION 3.31)
project(flagcalc CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(flagcalc main.cu
        graphs.cu
        graphs.h
        db.cu
        db.h
        graphio.cu
        graphio.h
        prob.cu
        prob.h
        verboseio.cpp
        verboseio.h
        asymp.cu
        asymp.h
        feature.cu
        feature.h
        workspace.cpp
        workspace.h
        thread_pool.cpp
        thread_safe_queue.cpp
        graphoutcome.cu
        graphoutcome.h
        mantel.h
        math.cu
        mathfn.h
        mathfn.cu
        absmeas.h
        ameas.cu
        math.h
        ameas.h
        meas.cu
        probsub.cu
        probsub.h
        cuda.cu
        cuda.cuh
        cudafn.cu
        cudaengine.cu
        cudaengine.cuh
        cudagraph.cu
        cudagraph.cuh
)


set_target_properties(flagcalc PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
