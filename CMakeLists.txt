cmake_minimum_required(VERSION 3.31)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

# Choose to set OFF for non-CUDA compiled code (though nvcc is still required to compile this code)
set (INTERNALFLAGCALC_CUDA ON)

if (INTERNALFLAGCALC_CUDA)
    project(flagcalc CUDA)
    set(CMAKE_CUDA_STANDARD 20)
else()
    project(flagcalc CUDA)
    set(CMAKE_CXX_STANDARD 20)
endif ()

add_executable(flagcalc main.cu
        graphs.cu
        graphs.h
        db.cu
        db.h
        graphio.cu
        graphio.h
        prob.cu
        prob.h
        verboseio.cu
        verboseio.h
        asymp.cu
        asymp.h
        feature.cu
        feature.h
        workspace.cu
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
        config.h
        cudameas.cu
)



if (INTERNALFLAGCALC_CUDA)
    set_target_properties(flagcalc PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(flagcalc PRIVATE FLAGCALC_CMAKE_CUDA)
endif ()
