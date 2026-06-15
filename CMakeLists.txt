cmake_minimum_required(VERSION 3.31)
project(flagcalc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

option(ENABLE_CUDA "Enable CUDA acceleration" ON) # handled in CMake under settings

if (ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

    set(CMAKE_CUDA_STANDARD 20)
    add_compile_definitions(FLAGCALC_CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})

    add_library(
            cuda.cu
            cuda.cuh
            cudafn.cu
            cudaengine.cu
            cudaengine.cuh
            cudagraph.cu
            cudameas.cu
    )


endif ()


# Choose to set OFF for non-CUDA compiled code (though nvcc is still required to compile this code)
# set (INTERNALFLAGCALC_CUDA ON)
set (ENABLE_PYTHON ON)

# if (INTERNALFLAGCALC_CUDA)
#    project(flagcalc LANGUAGES CXX CUDA)
# else()
#    project(flagcalc LANGUAGES CXX CUDA)
#    set(CMAKE_CXX_STANDARD 20)
# endif ()

find_package(Boost CONFIG REQUIRED COMPONENTS regex)

 add_executable(flagcalc
         main.cpp
        graphs.cpp
        graphs.h
        graphio.cpp
        graphio.h
        prob.cpp
        prob.h
        verboseio.cpp
        verboseio.h
         asymp.h
        feature.cpp
        feature.h
        workspace.cpp
        workspace.h
        thread_pool.cpp
        thread_safe_queue.cpp
        graphoutcome.cpp
        graphoutcome.h
        mantel.h
        math.cpp
        mathfn.h
        mathfn.cpp
        absmeas.h
         math.h
         ameas.h
        meas.cpp
        probsub.cpp
        probsub.h
        #        cuda.cu
        #        cuda.cuh
        #        cudafn.cu
        #        cudaengine.cu
        #        cudaengine.cuh
        #        cudagraph.cu
        #        cudagraph.cuh
        #        cudameas.cu
)

if (ENABLE_CUDA)
    set_source_files_properties(
        main.cpp
        graphs.cpp
        graphs.h
        graphio.cpp
        graphio.h
        prob.cpp
        prob.h
        verboseio.cpp
        verboseio.h
        asymp.h
        feature.cpp
        feature.h
        workspace.cpp
        workspace.h
        thread_pool.cpp
        thread_safe_queue.cpp
        graphoutcome.cpp
        graphoutcome.h
        mantel.h
        math.cpp
        # mathfn.h
        mathfn.cpp
        absmeas.h
        math.h
        ameas.h
        meas.cpp
        probsub.cpp
        probsub.h
                 cuda.cu
                 cuda.cuh
                 cudafn.cu
                 cudaengine.cu
                 cudaengine.cuh
                 cudagraph.cu
                 cudagraph.cuh
                 cudameas.cu
        PROPERTIES LANGUAGE CUDA
)
endif ()

if (ENABLE_PYTHON)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    include_directories(${Python3_INCLUDE_DIRS})
    target_link_libraries(flagcalc PRIVATE ${Python3_LIBRARIES})
    target_compile_definitions(flagcalc PRIVATE FLAGCALC_CMAKE_PYTHON)
    add_compile_definitions(FLAGCALCWITHPYTHON)
endif ()

add_compile_definitions("LONGINT=long int")

target_link_libraries(flagcalc PRIVATE ${Boost_LIBRARIES})



#if (INTERNALFLAGCALC_CUDA)
#    set_target_properties(flagcalc PROPERTIES
#            CUDA_SEPARABLE_COMPILATION ON)
#    target_compile_definitions(flagcalc PRIVATE FLAGCALC_CMAKE_CUDA)
# endif ()

set_property(TARGET flagcalc PROPERTY CXX_STANDARD 20)

if (ENABLE_CUDA)
    set_target_properties(flagcalc PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(flagcalc PRIVATE FLAGCALC_CMAKE_CUDA)
endif ()
