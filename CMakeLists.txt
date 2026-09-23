cmake_minimum_required(VERSION 3.31)
project(flagcalc LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)

option(ENABLE_CUDA "Enable CUDA acceleration" ON) # handled in CMake under settings
option(ENABLE_PYTHON "Enable Python" ON) # handled in CMake under settings

find_package(scs REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

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

# Force Windows to link against the release Python library in Debug mode
if(MSVC AND WIN32)
    add_compile_definitions(Py_NO_LINK_LIB)
    # Alternatively, you can trick pyconfig.h by bypassing the _DEBUG check for Python:
    # add_compile_definitions(MS_NO_COREDLL)
endif()

# Force CMake to map the debug configuration to the release library
if(WIN32)
    set(Python3_FIND_ABI "OFF" "ANY" "ANY")
    # If using older CMake FindPython syntax:
    # target_link_libraries(your_target_name optimized ${Python3_LIBRARIES})
endif()

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
        sdp.h
        #        cuda.cu
        #        cuda.cuh
        #        cudafn.cu
        #        cudaengine.cu
        #        cudaengine.cuh
        #        cudagraph.cu
        #        cudagraph.cuh
        #        cudameas.cu
)

if (MSVC)
    target_compile_options(flagcalc PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/bigobj>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>
    )
    add_compile_definitions(MSVC)
endif ()


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
               ost  cudameas.cu
        PROPERTIES LANGUAGE CUDA
)
endif ()

if (ENABLE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development)
    find_package(Python COMPONENTS Interpreter Development)
    include_directories(${Python3_INCLUDE_DIRS})
    target_link_libraries(flagcalc PRIVATE ${Python3_LIBRARIES})
    target_compile_definitions(flagcalc PRIVATE FLAGCALC_CMAKE_PYTHON)
    add_compile_definitions(FLAGCALCWITHPYTHON)
endif ()

# to find SCS semi-definite programming headers
target_include_directories(flagcalc PRIVATE "/usr/local/lib")
target_include_directories(flagcalc PRIVATE "/usr/local/include")
target_link_libraries(flagcalc PRIVATE scs::scsdir)

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
