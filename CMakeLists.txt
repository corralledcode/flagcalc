cmake_minimum_required(VERSION 3.28)
project(flagcalc)

set(CMAKE_CXX_STANDARD 23)
include_directories(~/mysql-connector-cpp/include
        ~/mysql-connector-cpp/include/mysqlx)



add_executable(flagcalc main.cpp
        graphs.cpp
        graphs.h
        db.cpp
        db.h
        graphio.cpp
        graphio.h
        prob.cpp
        prob.h
        verboseio.cpp
        verboseio.h
        asymp.cpp
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
        mathfn.cpp
        absmeas.h
        ameas.cpp
        mathfn.h
        math.h
        ameas.h
        meas.cpp
        probsub.cpp
        probsub.h
        mathfn.cpp
)
