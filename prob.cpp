//
// Created by peterglenn on 6/10/24.
//
// many of the functions in this file prob.cpp have been depcrecated and
// replaced by the corresponging class methods in prob.h
// note still not sure how to define class methods
// in a separate file such as here in prob.cpp...

#include "prob.h"

#include <functional>
#include <future>
#include <random>
#include <iostream>
#include <stdbool.h>

//#define THREADPOOL6


int samplematchingrandomgraphs( abstractparameterizedrandomgraph* rg, const int dim, const float edgecnt, const int outof ) { // returns the count of how many pairs share a fingerprint
    int cnt = 0;
    auto g5 = new graphtype(dim);
    auto g6 = new graphtype(dim);

    rg->setparams({std::to_string(dim),std::to_string(edgecnt),std::to_string(outof)});

    for (int i = 0; i < outof; ++i) {
        rg->randomgraph(g5);
        //osadjacencymatrix(std::cout,g5);
        //std::cout << "\n";
        rg->randomgraph(g6);
        //osadjacencymatrix(std::cout,g6);
        //std::cout << "\n\n";

        auto ns5 = new neighbors(g5);
        //osneighbors(std::cout,ns5);

        auto ns6 = new neighbors(g6);
        //osneighbors(std::cout,ns5);

        FP fps5[dim];
        for (vertextype n = 0; n < dim; ++n) {
            fps5[n].v = n;
            fps5[n].ns = nullptr;
            fps5[n].nscnt = 0;
            fps5[n].parent = nullptr;
        }

        takefingerprint(ns5,fps5,dim);

        //osfingerprint(std::cout,ns5,fps5,g5.dim);

        FP fps6[g6->dim];
        for (vertextype n = 0; n < dim; ++n) {
            fps6[n].v = n;
            fps6[n].ns = nullptr;
            fps6[n].nscnt = 0;
            fps6[n].parent = nullptr;
        }

        takefingerprint(ns6,fps6,dim);

        /*
        FP fpstmp5;
        fpstmp5.parent = nullptr;
        fpstmp5.ns = fps5;
        fpstmp5.nscnt = dim;

        FP fpstmp6;
        fpstmp6.parent = nullptr;
        fpstmp6.ns = fps6;
        fpstmp6.nscnt = dim;
*/
        //osfingerprint(std::cout,ns6,fps6,g6.dim);
        //if (FPcmp(ns5,ns6,&fpstmp5,&fpstmp6) == 0) {

        if (FPcmp(ns5,ns6,fps5,fps6) == 0) {
            //std::cout << "Fingerprints MATCH\n";
            cnt++;
        } else {
            //std::cout << "Fingerprints DO NOT MATCH\n";
        }
        freefps(fps5, dim);
        freefps(fps6, dim);
        delete ns5;
        delete ns6;
    }
    //verboseio vio;
    //verbosedbio vdbio(getenv("DBSERVER"), getenv("DBUSR"), getenv("DBPWD"), getenv("DBSCHEMA"));
    //vio = vdbio;
    //vio.output("Random probability of fingerprints matching is " + std::to_string(cnt) + " out of " + std::to_string(outof) + " == " + std::to_string(float(cnt)/float(outof)) + "\n");
    // the above four lines are commented out until MySQL C++ Connector is up and working (i.e. in files verboseio.h/cpp)
    return cnt;

}


/* PRE-threading variant
std::vector<graph> randomgraphs( abstractrandomgraph* rg, const int dim, const float edgecnt, const int cnt ) {
    std::vector<graph> gv {};
    gv.resize(cnt);
    for (int i = 0; i < cnt; ++i) {
        gv[i].dim = dim;
        gv[i].adjacencymatrix = (bool*)malloc(dim * dim * sizeof(bool));
    }

    for (int i = 0; i < cnt; ++i) {
        rg->randomgraph(&(gv[i]),edgecnt);
        //osadjacencymatrix(std::cout,g5);
        //std::cout << "\n";
        //osneighbors(std::cout,ns5);
    }
    return gv;
}
*/



std::vector<graphtype*> randomgraphs( abstractparameterizedrandomgraph* rg, const int dim, const float edgecnt, const int cnt ) {
/*    std::vector<graph> gv {};
    gv.resize(cnt);

    for (int i = 0; i < cnt; ++i) {
        gv[i].dim = dim;
        gv[i].adjacencymatrix = (bool*)malloc(dim * dim * sizeof(bool));
    }
    for (int i = 0; i < cnt; ++i) {
        rg->randomgraph(&(gv[i]),edgecnt);
        //osadjacencymatrix(std::cout,g5);
        //std::cout << "\n";
        //osneighbors(std::cout,ns5);
    }

    return gv;*/
    rg->setparams({std::to_string(dim),std::to_string(edgecnt),std::to_string(cnt)});

    return rg->randomgraphs(cnt);  //<-- an attempt to use the multithreaded, but it is no faster than the above
}


/*
void randomgraph( graph* gptr, const float edgecnt ) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,10000);
    for (int n = 0; n < gptr->dim; ++n) {
        gptr->adjacencymatrix[n*gptr->dim + n] = 0;
        for (int i = n+1; i < gptr->dim; ++i) {
            gptr->adjacencymatrix[n*gptr->dim + i] = (dist10000(rng) < (10000*float(edgecnt)/(gptr->dim * (gptr->dim-1)/2.0)));
            gptr->adjacencymatrix[i*gptr->dim + n] = gptr->adjacencymatrix[n*gptr->dim+i];
        }
    }
}

void randomconnectedgraphfixededgecnt( graph* gptr, const int edgecnt ) {
    for (int i = 0; i < gptr->dim; ++i) {
        for (int j = 0; j < gptr->dim; ++j) {
            gptr->adjacencymatrix[gptr->dim*i + j] = false;
        }
    }
    if (edgecnt == 0)
        return;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,9999);
    std::vector<vertextype> edges {};
    vertextype v1 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/10000.0);
    vertextype v2 = v1;
    while (v2 == v1) {
        v2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/10000.0);
    }
    bool visited[gptr->dim];
    for (vertextype i = 0; i < (gptr->dim); ++i) {
        visited[i]=false;
    }
    visited[v1] = true;
    visited[v2] = true;
    int visitedcnt = 2;
    gptr->adjacencymatrix[v1*(gptr->dim) + v2] = true;
    gptr->adjacencymatrix[v2*(gptr->dim) + v1] = true;
    int cnt = edgecnt-1;
    while (cnt > 0) {
        vertextype candidatev1index = (vertextype)(float((dist10000(rng)) * float(visitedcnt))/10000.0);
        int cnt2 = 0;
        vertextype i = -1;
        while (cnt2 <= candidatev1index) {
            ++i;
            if (visited[i]) {
                cnt2++;
            }
        }
        vertextype candidatev1 = i;
        //std::cout << "candidatev1 == i == " << i << "\n";
        vertextype candidatev2 = candidatev1;
        while (candidatev2 == candidatev1) {
            candidatev2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/10000.0);
        }
        if ((candidatev2 >= gptr->dim) || (candidatev1 >= gptr->dim))
            std::cout << "ERROR\n";
        if (!visited[candidatev2]) {
            visitedcnt++;
            visited[candidatev2] = true;
        }
        if (!(gptr->adjacencymatrix[(candidatev1*gptr->dim) + candidatev2])) {
            gptr->adjacencymatrix[candidatev1*(gptr->dim) + candidatev2] = true;
            gptr->adjacencymatrix[candidatev2*(gptr->dim) + candidatev1] = true;
            cnt--;
        }
    }


}


/*
void randomconnectedgraph( graph* gptr ) {
    for (int i = 0; i < gptr->dim; ++i) {
        for (int j = 0; j < gptr->dim; ++j) {
            gptr->adjacencymatrix[gptr->dim*i + j] = false;
        }
    }
    if (gptr->dim == 0)
        return;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,9999);
    std::vector<vertextype> edges {};
    vertextype v1 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/10000.0);
    vertextype v2 = v1;
    while (v2 == v1) {
        v2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/10000.0);
    }
    bool visited[gptr->dim];
    for (vertextype i = 0; i < (gptr->dim); ++i) {
        visited[i]=false;
    }
    visited[v1] = true;
    visited[v2] = true;
    int visitedcnt = 2;
    gptr->adjacencymatrix[v1*(gptr->dim) + v2] = true;
    gptr->adjacencymatrix[v2*(gptr->dim) + v1] = true;
    while (visitedcnt<gptr->dim) {
        vertextype candidatev1index = (vertextype)(float((dist10000(rng)) * float(visitedcnt))/10000.0);
        int cnt2 = 0;
        vertextype i = -1;
        while (cnt2 <= candidatev1index) {
            ++i;
            if (visited[i]) {
                cnt2++;
            }
        }
        vertextype candidatev1 = i;
        //std::cout << "candidatev1 == i == " << i << "\n";
        vertextype candidatev2 = candidatev1;
        while (candidatev2 == candidatev1) {
            candidatev2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/10000.0);
        }
        if ((candidatev2 >= gptr->dim) || (candidatev1 >= gptr->dim))
            std::cout << "ERROR\n";
        if (!visited[candidatev2]) {
            visitedcnt++;
            visited[candidatev2] = true;
        }
        if (!(gptr->adjacencymatrix[(candidatev1*gptr->dim) + candidatev2])) {
            gptr->adjacencymatrix[candidatev1*(gptr->dim) + candidatev2] = true;
            gptr->adjacencymatrix[candidatev2*(gptr->dim) + candidatev1] = true;
        }
    }


}
*/