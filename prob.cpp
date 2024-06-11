//
// Created by peterglenn on 6/10/24.
//

#include "prob.h"
#include <random>
#include <iostream>
#include <stdbool.h>

void randomgraph( graph* gptr, const int edgecnt ) {
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
