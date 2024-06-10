//
// Created by peterglenn on 6/10/24.
//

#include "prob.h"
#include <random>
#include <iostream>

void randomgraph( graph* gptr, int edgecnt ) {
    for (int n = 0; n < gptr->dim; ++n) {
        gptr->adjacencymatrix[n*gptr->dim + n] = 0;
        for (int i = n+1; i < gptr->dim; ++i) {
            std::random_device dev;
            std::mt19937 rng(dev());
            std::uniform_int_distribution<std::mt19937::result_type> dist1000(0,1000);
            gptr->adjacencymatrix[n*gptr->dim + i] = (dist1000(rng) < (1000*float(edgecnt)/(gptr->dim * (gptr->dim-1)/2.0)));
            gptr->adjacencymatrix[i*gptr->dim + n] = gptr->adjacencymatrix[n*gptr->dim+i];
        }
    }
}
