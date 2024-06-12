//
// Created by peterglenn on 6/12/24.
//

#ifndef ASYMP_H
#define ASYMP_H
#include <bits/stl_algo.h>

#include "graphs.h"
#include "prob.h"

class measure {
public:
    virtual int takemeasure( graph g, neighbors ns ) {return 0;};
};

class criterion {
public:
    virtual bool checkcriterion( graph g, neighbors ns ) {return false;};
};

class asymp {
public:
    virtual float computeasymptotic( criterion* cr, measure* ms, const int outof, const int dim ) {
        float max = 0.0;
        for (int n = 0; n < dim*(dim-1)/2.0; ++n) {
            graph g;
            g.dim = dim;
            randomconnectedgraphfixededgecnt* rg = new randomconnectedgraphfixededgecnt(n);
            g.adjacencymatrix = (bool*)malloc(g.dim * g.dim * sizeof(bool));
            for (int k = 0; k < outof; ++k) {
                rg->randomgraph(&g);
                neighbors ns;
                ns = computeneighborslist(g);
                if (cr->checkcriterion(g,ns)) {
                    //float tmp = ms->takemeasure(g,ns);
                    //max = (tmp > max ? tmp : max);
                    max = n;
                }
            }
            delete rg;
        }
        return max;
    }
};


class trianglefreecriterion : public criterion {
public:
    bool checkcriterion(graph g, neighbors ns) override {
        for (int n = 0; n < g.dim-2; ++n) {
            for (int i = n+1; i < g.dim-1; ++i) {
                for (int k = i+1; k < g.dim; ++k) {
                    if (g.adjacencymatrix[n*g.dim + k]
                        && g.adjacencymatrix[i*g.dim + k]
                        && g.adjacencymatrix[n*g.dim + i])
                        return false;
                }
            }
        }

        //std::cout << "Triangle free!\n";
        //osadjacencymatrix(std::cout, g);
        //std::cout << "\n";
        return true;
    }
};

class edgecountmeasure : public measure {
public:
    int takemeasure( graph g, neighbors ns ) override {
        int edgecnt = 0;
        for (int n = 0; n < g.dim-1; ++n) {
            for (int i = n+1; i < g.dim; ++i) {
                if (g.adjacencymatrix[n*g.dim + i]) {
                    edgecnt++;
                }
            }
        }
        return edgecnt;
    }
};


#endif //ASYMP_H
