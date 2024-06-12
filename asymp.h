//
// Created by peterglenn on 6/12/24.
//

#ifndef ASYMP_H
#define ASYMP_H
#include <bits/stl_algo.h>

#include "graphs.h"
#include "prob.h"
#include "workspace.h"

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
    virtual float computeasymptotic( criterion* cr, measure* ms, const int outof, const int dim, std::ostream& os, workspace* ws ) {
        int max = 0;
        int sampled = 0;
        int n = 1;
        std::vector<bool*> samplegraphs;
        randomconnectedgraphfixededgecnt* rg = new randomconnectedgraphfixededgecnt(n);
        graph g;
        g.dim = dim;
        g.adjacencymatrix = (bool*)malloc(g.dim * g.dim * sizeof(bool));
        while (sampled < outof) {
            while (max < n && sampled < outof) {
                rg->randomgraph(&g);
                neighbors ns;
                //ns = computeneighborslist(g); ns isn't used by criterion...
                if (cr->checkcriterion(g,ns)) {
                    //float tmp = ms->takemeasure(g,ns);
                    //max = (tmp > max ? tmp : max);
                    max = n;
                    bool* tmpadjacencymatrix = (bool*)malloc(g.dim * g.dim * sizeof(bool));
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            tmpadjacencymatrix[g.dim*i + j] = g.adjacencymatrix[g.dim*i+j];
                        }
                    }
                    samplegraphs.push_back(tmpadjacencymatrix);
                    n++;
                    delete rg;
                    rg = new randomconnectedgraphfixededgecnt(n);
                }
                sampled++;
            }
            //free(ns.neighborslist);
        }
        free(g.adjacencymatrix);
        graph tmpg;
        tmpg.dim = dim;
        for (int i = 0; i < samplegraphs.size(); ++i) {
            tmpg.adjacencymatrix = samplegraphs[i];
            std::cout << "Size n == " << i << ":\n";
            osadjacencymatrix(os, tmpg);
        }
        graphitem* gi = new graphitem();
        gi->g.dim = dim;
        gi->g.adjacencymatrix = samplegraphs[samplegraphs.size()-1];
        gi->name = ws->getuniquename();
        ws->items.push_back(gi);
        for (int i = 0; i < samplegraphs.size()-1; ++i) {
            free(samplegraphs[i]);
        }
        return max;
    }
};


class trianglefreecriterion : public criterion {
public:
    bool checkcriterion(graph g, neighbors ns) override {
        for (int n = 0; n < g.dim-2; ++n) {
            for (int i = n+1; i < g.dim-1; ++i) {
                if (g.adjacencymatrix[n*g.dim + i]) {
                    for (int k = i+1; k < g.dim; ++k) {
                        if (g.adjacencymatrix[n*g.dim + k]
                            && g.adjacencymatrix[i*g.dim + k])
                            return false;
                    }
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
