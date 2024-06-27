//
// Created by peterglenn on 6/12/24.
//

#ifndef ASYMP_H
#define ASYMP_H
//#include <bits/stl_algo.h>

//choose one of the two #defines
//#define THREADED3
#define NOTTHREADED3

#ifdef THREADED3
#include <thread>
#endif


#include <future>
#include <thread>

#include "graphs.h"
#include "prob.h"
#include "workspace.h"

class abstractmeasure {
public:
    virtual int takemeasure( const graphtype* g, const neighbors* ns ) {return 0;}
};

template<typename T>
class abstractcriterion {
protected:
    virtual T internalcheckcriterion( const graphtype* g, const neighbors* ns ) {}
public:
    const bool negated;
    std::string name = "_abstractcriterion internal use";
    virtual std::string shortname() {return "_ac";}

    T checkcriterion( const graphtype* g, const neighbors* ns ) {
        return negated != internalcheckcriterion(g,ns);
    }
    abstractcriterion( bool negatedin, std::string namein ) :negated{negatedin}, name{namein} {}
};

inline int threadcomputeasymp( randomconnectedgraphfixededgecnt* rg, abstractcriterion<bool>* cr, graphtype* g,
    int n, const int outof, int sampled, bool** samplegraph )
{
    int max = 0;
    //int sampled = 0;
    while( sampled < outof )
    {
        sampled++;
        rg->randomgraph(g,n);
        //auto ns = new neighbors(g);
        //ns->computeneighborslist(g); ns isn't used by criterion...
        if (cr->checkcriterion(g,nullptr)) {
            //float tmp = ms->takemeasure(g,ns);
            //max = (tmp > max ? tmp : max);
            max = n;
            //bool* tmpadjacencymatrix = (bool*)malloc(g.dim * g.dim * sizeof(bool));
            *samplegraph = (bool*)malloc(g->dim * g->dim * sizeof(bool));
            for (int i = 0; i < g->dim; ++i) {
                for (int j = 0; j < g->dim; ++j) {
                    (*samplegraph)[g->dim*i + j] = g->adjacencymatrix[g->dim*i+j];
                }
            }
            std::cout << "1\n";
            return sampled;
        }
    }
    //free(ns.neighborslist); // nothing to free
    *samplegraph = nullptr;
    return sampled;
}




class trianglefreecriterion : public abstractcriterion<bool> {
protected:
    bool internalcheckcriterion( const graphtype* g, const neighbors* ns) override {
        int dim = g->dim;
        for (int n = 0; n < dim-2; ++n) {
            for (int i = n+1; i < dim-1; ++i) {
                if (g->adjacencymatrix[n*dim + i]) {
                    for (int k = i+1; k < dim; ++k) {
                        if (g->adjacencymatrix[n*dim + k]
                            && g->adjacencymatrix[i*dim + k])
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


public:
    std::string shortname() override {return negated ? "cr2" : "cr1";}
    trianglefreecriterion(bool negatedin) : abstractcriterion(negatedin,negatedin ? "triangle-free criterion (not)" : "triangle-free criterion") {}
};


class embedscriterion : public abstractcriterion<bool> {
protected:
    bool internalcheckcriterion( const graphtype* g, const neighbors* ns) override {
        return (embeds(flagns, fp, ns));
    }

public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    std::string shortname() override {return "cr2";}
    embedscriterion(neighbors* flagnsin,FP* fpin,bool negatedin) : abstractcriterion(negatedin, negatedin ? "embeds flag criterion (not)" : "embeds flag criterion"), flagg{flagnsin->g},flagns{flagnsin},fp{fpin} {}
};


class edgecountmeasure : public abstractmeasure {
public:
    int takemeasure( const graphtype* g, const neighbors* ns ) override {
        int edgecnt = 0;
        for (int n = 0; n < g->dim-1; ++n) {
            for (int i = n+1; i < g->dim; ++i) {
                if (g->adjacencymatrix[n*g->dim + i]) {
                    edgecnt++;
                }
            }
        }
        return edgecnt;
    }
};


#endif //ASYMP_H
