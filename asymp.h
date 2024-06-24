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
    virtual int takemeasure( const graphtype* g, const neighbors* ns ) {return 0;};
};

template<typename T>
class abstractcriterion {
public:
    std::string name = "_abstractcriterion internal use";
    virtual std::string shortname() {return "_ac";}

    virtual T checkcriterion( const graphtype* g, const neighbors* ns ) {return {};};
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


class asymp {
public:
    virtual float computeasymptotic( abstractcriterion<bool>* cr, abstractmeasure* ms, const int outof, const int dim, std::ostream& os, workspace* ws ) {
        int max = 0;
        int sampled = 0;
        int n = 1;
        std::vector<bool*> samplegraphs;
        auto rg = new randomconnectedgraphfixededgecnt();
        auto g = new graphtype(dim);
#ifdef THREADED3

        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        std::vector<graph> gv {};
        gv.resize(thread_count);
        for (int n = 0; n < thread_count; ++n)
        {
            gv[n].dim = dim;
            gv[n].adjacencymatrix = (bool*)malloc(gv[n].dim * gv[n].dim * sizeof(bool));
        }
        std::vector<bool*> samplegraph {};
        samplegraph.resize(thread_count);
        while (sampled < outof)
        {
            std::vector<std::future<int>> t {};
            t.resize(thread_count);
            int threadoutof = outof/thread_count;
            for (int m = 0; m < thread_count; ++m) {
                t[m] = std::async(&threadcomputeasymp,rg,cr,gv[m],n,outof,sampled,&(samplegraph[m]));
            }
            std::vector<int> threadsampled {};
            threadsampled.resize(thread_count);
            for (int m = 0; m < thread_count; ++m) {
                //t[m].join();
                //t[m].detach();
                threadsampled[m] = t[m].get();
            }
            bool found = false;
            for (int m = 0; m < thread_count; ++m)
            {
                if (threadsampled[m] < outof)
                {
                    if (samplegraph[m] != nullptr)
                    {
                        found = true;
                        samplegraphs.push_back(samplegraph[m]);
                    }
                }
                sampled += threadsampled[m];
                std::cout << "sampled " << sampled << ", threadsampled " << threadsampled[m] << " samplegraph == nullptr" << (samplegraph[m]==nullptr) <<"\n";
            }
            if (found)
            {
                ++n;
            }
            max = n-1;
        }

#endif


#ifndef THREADED3
        while (sampled < outof) {
            while (max < n && sampled < outof) {
                rg->randomgraph(g,n);
                //auto ns = new neighbors(g);
                //ns = computeneighborslist(g); ns isn't used by criterion...
                if (cr->checkcriterion(g,nullptr)) {
                    //float tmp = ms->takemeasure(g,ns);
                    //max = (tmp > max ? tmp : max);
                    max = n;
                    bool* tmpadjacencymatrix = (bool*)malloc(g->dim * g->dim * sizeof(bool));
                    for (int i = 0; i < dim; ++i) {
                        for (int j = 0; j < dim; ++j) {
                            tmpadjacencymatrix[g->dim*i + j] = g->adjacencymatrix[g->dim*i+j];
                        }
                    }
                    samplegraphs.push_back(tmpadjacencymatrix);
                    n++;
                }
                sampled++;
            }
            //free(ns.neighborslist); // nothing to free
        }

#endif

        //free(g.adjacencymatrix);

/*        for (int i = 0; i < samplegraphs.size()-1; ++i) {
            //std::cout << "Size n == " << i << ":\n";
            //osadjacencymatrix(os, tmpg);
        }*/

        graphitem* gi = new graphitem();
        gi->g = g;
        gi->g->adjacencymatrix = samplegraphs[samplegraphs.size()-1];
        gi->name = ws->getuniquename(gi->classname);
        gi->ns = new neighbors(g);
        //gi->ns->computeneighborslist();
        ws->items.push_back(gi);
        for (int i = 0; i < (samplegraphs.size()-1); ++i) {
            free(samplegraphs[i]);
        }
        delete rg;
        return max;
    }
};


class trianglefreecriterion : public abstractcriterion<bool> {
public:
    std::string shortname() override {return "cr1";}
    trianglefreecriterion() : abstractcriterion() {
        name = "triangle-free criterion";
    }
    bool checkcriterion( const graphtype* g, const neighbors* ns) override {
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
};


class embedscriterion : public abstractcriterion<bool> {
public:
    graphtype* flagg;
    neighbors* flagns;
    std::string shortname() override {return "cr2";}
    embedscriterion(neighbors* flagns) : abstractcriterion() {
        name = "embeds flag criterion";
        this->flagg = flagns->g;
        this->flagns = flagns;
    }
    bool checkcriterion( const graphtype* g, const neighbors* ns) override {
        return (embeds(flagns, ns));
    }
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
