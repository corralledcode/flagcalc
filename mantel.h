//
// Created by peterglenn on 6/25/24.
//

#ifndef MANTEL_H
#define MANTEL_H
#include "asymp.h"
#include "graphoutcome.h"


class asymp {
public:
    virtual float computeasymptotic( abstractcriterion<bool>* cr, abstractmeasure<float>* ms, const int outof, const int dim, std::ostream& os, workspace* ws ) {
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

#endif //MANTEL_H
