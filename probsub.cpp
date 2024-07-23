//
// Created by peterglenn on 7/22/24.
//

#include "probsub.h"
#include <future>
#include <iostream>
#include <random>
#include <iostream>
#include <string>
#include <cstring>

#include "asymp.h"
#include "graphs.h"

#define RANDOMRANGE 10000
#define RANDOMRANGEdouble 10000.0


class abstractsubrandomgraph {
public:
    virtual std::string shortname() {return "_arg";};
    std::string name;

    virtual void randomgraph( graphtype* gptr, graphtype* parentg, std::vector<int>* subg ) {}
    virtual std::vector<graphtype*> randomgraphs( const int dim, graphtype* parentg, std::vector<int>* subg, const int cnt ) { return {};}
    abstractsubrandomgraph(const std::string namein) : name{namein} {}
};

class abstractparameterizedsubrandomgraph : public abstractsubrandomgraph {
public:
    std::vector<std::string> ps;
    virtual void setparams( std::vector<std::string> psin ) {
        ps = psin;
    }
    abstractparameterizedsubrandomgraph(const std::string namein) : abstractsubrandomgraph(namein) {}
};

class legacyabstractsubrandomgraph : public abstractsubrandomgraph {
public:
    std::vector<graphtype*> subrandomgraphsinternal(const int dim, graphtype* parentg, std::vector<int>* subg, const int cnt) {
        std::vector<graphtype*> res {};
        res.resize(cnt);
        for (int i = 0; i < cnt; ++i) {
            graphtype* rg = new graphtype(dim);
            randomgraph(rg,parentg,subg);;
            res[i] = rg;
        }
        return res;
    }


    virtual void randomgraph( graphtype* gptr, graphtype* parentg, std::vector<int>* subg ) {};
    /* The rules for the above virtual function: it must not modify the object's variables
     * lest it create a contention when threaded. Use local variables instead,
     * e.g. such as _edgecnt; just use the local version of it. */

    std::vector<graphtype*> randomgraphs( const int dim, graphtype* parentg, std::vector<int>* subg, const int cnt ) {
        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        double section = double(cnt)/double(thread_count);
        // Note: MUST play safe in the abstractrandomgraph class so it won't contend

        std::vector<std::future<std::vector<graphtype*>>> t {};
        t.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            const int startidx = int(j*section);
            const int stopidx = int((j+1.0)*section);
            t[j] = std::async(&legacyabstractsubrandomgraph::subrandomgraphsinternal,this,dim,parentg,subg,stopidx-startidx);
        }
        std::vector<std::vector<graphtype*>> gvv {};
        gvv.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            gvv[j] = t[j].get();
        }
        std::vector<graphtype*> res {};
        res.resize(cnt);
        //std::cout << (thread_count-1)*section + int(section) << "<-- highest index\n";
        for (int j = 0; j < thread_count; ++j) {
            //std::cout<<"section " << section << " , j*section  ="<< j*section << "\n";
            for (int i = 0; i < gvv[j].size(); ++i) {
                res[int(j*section + i)] = gvv[j][i];
                //res.push_back(gvv[j][i]);
            }
        }
        return res;
    }

    legacyabstractsubrandomgraph(std::string namein) : abstractsubrandomgraph(namein) {}
};

class legacystdrandomsubgraph : public legacyabstractsubrandomgraph {
    double _edgecnt;
public:
    std::string shortname() {return "rs1";};
    legacystdrandomsubgraph() : legacyabstractsubrandomgraph("random graph of dim=p[0], and p[1] is how many") {}
    virtual void randomgraph( graphtype* gptr, graphtype* parentg, std::vector<int>* subg ) override {
        legacyabstractsubrandomgraph::randomgraph(gptr, parentg, subg);
        //name = "random graph with edgecnt probability " + std::to_string(_edgecnt);
        //_edgecnt = edgecnt;

        int dim = gptr->dim;
        int pdim = parentg->dim;
        zerograph(gptr);
        // bool* subgadjacencymatrix = (bool*)malloc(pdim*pdim*sizeof(bool));
        // memset( subgadjacencymatrix,0,pdim*pdim*sizeof(bool));
        bool* subgbool = (bool*)malloc(pdim*sizeof(bool));
        memset( subgbool,0,pdim*sizeof(bool));
        // for (auto a  : *subg)
            // for (auto b : *subg)
            // {
                // subgadjacencymatrix[a*pdim + b] = true;
            // }
        for (int i = 0; i < subg->size(); ++i)
            subgbool[(*subg)[i]] = true;

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);

        int d = subg->size();
        if (pdim <= subg->size())
        {
            // delete subgadjacencymatrix;
            delete subgbool;
            return;
        }
        int rvertex = -1;
        while (d < dim)
        {
            while (rvertex < 0 || (rvertex >= 0 && subgbool[rvertex]))
                rvertex = (int)((dist10000(rng) * (double)(pdim))/RANDOMRANGE);
            subgbool[rvertex] = true;
            ++d;
        }
        int r = 0;
        int s;
        int i = 0;
        while (i < pdim)
        {
            while (i < pdim && !subgbool[i])
                ++i;
            if (i >= pdim)
                break;
            int j = i+1;
            s = r+1;
            while (j < pdim)
            {
                while (j < pdim && !subgbool[j])
                    ++j;
                if (j >= pdim)
                    break;
                bool b = parentg->adjacencymatrix[i*pdim + j];
                gptr->adjacencymatrix[r*dim + s] = b;
                gptr->adjacencymatrix[s*dim + r] = b;
                ++s;
                ++j;
            }
            ++r;
            ++i;
        }
        // delete subgadjacencymatrix;
        delete subgbool;
    }
};


template<typename T>
class legacyrandomsubgraph : public abstractparameterizedsubrandomgraph {
protected:
    T* legacyrandomsubgraphptr {};
public:
    std::string shortname() {return legacyrandomsubgraphptr->shortname();}

    void randomgraph(graphtype* gptr,graphtype* parentg, std::vector<int>* subg ) override {
        legacyrandomsubgraphptr->randomgraph(gptr,parentg, subg);
    }
    std::vector<graphtype*> randomgraphs(const int dim,graphtype* parentg, std::vector<int>* subg, const int cnt ) override {
        // if (ps.size()>0)
            // dim = stoi(ps[0]);
        // if (ps.size()>1)
            // edgecnt = stof(ps[1]);
        return legacyrandomsubgraphptr->randomgraphs(dim,parentg,subg,cnt);
    }

    legacyrandomsubgraph() : legacyrandomsubgraphptr{new T}, abstractparameterizedsubrandomgraph("legacy random graph") {
        name = legacyrandomsubgraphptr->name;
    }
    ~legacyrandomsubgraph() {
        delete legacyrandomsubgraphptr;
    }
};
