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


#include <algorithm>
#include <future>
#include <signal.h>
#include <thread>

#include "graphs.h"
#include "prob.h"
//#include "measure.cpp"
//#include "workspace.h"
#include "math.cpp"

#define KNMAXCLIQUESIZE 12

class girthmeasure;

template<typename T>
class abstractmeasure {
protected:
public:
    int sz = 0;
    std::vector<graphtype*>* gptrs {};
    std::vector<neighbors*>* nsptrs {};
    std::string name = "_abstractcriterion";
    virtual std::string shortname() {return "_ac";}

    virtual void setsize(const int szin) {
        sz = szin;
    }
    virtual T takemeasure( const graphtype* g, const neighbors* ns ) {return {};}
    virtual T takemeasureidxed(const int idx ) {
        return takemeasure((*gptrs)[idx],(*nsptrs)[idx]);
    }


    explicit abstractmeasure( const std::string namein ) :name{namein} {}
    virtual ~abstractmeasure() = default;
};



template<typename T>
class abstractmemorymeasure : public abstractmeasure<T>{
protected:
public:
    T* res = nullptr;
    bool* computed = nullptr;
    //std::string name = "_abstractmemorymeasure";

    void setsize(const int szin) override {
        abstractmeasure<T>::setsize(szin);
        if (res != nullptr)
            delete res;
        if (computed != nullptr)
            delete computed;
        res = (T*)(malloc(szin*sizeof(T)));
        computed = (bool*)(malloc(szin*sizeof(bool)));
        for (int n = 0; n < szin; ++n)
            computed[n] = false;
    }

    std::string shortname() override {return "_amc";}

    T takemeasureidxed( const int idx ) override
    {
        if (idx >= abstractmeasure<T>::sz)
            std::cout << "Error in size of abstractmemorymeasure\n";
        if (!computed[idx]) {
            res[idx] = this->takemeasure((*this->gptrs)[idx],(*this->nsptrs)[idx]);
            computed[idx] = true;
        }
        return res[idx];
    }

    void takemeasurethreadsection( const int startidx, const int stopidx ) {
        for (int i = startidx; i < stopidx; ++i)
            this->takemeasureidxed(i);
    }
    void takemeasurethreadsectionportion( const int startidx, const int stopidx, std::vector<bool>* todo ) {
        for (int i = startidx; i < stopidx; ++i)
            if ((*todo)[i])
                this->takemeasureidxed(i);
    }


    explicit abstractmemorymeasure( std::string namein ) : abstractmeasure<T>(namein) {}
    ~abstractmemorymeasure() {
        delete res;
        delete computed;
    }
};


class truecriterion : public abstractmemorymeasure<bool>{
protected:
public:
    std::string name = "truecriterion";

    virtual std::string shortname() {return "truec";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        //std::cout << "returning TRUE\n";
        return true;
    }
//    bool takemeasureidxed( const int idx ) override {
//        res[idx] = true;
//        computed[idx] = true;
//        return true;
//    }
    truecriterion() : abstractmemorymeasure<bool>("always true") {}
};

/*
inline int threadcomputeasymp( abstractparameterizedrandomgraph* rg, abstractmeasure<bool>* cr, graphtype* g,
    int n, const int outof, int sampled, bool** samplegraph )
{
    int max = 0;
    //int sampled = 0;
    rg->setparams({std::to_string(n)})
    while( sampled < outof )
    {
        sampled++;
        rg->randomgraph(g);
        //auto ns = new neighbors(g);
        //ns->computeneighborslist(g); ns isn't used by criterion...
        if (cr->takemeasure(g,nullptr)) {
            //double tmp = ms->takemeasure(g,ns);
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

*/


class trianglefreecriterion : public abstractmemorymeasure<bool> {
public:
    bool takemeasure( const graphtype* g, const neighbors* ns) override {
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

    std::string shortname() override {return "cr1";}
    trianglefreecriterion() : abstractmemorymeasure("triangle-free criterion") {}
};

class kncriterion : public abstractmemorymeasure<bool> {
public:
    const int n;
    bool takemeasure( const graphtype* g, const neighbors* ns, const int mincnt = 1)  {
        if (n <= 0)
            return true;
        int dim = g->dim;


        std::vector<int> subsets {};
        enumsizedsubsets(0,n,nullptr,0,dim,&subsets);

        bool found = false;
        int foundcnt = 0;
        int j = 0;
        while ((foundcnt < mincnt) && (j < (subsets.size()/n))) {
            found = true;
            for (auto i = 0; found && (i < n-1); ++i) {
                for (auto k = i+1; found && (k < n); ++k)
                    found = found && g->adjacencymatrix[dim*subsets[j*n + i] + subsets[j*n + k]];
            }
            foundcnt += (found ? 1 : 0);
            ++j;
        }
        return foundcnt >= mincnt;
    }


    std::string shortname() override {return "k" + std::to_string(n);}
    kncriterion( const int nin) : abstractmemorymeasure("embeds K_" + std::to_string(nin) + " criterion"), n{nin} {}
};


template<typename T>
class abstractmemoryparameterizedmeasure : public abstractmemorymeasure<T> {
public:
    std::vector<std::string> ps {};

    virtual void setparams( const std::vector<std::string> pin ) {
        ps = pin;
    }

    abstractmemoryparameterizedmeasure( std::string namein ) : abstractmemorymeasure<T>(namein) {}
};



class knparameterizedcriterion : public abstractmemoryparameterizedmeasure<bool> {
protected:
    std::vector<kncriterion*> kns {};
public:
    std::string shortname() override {return "Knc";}
    void populatekns() {
        kns.resize(KNMAXCLIQUESIZE);
        for (int i = 0; i < KNMAXCLIQUESIZE; ++i) {
            auto kn = new kncriterion(i);
            kns[i] = kn;
        }
    }
    bool takemeasure( const graphtype* g, const neighbors* ns ) override {

        if (ps.size() >= 1 && is_number(ps[0])) {
            int sz = stoi(ps[0]);
            int mincnt = 1;
            if (ps.size() == 2 && is_number(ps[1]))
                mincnt = stoi(ps[1]);
            if (0<=sz && sz <= KNMAXCLIQUESIZE)
                return kns[stoi(ps[0])]->takemeasure(g,ns,mincnt);
            else {
                std::cout<< "Increase KNMAXCLIQUESIZE compiler define (current value " + std::to_string(KNMAXCLIQUESIZE) + ")";
                return false;
            }
        }
        // recode to prepare kn's in advance, and perhaps a faster algorithm than using FP
    }

    knparameterizedcriterion() : abstractmemoryparameterizedmeasure<bool>("Parameterized K_n criterion (parameter is complete set size)") {
        populatekns();
    }
    ~knparameterizedcriterion() {
        for (auto kn : kns)
            delete kn;
    }
};

class embedscriterion : public abstractmemorymeasure<bool> {
protected:

public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    std::string shortname() override {return "embedsc";}
    embedscriterion(neighbors* flagnsin,FP* fpin) : abstractmemorymeasure("embeds flag criterion"), flagg{flagnsin->g},flagns{flagnsin},fp{fpin} {}
    bool takemeasure( const graphtype* g, const neighbors* ns) override {
        return (embedsquick(flagns, fp, ns, 1));
    }

};



class legacyembedscriterion : public abstractmemorymeasure<bool> {
protected:

public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    std::string shortname() override {return "lembedsc";}
    legacyembedscriterion(neighbors* flagnsin,FP* fpin) : abstractmemorymeasure("legacy embeds flag criterion"), flagg{flagnsin->g},flagns{flagnsin},fp{fpin} {}
    bool takemeasure( const graphtype* g, const neighbors* ns) override {
        return (embeds(flagns, fp, ns, 1));
    }

};


class sentenceofcriteria : public abstractmemorymeasure<bool> {
protected:
    //std::vector<abstractmemorymeasure<bool>*> cs;
    logicalsentence ls;
    std::vector<bool*> variables {};
    const int sz2;

public:

    std::string shortname() override {return "crSENTENCE";}
    bool takemeasureidxed( const int idx ) override {
        if (!computed[idx]) {
            std::vector<bool> tmpres {};
            tmpres.resize(variables.size());
            for (int i = 0; i < variables.size(); ++i )
                tmpres[i] = variables[i][idx];
            res[idx] = evalsentence(ls,tmpres);  // check for speed cost
            computed[idx] = true;
        }
        return res[idx]; //abstractmemorymeasure::takemeasureidxed(idx);
    }

    sentenceofcriteria( std::vector<bool*> variablesin, const int szin, std::string sentence, std::string stringin )
        : abstractmemorymeasure<bool>(stringin == "" ? "logical sentence of several criteria" : stringin),
            variables{variablesin}, ls{parsesentence(sentence)}, sz2{szin} {
        setsize(sz2);
    }

};

class andcriteria : public sentenceofcriteria {
public:
    std::string shortname() override {return "crAND";}

    andcriteria( std::vector<bool*> variablesin, const int szin )
            : sentenceofcriteria(variablesin,szin,"","logical AND of several criteria")
    {
        std::string s {};

        if (variables.size() > 0) {
            s = "0";
            for (int i = 1; i < variables.size(); ++i) {
                s += " AND " + std::to_string(i);
            }
            ls = parsesentence(s);
        } else
            ls = {};
    }

};

class orcriteria : public sentenceofcriteria {
public:
    std::string shortname() override {return "crOR";}

    orcriteria( std::vector<bool*> variablesin, const int szin )
            : sentenceofcriteria(variablesin,szin,"","logical OR of several criteria")
    {
        std::string s {};

        if (variables.size() > 0) {
            s = "0";
            for (int i = 1; i < variables.size(); ++i) {
                s += " OR " + std::to_string(i);
            }
            ls = parsesentence(s);
        } else
            ls = {};
    }

};


class notcriteria : public abstractmemorymeasure<bool> {
protected:
    const int nsz;
    std::vector<bool*> variables {};
public:
    std::string shortname() override {return "crNOT";}
    std::vector<bool> neg {};
    std::vector<bool*> resnot {};
    notcriteria(std::vector<bool*> variablesin, const int szin, std::vector<bool> negin)
        : abstractmemorymeasure<bool>("logical NOT of several criteria"), variables{variablesin}, neg{negin}, nsz{szin}
    {
        setsize(nsz);
        resnot.resize(variables.size());
        for (int i = 0; i < variables.size(); ++i)
            resnot[i] = (bool*)malloc(nsz*sizeof(bool));
    }

    ~notcriteria()
    {
        for (int i = 0; i < resnot.size(); ++i)
        {
            free(resnot[i]);
        }
    }

    bool takemeasureidxed(const int idx) override
    {
        for (int i = 0; i < resnot.size(); ++i)
        {
            resnot[i][idx] = (variables[i][idx]) != neg[i];
        }
        // technically the return value should be a vector
        if (!resnot.empty()) {
            bool found = false;
            int i = 0;
            while(!found && (i < neg.size())) {
                found = neg[i] == true;
                ++i;
            }
            if (found && (i > 0))
            {
                res[idx] = resnot[i-1][idx];
                computed[idx] = true;
                return res[idx];;
            }
            else
            {
                computed[idx] = true;
                res[idx] = resnot[0][idx];
                return res[idx];
            }
        } else
            return true;
    }

};


template<typename T,typename M> abstractmemorymeasure<M>* factory(void) {
    return new T;
}


class forestcriterion : public abstractmemorymeasure<bool>
{
public:
    virtual std::string shortname() {return "forestc";}

    forestcriterion() : abstractmemorymeasure<bool>("Forest criterion") {}

    bool takemeasure( const graphtype* g, const neighbors* ns ) override
    {
        int dim = g->dim;
        if (dim <= 0)
            return true;
        int* visited = (int*)malloc(dim*sizeof(int));

        for (int i = 0; i < dim; ++i)
        {
            visited[i] = -1;
        }

        visited[0] = 0;
        bool allvisited = false;
        while (!allvisited)
        {
            bool changed = false;
            for ( int i = 0; i < dim; ++i)
            {
                if (visited[i] >= 0)
                {
                    for (int j = 0; j < ns->degrees[i]; ++j)
                    {
                        vertextype nextv = ns->neighborslist[i*dim+j];
                        // loop = if a neighbor of vertex i is found in visited
                        // and that neighbor is not the origin of vertex i
                        bool loop = false;
                        if (visited[nextv] >= 0)
                            if (visited[nextv] != i)
                                if (visited[i] != nextv)
                                    loop = true;

                        if (loop)
                        {
                            free (visited);
                            return false;
                        }
                        if (visited[nextv] < 0)
                        {
                            visited[nextv] = i;
                            changed = true;
                        }
                    }
                }
            }
            if (!changed) {
                allvisited = true;
                int firstunvisited = 0;
                while( allvisited && (firstunvisited < dim))
                {
                    allvisited &= (visited[firstunvisited] != -1);
                    ++firstunvisited;
                }
                if (allvisited)
                {
                    free (visited);
                    return true;
                }
                visited[firstunvisited-1] = firstunvisited-1;
                changed = true;
            }

        }

    }


};

class treecriterion : public abstractmemorymeasure<bool>
{
public:
    virtual std::string shortname() {return "treec";}

    treecriterion() : abstractmemorymeasure<bool>("Tree criterion") {}

    bool takemeasure( const graphtype* g, const neighbors* ns ) override
    {
        int dim = g->dim;
        if (dim <= 0)
            return true;
        int* visited = (int*)malloc(dim*sizeof(int));

        for (int i = 0; i < dim; ++i)
        {
            visited[i] = -1;
        }

        visited[0] = 0;
        bool allvisited = false;
        while (!allvisited)
        {
            bool changed = false;
            for ( int i = 0; i < dim; ++i)
            {
                if (visited[i] >= 0)
                {
                    for (int j = 0; j < ns->degrees[i]; ++j)
                    {
                        vertextype nextv = ns->neighborslist[i*dim+j];
                        // loop = if a neighbor of vertex i is found in visited
                        // and that neighbor is not the origin of vertex i
                        bool loop = false;
                        if (visited[nextv] >= 0)
                            if (visited[nextv] != i)
                                if (visited[i] != nextv)
                                    loop = true;

                        if (loop)
                        {
                            free (visited);
                            return false;
                        }
                        if (visited[nextv] < 0)
                        {
                            visited[nextv] = i;
                            changed = true;
                        }
                    }
                }
            }
            if (!changed) {
                allvisited = true;
                int firstunvisited = 0;
                while( allvisited && (firstunvisited < dim))
                {
                    allvisited &= (visited[firstunvisited] != -1);
                    ++firstunvisited;
                }
                if (allvisited)
                {
                    free (visited);
                    return true;
                } else {
                    free (visited);
                    return false;
                }
            }

        }

    }


};



class equationcriteria : public abstractmemorymeasure<bool> {
protected:
    //std::vector<abstractmemorymeasure<bool>*> cs;
    formulaclass* lhs;
    formulaclass* rhs;
    int eqtype = 0; // 0: equal; 1: >= ; -1: <=; 2: >; -2: <
    std::vector<double*> variables {};
    const int sz2;

public:

    std::string shortname() override {return "crEQUATION";}
    bool takemeasureidxed( const int idx ) override {
        if (!computed[idx]) {
            std::vector<double> tmpres {};
            tmpres.resize(variables.size());
            for (int i = 0; i < variables.size(); ++i )
                tmpres[i] = variables[i][idx];
            switch (eqtype) {
                case 0: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) == evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case 1: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) >= evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case 2: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) >  evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case -1: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) <= evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case -2: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) < evalformula(*rhs,tmpres,nullptr);
                    break;
                }
            }
            computed[idx] = true;
        }
        return res[idx]; //abstractmemorymeasure::takemeasureidxed(idx);
    }

    equationcriteria( std::vector<double*> variablesin, const int szin, std::string equationin, std::string stringin )
        : abstractmemorymeasure<bool>(stringin == "" ? "equation" : stringin),
            variables{variablesin}, sz2{szin} {
        setsize(sz2);
        std::string lhstr;
        std::string rhstr;
        parseequation(&equationin,&lhstr,&rhstr,&eqtype);
        lhs = parseformula(lhstr,nullptr);
        rhs = parseformula(rhstr,nullptr);
    }

    ~equationcriteria() {
        delete lhs;
        delete rhs;
    }

};











#endif //ASYMP_H
