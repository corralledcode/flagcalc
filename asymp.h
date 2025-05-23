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
#include <map>

#include "graphs.h"
#include "prob.h"
//#include "measure.cpp"
//#include "workspace.h"
#include "math.h"

class girthmeasure;

class criterion;
class measure;



struct abstms
{
    bool type;
    criterion* cs;
    measure* ms;
};

struct iteration
{
    measuretype mt;
    abstms a;
    int round;
    bool hidden;
};




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

    std::vector<double> ps {};
    int pssz = 0;

    std::map<std::vector<double>,std::pair<bool*, T*>> pres {};

    virtual void setparams( const std::vector<double>& pin ) {
        ps = pin;
    }

    virtual T pstakemeasureidxed( const int idx, const std::vector<double>& psin)
    {
        setparams(psin);
        return takemeasureidxed(idx);
    }

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
        pres.clear();
    }

    std::string shortname() override {return "_amc";}

    T takemeasureidxed( const int idx ) override
    {
        if (idx >= abstractmeasure<T>::sz)
            std::cout << "Error in size of abstractmemorymeasure\n";
        if (pssz > 0 && ps.size()==pssz)
        {
            if ( auto search = pres.find(ps); search != pres.end())
            {
                if (search->second.first[idx])
                    return search->second.second[idx];
                else
                {
                    search->second.second[idx] = this->takemeasure((*this->gptrs)[idx],(*this->nsptrs)[idx]);
                    search->second.first[idx] = true;
                    res[idx] = search->second.second[idx];
                    return search->second.second[idx];
                }
            } else
            {
                pres[ps].first = (bool*)malloc(this->sz*sizeof(bool));
                for (int n = 0; n < this->sz; ++n)
                    pres[ps].first[n] = false;
                pres[ps].second = (T*)malloc(this->sz*sizeof(T));
                pres[ps].second[idx] = this->takemeasure((*this->gptrs)[idx],(*this->nsptrs)[idx]);
                pres[ps].first[idx] = true;
                res[idx] = pres[ps].second[idx];
                return pres[ps].second[idx];
            }
        }
        if (!computed[idx]) {
            res[idx] = this->takemeasure((*this->gptrs)[idx],(*this->nsptrs)[idx]);
            computed[idx] = true;
        }
        return res[idx];
    }

    virtual void takemeasurethreadsection( const int startidx, const int stopidx ) {
        for (int i = startidx; i < stopidx; ++i)
            this->takemeasureidxed(i);
    }
    virtual void takemeasurethreadsectionportion( const int startidx, const int stopidx, std::vector<bool>* todo ) {
        for (int i = startidx; i < stopidx; ++i)
            if ((*todo)[i])
                this->takemeasureidxed(i);
    }


    explicit abstractmemorymeasure( std::string namein ) : abstractmeasure<T>(namein) {}
    ~abstractmemorymeasure() override {
        delete res;
        for (auto p : pres)
        {
            delete p.second.first;
            delete p.second.second;
        }

        pres.clear();
        delete computed;
    }
};


class criterion : public abstractmemorymeasure<bool>
{
public:
    bool negated;

    virtual std::string shortname() {return "_c";}

    bool takemeasureidxed(const int idx) override
    {
        return negated != abstractmemorymeasure<bool>::takemeasureidxed(idx);
    }
    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        return negated != abstractmemorymeasure<bool>::takemeasure(g,ns);
    }

    criterion( const std::string namein )
        : abstractmemorymeasure<bool>(namein) {}

};




class truecriterion : public criterion{
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
    truecriterion() : criterion("always true") {}
};


class trianglefreecriterion : public criterion {
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
    trianglefreecriterion() : criterion("triangle-free criterion") {}
};

class kncriterion : public criterion {
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
            for (auto i = 0; found && (i < n); ++i) {
                for (auto k = i+1; found && (k < n); ++k)
                    found = found && g->adjacencymatrix[dim*subsets[j*n + i] + subsets[j*n + k]];
            }
            foundcnt += (found ? 1 : 0);
            ++j;
        }
        return foundcnt >= mincnt;
    }


    std::string shortname() override {return "k" + std::to_string(n);}
    kncriterion( const int nin) : criterion("embeds K_" + std::to_string(nin) + " criterion"), n{nin} {}
};


// template<typename T>
// class abstractmemoryparameterizedmeasure : public abstractmemorymeasure<T> {
// public:
    // std::vector<std::string> ps {};

    // virtual void setparams( const std::vector<std::string> pin ) {
        // ps = pin;
    // }

    // abstractmemoryparameterizedmeasure( std::string namein ) : abstractmemorymeasure<T>(namein) {}
// };








#endif //ASYMP_H
