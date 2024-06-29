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
//#include "workspace.h"





template<typename T>
class abstractcriterion {
protected:
public:
    std::vector<graphtype*>* gptrs {};
    std::vector<neighbors*>* nsptrs {};
    //T res {};
    std::string name = "_abstractcriterion";
    virtual std::string shortname() {return "_ac";}

    virtual T checkcriterion( const graphtype* g, const neighbors* ns ) {}
    virtual T checkcriterionidxed( const int idx )
    {
        return checkcriterion((*gptrs)[idx],(*nsptrs)[idx]);
    }
    abstractcriterion( std::string namein ) :name{namein} {}
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
public:
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


    std::string shortname() override {return "cr1";}
    trianglefreecriterion() : abstractcriterion("triangle-free criterion") {}
};

class kncriterion : public abstractcriterion<bool> {
public:
    const int n;
    bool checkcriterion( const graphtype* g, const neighbors* ns) override {
        int dim = g->dim;

        std::vector<int> subsets {};
        enumsizedsubsets(0,n,nullptr,0,dim,&subsets);

        bool found = false;
        int j = 0;
        while (!found && j < (subsets.size()/n)) {
            found = true;
            for (auto i = 0; found && (i < n-1); ++i) {
                for (auto k = i+1; found && (k < n); ++k)
                    found = found && g->adjacencymatrix[dim*subsets[j*n + i] + subsets[j*n + k]];
            }
            ++j;
        }
        return found;
    }


    std::string shortname() override {return "k" + std::to_string(n);}
    kncriterion( const int nin) : abstractcriterion("embeds K_" + std::to_string(nin) + " criterion"), n{nin} {}
};




class embedscriterion : public abstractcriterion<bool> {
protected:

public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    std::string shortname() override {return "cr3";}
    embedscriterion(neighbors* flagnsin,FP* fpin) : abstractcriterion("embeds flag criterion"), flagg{flagnsin->g},flagns{flagnsin},fp{fpin} {}
    bool checkcriterion( const graphtype* g, const neighbors* ns) override {
        return (embeds(flagns, fp, ns));
    }
};

enum class logicalconnective {lcand, lcor};

struct logicalsentence {
    int item {};
    std::vector<logicalsentence> ls {};
    logicalconnective lc {logicalconnective::lcand};
    bool negated {false};
};

inline bool evalsentence( logicalsentence ls, std::vector<bool> literals ) {

    bool res;

    if (ls.ls.size()==0) {
        if (ls.item >= 0) {
            res = literals[ls.item];
        } else {
            return true;
        }
    }


    if (ls.ls.size() > 0 && ls.lc == logicalconnective::lcand) {
        res = true;
        int n = 0;
        while (res && n < ls.ls.size())
            res &= evalsentence( ls.ls[n++], literals);
    }
    if (ls.ls.size() > 0 && ls.lc == logicalconnective::lcor) {
        res = false;
        int n = 0;
        while (!res && n < ls.ls.size())
            res |= evalsentence(ls.ls[n++],literals);
    }
    //std::cout << (ls.ls.size()) << "ls.ls.size()\n";
    //std::cout << res << " (res)\n";
    return ls.negated != res;
}

inline logicalsentence lscombine( const logicalsentence ls1, const logicalsentence ls2, const logicalconnective lc ) {
    logicalsentence res;
    res.ls.push_back(ls1);
    res.ls.push_back(ls2);
    res.lc = lc;
    res.negated = false;
    res.item = -1;
    return res;
}

inline std::vector<std::string> parsecomponents( std::string str ) {
    std::string partial {};
    std::vector<std::string> components {};
    for (auto ch : str) {
        if (ch == ' ') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            continue;
        }
        if (ch == '(') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("(");
            continue;
        }
        if (ch == ')') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back(")");
            continue;
        }
        partial += ch;
    }
    if (partial != "") {
        components.push_back(partial);
    }
    //for (auto c : components)
    //    std::cout << "c == " << c << ", ";
    //std::cout << "\n";
    return components;
}

inline bool is_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(),
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

inline logicalsentence parsesentenceinternal (std::vector<std::string> components, int nesting ) {
    std::vector<std::string> left {};
    std::vector<std::string> right = components;
    logicalsentence ls;
    if (components.size() < 1) {
        std::cout << "Error in parsesentenceinternal\n";
        ls.ls.clear();
        ls.item = -1;
        return ls;
    }
    if (components.size() == 1 && is_number(components[0])) {
        ls.ls.clear();
        ls.item = stoi(components[0]);
        ls.negated = false;
        return ls;
    }

    if (components.size() > 0 && components[0] == "NOT") {
        right.erase(right.begin(), right.begin()+1);
        ls = parsesentenceinternal( right, nesting );
        ls.negated = !ls.negated;
        return ls;
    }


    if (components.size() >= 2) {
        if (components[0] == "(" && components[components.size()-1] == ")") {
            std::vector<std::string>::const_iterator first = components.begin()+1;
            std::vector<std::string>::const_iterator last = components.begin()+components.size()-1;
            std::vector<std::string> tempcomponents(first,last);
            components = tempcomponents;
        }

    }

    right = components;
    left.clear();
    int j = 0;
    logicalsentence ls1;
    logicalsentence ls2;
    while (j < components.size()) {
        right.erase(right.begin(), right.begin()+1);
        if (nesting <= 0) {
            if ((components[j] == "AND" || components[j] == "OR")) {
                ls1 = parsesentenceinternal(left,nesting);
                ls2 = parsesentenceinternal(right,nesting);
                if (components[j] == "AND") {
                    return lscombine(ls1,ls2,logicalconnective::lcand);
                }
                if (components[j] == "OR") {
                    return lscombine(ls1,ls2,logicalconnective::lcor);
                }
            }
        }
        if (components[j] == "(") {
            ++nesting;
            //return parsesentenceinternal( right, nesting );
        }
        if (components[j] == ")") {
            --nesting;
            //return parsesentenceinternal( left, nesting );
        }
        left.push_back(components[j]);
        ++j;
    }


    std::cout << "Ill-formed overall sentence\n";
    ls.ls.clear();
    ls.item = -1;
    ls.negated = false;
    return ls;
}

inline logicalsentence parsesentence( std::string sentence ) {
    if (sentence != "")
        return parsesentenceinternal( parsecomponents(sentence), 0 );
    else {
        logicalsentence ls;
        ls.ls.clear();
        ls.item = -1;
        return ls;
    }
}

class sentenceofcriteria : public abstractcriterion<bool> {
protected:
    //std::vector<abstractcriterion<bool>*> cs;
    logicalsentence ls;
    std::vector<bool*> variables {};
    const int sz;

public:
    std::string shortname() override {return "crSENTENCE";}

    bool checkcriterionidxed( const int idx ) override {
        std::vector<bool> res {};
        res.resize(variables.size());
        for (int i = 0; i < variables.size(); ++i )
            res[i] = variables[i][idx];
        return evalsentence(ls,res);  // check for speed cost
    }

    sentenceofcriteria( std::vector<bool*> variablesin, const int szin, std::string sentence, std::string stringin )
        : abstractcriterion<bool>(stringin == "" ? "logical sentence of several criteria" : stringin),
            variables{variablesin}, ls{parsesentence(sentence)}, sz{szin} {}

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

class notcriteria : public abstractcriterion<bool> {
protected:
    const int sz;
    std::vector<bool*> variables {};
public:
    std::string shortname() override {return "crNOT";}
    std::vector<bool> neg {};
    std::vector<bool*> res {};
    notcriteria(std::vector<bool*> variablesin, const int szin, std::vector<bool> negin)
        : abstractcriterion<bool>("logical NOT of several criteria"), variables{variablesin}, neg{negin}, sz{szin}
    {
        res.resize(variables.size());
        for (int i = 0; i < variables.size(); ++i)
            res[i] = (bool*)malloc(sz*sizeof(bool));
    }

    ~notcriteria()
    {
        for (int i = 0; i < res.size(); ++i)
        {
            free(res[i]);
        }
    }

    bool checkcriterionidxed(const int idx) override
    {
        for (int i = 0; i < res.size(); ++i)
        {
            res[i][idx] = variables[i][idx] != neg[i];
        }
        // technically the return value should be a vector
        return true;
    }

};

#endif //ASYMP_H
