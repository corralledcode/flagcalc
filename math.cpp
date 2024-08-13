//
// Created by peterglenn on 7/10/24.
//

#include <iostream>
#include <map>
#include <algorithm>
#include <vector>
#include <cmath>
#include "mathfn.h"
#include "math.h";

#include <cstring>

#include "feature.h"
#include "graphs.h"

#define QUANTIFIER_FORALL "FORALL"
#define QUANTIFIER_EXISTS "EXISTS"

inline bool is_number(const std::string& s)
{
    if (!s.empty() && s[0] == '-')
        return (is_number(s.substr(1,s.size()-1)));
    return !s.empty() && std::find_if(s.begin(),
        s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

inline bool is_real(const std::string& s)
{
    char* end = nullptr;
    double val = strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && val != HUGE_VAL;
}

inline bool evalsentence( logicalsentence ls, std::vector<bool> literals ) {

    bool res;

    if (ls.ls.size()==0) {
        if (ls.item >= 0 && ls.item < literals.size()) {
            res = literals[ls.item];
        } else {
            if (ls.item < 0 && (literals.size() + ls.item >= 0))
                res = literals[literals.size() + ls.item];
            else
                return true;
        }
    }


    if (ls.ls.size() > 0 && ls.lc == logicalconnective::lcand) {
        res = true;
        int n = 0;
        while (res && n < ls.ls.size())
            res = res && evalsentence( ls.ls[n++], literals);
    }
    if (ls.ls.size() > 0 && ls.lc == logicalconnective::lcor) {
        res = false;
        int n = 0;
        while (!res && n < ls.ls.size())
            res = res || evalsentence(ls.ls[n++],literals);
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
    res.item = 0;
    return res;
}



inline std::vector<std::string> parsecomponents( std::string str) {
    std::string partial {};
    std::vector<std::string> components {};
    bool bracketed = false;
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
        if (ch == '+') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("+");
            continue;
        }
        if (!bracketed && ch == '-') {
            if (components.size() > 0) {
                bool keyword = false;
                for (auto k : operatorsmap)
                    keyword = keyword || k.first == components[components.size()-1];
                if (!keyword && components[components.size()-1] != "(") {
                    if (partial != "") {
                        components.push_back(partial);
                        partial = "";
                    }
                    components.push_back("-");
                    continue;
                }
            }
        }
        if (ch == '*') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("*");
            continue;
        }
        if (ch == '/') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("/");
            continue;
        }
        if (ch == '^') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("^");
            continue;
        }
        if (ch == ',') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back(",");
            continue;
        }

        if (ch == '=')
        {
            if (partial != "")
            {
                if (partial[partial.size()-1] == '<' || partial[partial.size()-1] == '>' || partial[partial.size()-1] == '=' || partial[partial.size()-1] == '!')
                {
                    partial.push_back(ch);
                    if (partial.size() > 2)
                        components.push_back(partial.substr(0,partial.size()-2));
                    components.push_back( partial.substr(partial.size() - 2, 2));
                    partial = "";
                    continue;
                }
            }
        }
        if (ch == '&')
        {
            if (partial != "")
            {
                if (partial[partial.size()-1] == '&')
                {
                    partial.push_back(ch);
                    if (partial.size() > 2)
                        components.push_back(partial.substr(0,partial.size()-2));
                    components.push_back( partial.substr(partial.size() - 2, 2));
                    partial = "";
                    continue;
                }
            }
        }
        if (ch == '|')
        {
            if (partial != "")
            {
                if (partial[partial.size()-1] == '|')
                {
                    partial.push_back(ch);
                    if (partial.size() > 2)
                        components.push_back(partial.substr(0,partial.size()-2));
                    components.push_back( partial.substr(partial.size() - 2, 2));
                    partial = "";
                    continue;
                }
            }
        }

        if (ch == '[') {
            bracketed = true;
        }
        if (ch == ']') {
            bracketed = false;
            partial += ']';
            components.push_back(partial);
            partial = "";
            continue;
        }
        if (ch == '<')
        {
            if (partial != "")
            {
                components.push_back(partial);
            }
            partial = "<";
            continue;
        }
        if (ch == '>')
        {
            if (partial != "")
            {
                components.push_back(partial);
            }
            partial = ">";
            continue;
        }
        if (partial.size() > 0 && partial[partial.size()-1] == '!')
        {
            if (partial.size() > 1)
                components.push_back(partial.substr(0,partial.size()-1));
            components.push_back( partial.substr(partial.size() - 1, 1));
            partial = "";
        }
        if (partial.size() > 0 && (partial[partial.size()-1] == '<' || partial[partial.size()-1] == '>'))
        {
            components.push_back( partial );
            partial = "";
        }
        partial += ch;
    }
    if (partial != "") {
        components.push_back(partial);
    }

    // for (auto c : components)
        // std::cout << "c == " << c << ", ";
    // std::cout << "\n";
    return components;
}


inline logicalsentence parsesentenceinternal (std::vector<std::string> components, int nesting ) {
    std::vector<std::string> left {};
    std::vector<std::string> right = components;
    logicalsentence ls;
    if (components.size() < 1) {
        std::cout << "Error in parsesentenceinternal\n";
        ls.ls.clear();
        ls.item = 0;
        return ls;
    }
    if ((components.size() == 1) && is_number(components[0])) {
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



    //for (auto t : components)
    //    std::cout << t << ", ";
    //std::cout << "\n";




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
        }
        if (components[j] == ")") {
            --nesting;
        }
        left.push_back(components[j]);
        ++j;
    }

    if (components.size() >= 3) {
        if (components[0] == "(" && components[components.size()-1] == ")") {
            std::vector<std::string>::const_iterator first = components.begin()+1;
            std::vector<std::string>::const_iterator last = components.begin()+components.size()-1;
            std::vector<std::string> tempcomponents(first,last);
            components = tempcomponents;
            return parsesentenceinternal(components,nesting);
        }

    }



    std::cout << "Ill-formed overall sentence\n";
    ls.ls.clear();
    ls.item = 0;
    ls.negated = false;
    return ls;
}

inline logicalsentence parsesentence( const std::string sentence ) {
    if (sentence != "")
        return parsesentenceinternal( parsecomponents(sentence), 0 );
    else {
        logicalsentence ls;
        ls.ls.clear();
        ls.item = 0;
        return ls;
    }
}

class formulaclass;

inline bool booleanops( const formulaoperator fo)
{
    return (fo == formulaoperator::foand
            || fo == formulaoperator::foor
            || fo == formulaoperator::foimplies
            || fo == formulaoperator::foiff
            || fo == formulaoperator::foxor
            || fo == formulaoperator::foif);
}

inline bool equalityops( const formulaoperator fo)
{
    return (fo == formulaoperator::folte
            || fo == formulaoperator::folt
            || fo == formulaoperator::foe
            || fo == formulaoperator::fogt
            || fo == formulaoperator::fogte
            || fo == formulaoperator::fone);
}

template<typename T, typename T1, typename T2>
T eval2ary(const T1 in1, const T2 in2, const formulaoperator fo)
{
    T res;
    if (fo == formulaoperator::foplus) {
        res = in1 + in2;
    }
    if (fo == formulaoperator::fominus) {
        res = in1 - in2;
    }
    if (fo == formulaoperator::fotimes) {
        res = in1 * in2;
    }
    if (fo == formulaoperator::fodivide) {
        res = (T)in1 / (T)in2;
    }
    if (fo == formulaoperator::foexponent) {
        res = pow(in1, in2);
    }
    if (fo == formulaoperator::foand) {
        res = in1 && in2;
    }
    if (fo == formulaoperator::foor) {
        res = in1 || in2;
    }
    if (fo == formulaoperator::foxor) {
        res = in1 != in2;
    }
    if (fo == formulaoperator::foimplies) {
        res = (!in1) || in2;
    }
    if (fo == formulaoperator::foif) {
        res = in1 || (!in2);
    }
    if (fo == formulaoperator::foiff) {
        res = in1 == in2;
    }
    return res;
}

template<typename T1, typename T2>
bool eval2aryeq( const T1 in1, const T2 in2, const formulaoperator fo) {
    bool res;
    if (fo == formulaoperator::foe) {
        res = in1 == in2;
    }
    if (fo == formulaoperator::folte) {
        res = in1 <= in2;
    }
    if (fo == formulaoperator::folt) {
        res = in1 < in2;
    }
    if (fo == formulaoperator::fogte) {
        res = in1 >= in2;
    }
    if (fo == formulaoperator::fogt) {
        res = in1 > in2;
    }
    if (fo == formulaoperator::fone) {
        res = in1 != in2;
    }
    return res;
}

valms evalformula::evalpslit( const int idx, std::vector<valms>& psin )
{
    valms res;
    res.t = measuretype::mtbool;
    res.v.bv = false;
    return res;
}



valms evalformula::evalvariable( std::string& vname ) {
    // if (!populated && (vname == "E" || vname == "V" || vname == "NE")) {
/*
        int e = lookup_variable("E",*variables);
        int v = lookup_variable("V",*variables);
        int ne = lookup_variable("NE",*variables);
        if (e >= 0)
        {
            delete (*variables)[e];
            variables->erase(variables->begin()+ e);
        }
        if (v >= 0)
        {
            delete (*variables)[v];
            variables->erase(variables->begin()+ v);
        }
        if (ne >= 0)
        {
            delete (*variables)[ne];
            variables->erase(variables->begin() + ne);
        }
*/
        // (*populatevariablesbound)();
        // populated = true;
    // }

    int i = lookup_variable(vname,variables);
    valms res;
    if (i < 0)
    {
        res.t = mtdiscrete;
        res.v.iv = 0;
    } else
    {
        res = variables[i]->qs;
    }
    return res;
}

valms evalformula::eval( formulaclass& fc)
{

    valms res;
    if (fc.fo == formulaoperator::fotrue)
    {
        res.t = measuretype::mtbool;
        res.v.bv = true;
        return res;
    }
    if (fc.fo == formulaoperator::fofalse)
    {
        res.t = measuretype::mtbool;
        res.v.bv = false;
        return res;
    }
    if (fc.fo == formulaoperator::foconstant)
    {
        res.t = fc.v.t;
        switch (res.t)
        {
        case measuretype::mtbool: res.v.bv = fc.v.v.bv;
            return res;
        case mtdiscrete: res.v.iv = fc.v.v.iv;
            return res;
        case mtcontinuous: res.v.dv = fc.v.v.dv;
            return res;
        }
    }


    if (fc.fo == formulaoperator::fofunction) {
        bool found = false;
        double (*fn)(std::vector<double>&) = fc.v.fns.fn;

        std::vector<double> ps;
        for (auto f : fc.v.fns.ps) {
            auto a = eval(*f);
            valms r;
            switch (a.t)
            {
            case mtbool: r.v.dv = (double)a.v.bv;
                break;
            case mtdiscrete: r.v.dv = (double)a.v.iv;
                break;
            case mtcontinuous: r.v.dv = a.v.dv;
                break;
            }
            ps.push_back(r.v.dv);
        }
        res.v.dv = fn(ps);
        res.t = measuretype::mtcontinuous;
        return res;
    }


    if (fc.fo == formulaoperator::fovariable) {
        res = evalvariable(fc.v.qc->name);
        return res;
    }

    if (fc.fo == formulaoperator::foqforall || fc.fo == formulaoperator::foqexists) {
        res.v.bv = true;

        res.t = mtbool;
        switch (fc.v.qc->qs.t) {
            case mtset: {

                valms v = eval(*fc.v.qc->superset);

                bool responsibletodelete = false;
                if (v.t == mtcontinuous) {
                    v.v.iv = (int)v.v.dv;
                    v.t = mtdiscrete;
                }
                if (v.t == mtdiscrete) {
                    v.t = mtset;
                    v.setsize = v.v.iv;
                    v.v.iset = (bool*)malloc(v.setsize*sizeof(bool));
                    memset(v.v.iset,true,v.setsize*sizeof(bool));
                    responsibletodelete = true;
                }

                std::vector<int> subset {};
                for (int i = 0; i < v.setsize; ++i) {
                    if (v.v.iset[i])
                        subset.push_back(i);
                }
                std::vector<int> subsets;
                int m = lookup_variable(fc.v.qc->name, variables);
                variables[m]->qs.v.iset = (bool*)malloc(v.setsize*sizeof(bool));
                variables[m]->qs.setsize = v.setsize;
                // fc.v.qc->qs.v.iset = (bool*)malloc(v.setsize*sizeof(bool));
                // fc.v.qc->qs.setsize = v.setsize;
                for (int i = 1; res.v.bv && (i < subset.size() +1); ++i) {
                    subsets.clear();
                    enumsizedsubsets(0,i,nullptr,0,subset.size(),&subsets);
                    int numberofsubsets = subsets.size()/i;
                    for (int k = 0; res.v.bv && (k < numberofsubsets); ++k) {
                        memset(variables[m]->qs.v.iset,false,v.setsize*sizeof(bool));
                        // memset(fc.v.qc->qs.v.iset,false,v.setsize*sizeof(bool));
                        for (int j = 0; (j < i); ++j) {
                            variables[m]->qs.v.iset[subset[subsets[k*i+j]]] = true;
                            // fc.v.qc->qs.v.iset[subset[subsets[k*i + j]]] = true;
                        }
                        if (fc.fo == formulaoperator::foqexists)
                            res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                        if (fc.fo == formulaoperator::foqforall)
                            res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                    }
                }
                if (fc.fo == formulaoperator::foqexists)
                    res.v.bv = !res.v.bv;
                if (responsibletodelete)
                    delete v.v.iset;
                // delete fc.v.qc->qs.v.iset;
                break;
            }


            case mtpairset: {

                valms v = eval(*fc.v.qc->superset);

                bool responsibletodelete = false;
                if (v.t == mtcontinuous) {
                    v.v.iv = (int)v.v.dv;
                    v.t = mtdiscrete;
                }
                if (v.t == mtdiscrete) {
                    v.t = mtset;
                    v.setsize = v.v.iv;
                    v.v.iset = (bool*)malloc((v.setsize*(v.setsize-1)/2) * sizeof(bool));
                    memset(v.v.iset,true,(v.setsize*(v.setsize-1)/2)*sizeof(bool));
                    responsibletodelete = true;
                }

                std::vector<int> subset {};
                for (int i = 0; i < (v.setsize*(v.setsize-1)/2); ++i) {
                    if (v.v.iset[i])
                        subset.push_back(i);
                }
                std::vector<int> subsets;
                int m = lookup_variable(fc.v.qc->name, variables);
                variables[m]->qs.v.iset = (bool*)malloc((v.setsize*(v.setsize-1)/2)*sizeof(bool));
                variables[m]->qs.setsize = v.setsize;
                // fc.v.qc->qs.v.iset = (bool*)malloc(v.setsize*sizeof(bool));
                // fc.v.qc->qs.setsize = v.setsize;
                for (int i = 1; res.v.bv && (i < subset.size()+1); ++i) {
                    subsets.clear();
                    enumsizedsubsets(0,i,nullptr,0,subset.size(),&subsets);
                    int numberofsubsets = subsets.size()/i;
                    for (int k = 0; res.v.bv && (k < numberofsubsets); ++k) {
                        memset(variables[m]->qs.v.iset,false,(v.setsize*(v.setsize-1)/2)*sizeof(bool));
                        // memset(fc.v.qc->qs.v.iset,false,v.setsize*sizeof(bool));
                        for (int j = 0; (j < i); ++j) {
                            variables[m]->qs.v.iset[subset[subsets[k*i+j]]] = true;
                            // fc.v.qc->qs.v.iset[subset[subsets[k*i + j]]] = true;
                        }
                        if (fc.fo == formulaoperator::foqexists)
                            res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                        if (fc.fo == formulaoperator::foqforall)
                            res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                    }
                }
                if (fc.fo == formulaoperator::foqexists)
                    res.v.bv = !res.v.bv;
                if (responsibletodelete)
                    delete v.v.iset;
                // delete fc.v.qc->qs.v.iset;
                break;
            }



            case mtdiscrete: {
                valms v = eval(*fc.v.qc->superset);
                int maxint = -1;
                switch (v.t) {
                    case mtbool: maxint = (int)v.v.bv; break;
                    case mtdiscrete: maxint = v.v.iv; break;
                    case mtcontinuous: maxint = (int)v.v.dv; break;
                }
                auto superset = v.v.iset;
                auto supersetsize = v.setsize;
                bool* ss = nullptr;
                if (maxint >= 0) {
                    ss = (bool*)malloc(maxint*sizeof(bool));
                    memset(ss,true,maxint*sizeof(bool));
                    superset = ss;
                    supersetsize = maxint;
                }

                int i = lookup_variable(fc.v.qc->name, variables);
                for (variables[i]->qs.v.iv = 0; (res.v.bv) && variables[i]->qs.v.iv < supersetsize; ++variables[i]->qs.v.iv) {
                // for (fc.v.qc->qs.v.iv = 0; (res.v.bv) && fc.v.qc->qs.v.iv < supersetsize; ++fc.v.qc->qs.v.iv) {
                    if (superset[variables[i]->qs.v.iv]) {
                        if (fc.fo == formulaoperator::foqexists)
                            res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                        if (fc.fo == formulaoperator::foqforall)
                            res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                        // std::cout << fc.v.qc->qs.v.iv << " iv \n";
                    }
                }
                if (fc.fo == formulaoperator::foqexists)
                    res.v.bv = !res.v.bv;
                delete ss;
                break;
            }


            case mtpair: {
                valms v = eval(*fc.v.qc->superset);
                int maxint = -1;
                switch (v.t) {
                    case mtbool: maxint = (int)v.v.bv; break;
                    case mtdiscrete: maxint = v.v.iv; break;
                    case mtcontinuous: maxint = (int)v.v.dv; break;
                }
                auto superset = v.v.iset;
                auto supersetsize = v.setsize;
                bool* ss = nullptr;
                if (maxint >= 0) {
                    ss = (bool*)malloc((maxint*(maxint-1)/2)*sizeof(bool));
                    memset(ss,true,(maxint*(maxint-1)/2)*sizeof(bool));
                    superset = ss;
                    supersetsize = maxint;
                }

                int i = lookup_variable(fc.v.qc->name, variables);
                int gapidx = v.setsize-1;
                int idx = gapidx;
                for (variables[i]->qs.v.ip.i = 0; (res.v.bv) && variables[i]->qs.v.ip.i < supersetsize; ++variables[i]->qs.v.ip.i)
                {
                    for (variables[i]->qs.v.ip.j = variables[i]->qs.v.ip.i + 1; (res.v.bv) && variables[i]->qs.v.ip.j < supersetsize; ++variables[i]->qs.v.ip.j) {
                        // for (fc.v.qc->qs.v.iv = 0; (res.v.bv) && fc.v.qc->qs.v.iv < supersetsize; ++fc.v.qc->qs.v.iv) {
                        if (superset[idx + variables[i]->qs.v.ip.i - variables[i]->qs.v.ip.j]) {
                            if (fc.fo == formulaoperator::foqexists)
                                res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                            if (fc.fo == formulaoperator::foqforall)
                                res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                            // std::cout << fc.v.qc->qs.v.iv << " iv \n";
                        }
                    }
                    --gapidx;
                    idx = idx + gapidx;
                }
                if (fc.fo == formulaoperator::foqexists)
                    res.v.bv = !res.v.bv;
                delete ss;
                break;
            }



        }
        return res;
    }


    if ((fc.fo == formulaoperator::foliteral) || (fc.fcleft == nullptr && fc.fcright==nullptr)) {
       if (fc.v.lit.ps.empty())
       {
           if (fc.v.lit.l >= 0 && fc.v.lit.l < literals->size()) {
               res = (*literals)[fc.v.lit.l];
           } else {
               if (fc.v.lit.l < 0 && ((int)literals->size() + fc.v.lit.l >= 0))
               {
                   res = (*literals)[literals->size() + fc.v.lit.l];
               }
               else {
                   std::cout << "Error eval'ing formula\n";
                   return res;
               }
           }
       } else {
           std::vector<valms> ps {};
           for (auto f : fc.v.lit.ps) {
               ps.push_back(eval(*f));
           }
           res = evalpslit(fc.v.lit.l,ps);
        }
        return res;
    }


    valms resright = eval(*fc.fcright);

    if (fc.fo == formulaoperator::fonot)
    {
        res.t = measuretype::mtbool;
        switch (resright.t) {
        case mtbool: res.v.bv = !resright.v.bv;
            return res;
        case mtdiscrete: res.v.bv = !((bool)resright.v.iv);
            return res;
        case mtcontinuous: res.v.bv = (abs(resright.v.dv) < 0.0000001);
            return res;
        }
    }

    valms resleft = eval(*fc.fcleft);

    res.t = fc.v.t;
    // if (!booleanops(fc.fo) && (res.t == mtbool))
        // res.t = mtcontinuous;
    if (equalityops(fc.fo))
    {
        res.t = mtbool;
        switch(resleft.t)
        {
        case mtpair:
            switch (resright.t)
            {
        case mtpair:
                {
                    if (fc.fo == formulaoperator::foe || fc.fo == formulaoperator::fone)
                        res.v.bv = (resleft.v.ip.i == resright.v.ip.i
                                    && resleft.v.ip.j == resright.v.ip.j) != (fc.fo == formulaoperator::fone);
                    else
                        if (resleft.v.ip.i == resright.v.ip.i)
                            res.v.bv = eval2aryeq<int,int>(resleft.v.ip.j, resright.v.ip.j,fc.fo);
                        else
                            res.v.bv = eval2aryeq<int,int>(resleft.v.ip.i, resright.v.ip.i, fc.fo);
                    break;
                }
                std::cout << "Error in evalformula::eval comparing int pair to non-int-pair\n";
            }
            break;
        case mtbool:
            switch (resright.t)
            {
        case mtbool: res.v.bv = eval2aryeq<bool,bool>(resleft.v.bv,resright.v.bv,fc.fo);
                break;
        case mtdiscrete: res.v.bv = eval2aryeq<bool,int>(resleft.v.bv,resright.v.iv,fc.fo);
                break;
        case mtcontinuous: res.v.bv = eval2aryeq<bool,double>(resleft.v.bv,resright.v.dv,fc.fo);
                break;
            }
            break;
        case mtdiscrete:
            switch (resright.t)
            {
        case mtbool: res.v.bv = eval2aryeq<int,bool>(resleft.v.iv,resright.v.bv,fc.fo);
                break;
        case mtdiscrete: res.v.bv = eval2aryeq<int,int>(resleft.v.iv,resright.v.iv,fc.fo);
                break;
        case mtcontinuous: res.v.bv = eval2aryeq<int,double>(resleft.v.iv,resright.v.dv,fc.fo);
                break;
            }
            break;
        case mtcontinuous:
            switch (resright.t)
            {
        case mtbool: res.v.bv = eval2aryeq<double,bool>(resleft.v.dv,resright.v.bv,fc.fo);
                break;
        case mtdiscrete: res.v.bv = eval2aryeq<double,int>(resleft.v.dv,resright.v.iv,fc.fo);
                break;
        case mtcontinuous: res.v.bv = eval2aryeq<double,double>(resleft.v.dv,resright.v.dv,fc.fo);
                break;
            }
            break;
        }
        return res;
    }

    // switch (res.t)
    // {
    // case mtbool:
    switch(resleft.t) {
        case mtbool:
            switch (resright.t)
            {
                case mtbool: res.v.bv = eval2ary<bool,bool,bool>(resleft.v.bv,resright.v.bv,fc.fo);
                    res.t = mtbool;
                    break;
                case mtdiscrete: res.v.iv = eval2ary<int,bool,int>(resleft.v.bv,resright.v.iv,fc.fo);
                    res.t = mtdiscrete;
                    break;
                case mtcontinuous: res.v.dv = eval2ary<double,bool,double>(resleft.v.bv,resright.v.dv,fc.fo);
                    res.t = mtcontinuous;
                    break;
            }
            break;
        case mtdiscrete:
            switch (resright.t)
            {
                case mtbool: res.v.iv = eval2ary<int,int,bool>(resleft.v.iv,resright.v.bv,fc.fo);
                    res.t = mtdiscrete;
                    break;
                case mtdiscrete: if (fc.fo != formulaoperator::fodivide)
                {
                    res.v.iv = eval2ary<int,int,int>(resleft.v.iv,resright.v.iv,fc.fo);
                    res.t = mtdiscrete;
                } else
                {
                    res.v.dv = eval2ary<double,int,int>(resleft.v.iv,resright.v.iv,fc.fo);
                    res.t = mtcontinuous;
                }
                    break;
                case mtcontinuous: res.v.dv = eval2ary<double,int,double>(resleft.v.iv,resright.v.dv,fc.fo);
                    res.t = mtcontinuous;
                    break;
            }
            break;
        case mtcontinuous:
            switch (resright.t)
            {
                case mtbool: res.v.dv = eval2ary<double,double,bool>(resleft.v.dv,resright.v.bv,fc.fo);
                    res.t = mtcontinuous;
                    break;
                case mtdiscrete: res.v.dv = eval2ary<double,double,int>(resleft.v.dv,resright.v.iv,fc.fo);
                    res.t = mtcontinuous;
                    break;
                case mtcontinuous: res.v.dv = eval2ary<double,double,double>(resleft.v.dv,resright.v.dv,fc.fo);
                    res.t = mtcontinuous;
                    break;
            }
            break;
    }
    return res;

}


evalformula::evalformula() {}






inline formulaclass* fccombine( const formulavalue& item, formulaclass* fc1, formulaclass* fc2, formulaoperator fo ) {
    auto res = new formulaclass(item,fc1,fc2,fo);
    return res;
}


inline bool is_operator( const std::string& tok ) {
    for (auto o : operatorsmap)
        if (o.first == tok)
            return true;
    return false;
}

inline formulaoperator lookupoperator( const std::string& tok ) {

    for (auto o : operatorsmap ) {
        if (tok == o.first)
            return o.second;
    }
    return formulaoperator::fotimes;
}

inline bool is_truth( const std::string& tok )
{
    return tok == "TRUE" || tok == "FALSE";
}

inline formulaoperator lookuptruth( const std::string& tok )
{
    return tok == "TRUE" ? formulaoperator::fotrue : formulaoperator::fofalse;
}

inline int operatorprecedence( const std::string& tok1, const std::string& tok2) {
    formulaoperator tok1o = lookupoperator(tok1);
    formulaoperator tok2o = lookupoperator(tok2);

    if (precedencemap[tok1o] == precedencemap[tok2o])
        return 0;
    if (precedencemap[tok1o] < precedencemap[tok2o])
        return 1;
    if (precedencemap[tok1o] > precedencemap[tok2o])
        return -1;
}

inline bool is_function(std::string tok) {
    bool res = !tok.empty();
    for (auto c : tok)
        res = res && isalpha(c);
    return res;
}

inline bool is_literal(std::string tok) {
    return tok.size() > 2 && tok[0] == '[' && tok[tok.size()-1] == ']';
}

inline bool is_variable(std::string tok) {
    bool res = tok.size() > 0 && isalpha(tok[0]);
    for (int i = 1; res && (i < tok.size()); ++i) {
        res = res && isalnum(tok[i]);
    }
    return res;
}

inline int get_literal(std::string tok) {
    if (is_literal(tok))
        return stoi(tok.substr(1,tok.size()-2));
    return 0;
}


inline bool is_quantifier( std::string tok ) {
    return tok == QUANTIFIER_FORALL || tok == QUANTIFIER_EXISTS;
}




inline std::vector<std::string> Shuntingyardalg( const std::vector<std::string>& components, const std::vector<int>& litnumps) {
    std::vector<std::string> output {};
    std::vector<std::string> operatorstack {};

    int n = 0;
    while( n < components.size()) {
        const std::string tok = components[n++];
        if (is_real(tok))
        {
            output.insert(output.begin(),tok);
            continue;
        }
        if (is_number(tok)) {
            output.insert(output.begin(),tok);
            continue;
        }
        if (is_literal(tok)) {
            if (litnumps[get_literal(tok)] > 0)
                operatorstack.push_back(tok);
            else
                output.insert(output.begin(),tok);
            continue;
        }
        if (is_truth(tok)) {
            output.insert(output.begin(),tok);
            continue;
        }
        if (is_quantifier(tok)) {
            operatorstack.push_back(tok);
            continue;
        }
        if (is_operator(tok)) {
            if (operatorstack.size() >= 1) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while ((ostok != "(") && (operatorprecedence(ostok,tok) >= 0)) {
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (operatorstack.size() >= 1)
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                        break;
                }
            }
            operatorstack.push_back(tok);
            continue;
        }
        if (is_function(tok)) {
            operatorstack.push_back(tok);
            continue;
        }
        if (is_variable(tok)) {
            operatorstack.push_back(tok);
            continue;
        }
        if (tok == ",") {
            if (operatorstack.size() >= 1) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "(") {
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    ostok = operatorstack[operatorstack.size()-1];
                }
            }
            continue;
        }
        if (tok == "(") {
            operatorstack.push_back(tok);
            continue;
        }
        if (tok == ")") {
            std::string ostok;
            if (operatorstack.size() >= 1)
            {
                ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "(") {
                    if (operatorstack.empty()) {
                        std::cout << "Error mismatched parentheses\n";
                        return output;
                    }
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (operatorstack.size() >= 1)
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Error mismatched parentheses\n";
                        return output;
                    }
                }
            }
            if (operatorstack.empty() || operatorstack[operatorstack.size()-1] != "(") {
                std::cout << "Error mistmatched parentheses\n";
                return output;
            }
            operatorstack.resize(operatorstack.size()-1);
            if (operatorstack.size() > 0) {
                ostok = operatorstack[operatorstack.size()-1];
                if (is_function(ostok))
                {
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    continue;
                }
                if (is_quantifier(ostok)) {
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    continue;
                }
                if (is_literal(ostok)) {
                    if (litnumps[get_literal(ostok)] > 0)
                    {
                        output.insert(output.begin(),ostok);
                        operatorstack.resize(operatorstack.size()-1);
                        continue;
                    }
                }
                if (is_variable(ostok))
                {
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    continue;
                }
            }
            continue;
        }


    }
    while (operatorstack.size()> 0) {
        std::string ostok = operatorstack[operatorstack.size()-1];
        if (ostok == "(") {
            std::cout << "Error mismatched parentheses\n";
            return output;
        }
        output.insert(output.begin(),ostok);
        operatorstack.resize(operatorstack.size()-1);
    }

    // for (auto o : output)
        // std::cout << o << ", ";
    // std::cout << "\n";

    return output;

}




inline formulaclass* parseformulainternal(
    std::vector<std::string>& q,
    int& pos,
    const std::vector<int>& litnumps,
    const std::vector<measuretype>& littypes,
    std::vector<qclass*>* variables,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs )
{
    while( pos+1 < q.size()) {
        ++pos;
        std::string tok = q[pos];
        if (is_quantifier(tok)) {
            auto qc = new qclass;
            qc->eval( q, ++pos);
            qc->superset = parseformulainternal(q, pos, litnumps,littypes,variables,fnptrs);
            variables->push_back(qc);
            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes,variables,fnptrs);
            formulaclass* fcleft = nullptr;
            formulaoperator o = lookupoperator(tok);
            formulavalue fv {};
            fv.qc = qc;
            auto fc = fccombine(fv,fcleft,fcright,o);
            return fc;
        }
        if (is_operator(tok))
        {
            formulaoperator o = lookupoperator(tok);
            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes, variables,fnptrs);
            formulaclass* fcleft = nullptr;
            if (o != formulaoperator::fonot)
            {
                fcleft = parseformulainternal(q,pos,litnumps,littypes,variables,fnptrs);
            }
            if (fcright)
                return fccombine({0},fcleft,fcright,o);
        }
        int v = lookup_variable(tok,*variables);
        if (v >= 0) {
            formulavalue fv {};

            fv.t = (*variables)[v]->qs.t;
            fv.v = (*variables)[v]->qs.v;
            auto fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariable);
            fc->v.qc = (*variables)[v];
            return fc;
        }
        if (tok == "V" || tok == "E" || tok == "NE") {
            formulavalue fv {};
            fv.t = mtset;
            auto fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariable);
            fc->v.qc = new qclass;
            fc->v.qc->name = tok;
            return fc;
        }
        if (is_function(tok)) {
            std::vector<formulaclass*> ps {};
            // double (*f2)(std::vector<double>&);
            int argcnt = 0;
            for (auto f : *fnptrs)
                if (f.first == tok)
                    argcnt = f.second.second;
            std::vector<formulaclass*> psrev {};
            for (int i = 0; i < argcnt; ++i) {
                psrev.push_back(parseformulainternal(q,pos,litnumps,littypes,variables,fnptrs));
            }
            for (int i = psrev.size()-1; i >= 0; --i)
                ps.push_back(psrev[i]);

            formulavalue fv {};
            if (auto search = fnptrs->find(tok); search != fnptrs->end())
            {
                fv.fns.fn = search->second.first;
                fv.fns.ps = ps;
                fv.t = mtcontinuous;
                return fccombine(fv,nullptr,nullptr,formulaoperator::fofunction);
            } else
            {
                std::cout << "Unknown function " << tok << " in parseformula internal\n";
            }
        }
        if (is_literal(tok))
        {
            formulavalue fv {};
            fv.lit.l = get_literal(tok);
            fv.lit.ps.clear();
            fv.t = littypes[fv.lit.l];
            if (litnumps[fv.lit.l] == 0)
                return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
            else
            {
                int argcnt = litnumps[fv.lit.l];
                std::vector<formulaclass*> psrev {};
                for (int i = 0; i < argcnt; ++i) {
                    psrev.push_back(parseformulainternal(q,pos,litnumps,littypes,variables,fnptrs));
                }
                for (int i = psrev.size()-1; i >= 0; --i)
                    fv.lit.ps.push_back(psrev[i]);

                return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
            }

        }
        if (is_number(tok)) { // integer
            formulavalue fv {};
            fv.v.iv = stoi(tok);
            fv.t = mtdiscrete;
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }
        if (is_real(tok))
        {
            formulavalue fv {};
            fv.v.dv = stof(tok);
            fv.t = mtcontinuous;
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }
        if (is_truth(tok))
        {
            formulavalue fv {};
            fv.t = mtbool;
            formulaoperator t = lookuptruth(tok);
            fv.v.bv = t == formulaoperator::fotrue;
            return fccombine(fv,nullptr,nullptr,t);
        }
    }
    std::cout << "Error in parsing formula \n";
    auto fc = new formulaclass({0},nullptr,nullptr,formulaoperator::foconstant);
    return fc;
}

int preprocessforquantifiers( const std::vector<std::string>& components )
{
    int res = 0;
    for (auto c : components)
        if (is_quantifier(c))
            ++res;
    return res;
}


formulaclass* parseformula(
    const std::string& sentence,
    const std::vector<int>& litnumps,
    const std::vector<measuretype>& littypes,
    std::vector<qclass*>& variables,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs )
{
    if (sentence != "") {
        variables.clear();
        std::vector<std::string> c = parsecomponents(sentence);
        
        std::vector<std::string> components = Shuntingyardalg(c,litnumps);
        int pos = -1;
        return parseformulainternal( components,pos, litnumps, littypes, &variables,fnptrs);
    } else {
        auto fc = new formulaclass({0},nullptr,nullptr,formulaoperator::foconstant);
        return fc;
    }
}

