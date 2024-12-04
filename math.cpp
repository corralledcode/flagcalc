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

#include <complex>
#include <cstring>

#include "feature.h"
#include "graphs.h"

#define QUANTIFIERMODE_PRECEDING
#define SHUNTINGYARDVARIABLEARGUMENTKEY "arg#"

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
        if (ch == '%') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("%");
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

inline bool quantifierops( const formulaoperator fo )
{
    return (fo == formulaoperator::foqforall
            || fo == formulaoperator::foqexists
            || fo == formulaoperator::foqproduct
            || fo == formulaoperator::foqsum
            || fo == formulaoperator::foqmin
            || fo == formulaoperator::foqmax
            || fo == formulaoperator::foqrange
            || fo == formulaoperator::foqaverage
            || fo == formulaoperator::foqtally
            || fo == formulaoperator::foqcount
            || fo == formulaoperator::foqset
            || fo == formulaoperator::foqdupeset
            || fo == formulaoperator::foqunion
            || fo == formulaoperator::foqdupeunion
            || fo == formulaoperator::foqintersection);
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
    if (fo == formulaoperator::fomodulus)
    {
        res = (int)in1 % (int)in2;
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
        res = abs(in1 - in2) < ABSCUTOFF;
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
        res = abs(in1 - in2) >= ABSCUTOFF;
    }
    return res;
}


bool setsubseteq( itrpos* in1, itrpos* in2)
{
    bool res = true;
    in1->reset();
    while (!in1->ended() && res)
    {
        in2->reset();
        auto itm = in1->getnext();
        bool match = false;
        while (!match && !in2->ended())
        {
            auto itm2 = in2->getnext();
            if (itm.t == mtbool || itm.t == mtdiscrete || itm.t == mtcontinuous)
                match = itm == itm2;
            else
                if (itm.t == mtset && itm2.t == mtset)
                {
                    auto pos1 = itm.seti->getitrpos();
                    auto pos2 = itm2.seti->getitrpos();
                    match = setsubseteq( pos1, pos2) && setsubseteq( pos2, pos1);
                    delete pos1;
                    delete pos2;
                } else
                    if (itm.t == mttuple && itm2.t == mttuple)
                    {
                        auto pos1 = itm.seti->getitrpos();
                        auto pos2 = itm2.seti->getitrpos();
                        match = tupleeq( pos1, pos2);
                        delete pos1;
                        delete pos2;
                    }
                    else {
                        std::cout << "Mismatched set type in setsubseteq\n";
                        exit(1);
                    }
        }
        res = match;
    }
    return res;
}


bool tupleeq( itrpos* in1, itrpos* in2)
{
    bool res = true;
    in1->reset();
    in2->reset();
    if (in1->ended())
        if (in2->ended())
            return true;
        else
            return false;
    if (in2->ended())
        return false;

    while (res)
    {
        auto itm = in1->getnext();
        auto itm2 = in2->getnext();

        bool e1 = in1->ended();
        bool e2 = in2->ended();
        if (e1 != e2)
            return false;

        if (itm.t == mtbool || itm.t == mtdiscrete || itm.t == mtcontinuous)
            res = itm == itm2;
        else
            if (itm.t == mtset && itm2.t == mtset)
            {
                auto pos1 = itm.seti->getitrpos();
                auto pos2 = itm2.seti->getitrpos();
                res = setsubseteq( pos1, pos2) && setsubseteq( pos2, pos1);
                delete pos1;
                delete pos2;
            } else
                if (itm.t == mttuple && itm2.t == mttuple)
                {
                    auto pos1 = itm.seti->getitrpos();
                    auto pos2 = itm2.seti->getitrpos();
                    res = tupleeq( pos1, pos2);
                    delete pos1;
                    delete pos2;
                }
                else {
                    std::cout << "Mismatched tuple type in tupleeq\n";
                    exit(1);
                }
        if (e1) // which equals e2...
            return res;
    }
    return res;
}






bool eval2aryseteq( setitr* in1, setitr* in2, const formulaoperator fo )
{
    bool res;
    auto pos1 = in1->getitrpos();
    auto pos2 = in2->getitrpos();
    if (fo == formulaoperator::foe) {
        res = setsubseteq( pos1, pos2) && setsubseteq( pos2, pos1);
    }
    if (fo == formulaoperator::folte)
    {
        res = setsubseteq( pos1, pos2);
    }
    if (fo == formulaoperator::folt)
    {
        res = pos1->getsize() != pos2->getsize();
        if (res)
            res = setsubseteq(pos1, pos2);
    }
    if (fo == formulaoperator::fogte)
    {
        res = setsubseteq( pos2, pos1);
    }
    if (fo == formulaoperator::fogt)
    {
        res = pos1->getsize() != pos2->getsize();
        if (res)
            res = setsubseteq(pos2, pos1);
    }
    if (fo == formulaoperator::fone)
    {
        res = pos1->getsize() != pos2->getsize();
        if (!res)
        {
            res = !setsubseteq(pos1, pos2);
            if (!res)
                res = !setsubseteq(pos2, pos1);
        }
    }
    return res;
}


bool eval2arytupleeq( setitr* in1, setitr* in2, const formulaoperator fo )
{
    bool res;
    auto pos1 = in1->getitrpos();
    auto pos2 = in2->getitrpos();
    if (fo == formulaoperator::foe) {
        res = tupleeq( pos1, pos2);
    }


    /* .. to do: add some notion of comparing tuples

    if (fo == formulaoperator::folte)
    {
        res = setsubseteq( pos1, pos2);
    }
    if (fo == formulaoperator::folt)
    {
        res = pos1->getsize() != pos2->getsize();
        if (res)
            res = setsubseteq(pos1, pos2);
    }
    if (fo == formulaoperator::fogte)
    {
        res = setsubseteq( pos2, pos1);
    }
    if (fo == formulaoperator::fogt)
    {
        res = pos1->getsize() != pos2->getsize();
        if (res)
            res = setsubseteq(pos2, pos1);
    } */

    if (fo == formulaoperator::fone)
    {
        res = !tupleeq( pos1, pos2);
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



valms evalformula::evalvariable( std::string& vname, std::vector<int>& vidxin ) {
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
        if (vidxin.size() == 0)
            res = variables[i]->qs;
        else {
            if (vidxin.size() != 1)
            {
                std::cout << "Wrong number of parameters passed to de-index a variable\n";
            } else
            {
                int index = vidxin[0];
                auto pos = variables[i]->qs.seti->getitrpos();
                int i = 0;
                while (!pos->ended() && i++ <= index) {
                    res = pos->getnext();
                }
            }
        }

    }
    return res;
}

inline int pairlookup( const intpair p, const int dim)
{
    int gapidx = dim;
    int idx = gapidx;
    int i;
    for (i = 0; (i != p.i) && (i < dim); ++i)
    {
        --gapidx;
        idx += gapidx;
    }
    if (i == p.i)
        return idx + i - p.j;
    return 0;
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
        res.t = fc.v.v.t;
        switch (res.t)
        {
        case measuretype::mtbool: res.v.bv = fc.v.v.v.bv;
            return res;
        case mtdiscrete: res.v.iv = fc.v.v.v.iv;
            return res;
        case mtcontinuous: res.v.dv = fc.v.v.v.dv;
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


        if (fc.v.vs.ps.empty())
        {
            std::vector<int> ps {};
            res = evalvariable(fc.v.vs.name,ps);
        } else {
            std::vector<int> ps {};
            for (auto f : fc.v.vs.ps) {
                ps.push_back(eval(*f).v.iv);
                // std::cout << "ps type " << f->v.v.t << " type " << ps.back().t << "seti type " << ps.back().seti->t << "\n";
            }
            res = evalvariable(fc.v.vs.name,ps);
        }
        return res;
    }

    if (fc.fo == formulaoperator::foelt)
    {
        res.t = measuretype::mtbool;
        valms set = eval(*fc.fcright );
        valms itm = eval( *fc.fcleft );
        if (set.t == mtset || set.t == mttuple)
        {
            auto pos = set.seti->getitrpos();
            pos->reset();
            bool match = false;
            while ( !match && !pos->ended())
            {
                auto v = pos->getnext();
                if (v.t == mtbool || v.t == mtdiscrete || v.t == mtcontinuous)
                    match = match || itm == v;
                else
                    if (v.t == mtset)
                    {
                        if (itm.t == mtset || itm.t == mttuple)  // for now, auto convert a tuple to a set here
                        {
                            auto itmpos = itm.seti->getitrpos();
                            auto pospos = v.seti->getitrpos();
                            match = match || (setsubseteq(itmpos, pospos) && setsubseteq(pospos, itmpos));
                            delete itmpos;
                            delete pospos;
                        } else
                        {
                            std::cout << "Mismatched types in call to ELT\n";
                            exit(1);
                        }
                    } else
                        if (v.t == mttuple)
                        {
                            if (itm.t == mttuple )
                            {
                                auto itmpos = itm.seti->getitrpos();
                                auto pospos = v.seti->getitrpos();
                                match = match || (tupleeq(itmpos, pospos));
                                delete itmpos;
                                delete pospos;
                            } else
                                if (itm.t == mtset) // for now, generously auto convert the tuple to a set
                                {
                                    auto itmpos = itm.seti->getitrpos();
                                    auto pospos = v.seti->getitrpos();
                                    match = match || (setsubseteq(itmpos, pospos) && setsubseteq(pospos, itmpos));
                                    delete itmpos;
                                    delete pospos;
                                } else
                                {
                                    std::cout << "Mismatched types in call to ELT\n";
                                    exit(1);
                                }
                        }
            }
            res.v.bv = match;

                // if (itm.v.iv >= set.setsize)
            // {
                // std::cout << "Set size exceeded in call to ELT\n";
                // res.v.bv = false;
                // return res;
            // }
            // for (int i = 0; i < set.setsize; ++i)
                // std::cout << "set " << i << " == " << set.v.iset[i] << "\n";

            return res;
        }
        // else if (itm.t == mtpair && set.t == mtpairset)
        // {
            // int dim = set.setsize;
            // int idx = pairlookup( itm.v.ip, dim);
            // if (idx >= (dim * (dim-1))/2)
            // {
                // std::cout << "Pair set size exceeded in call to ELT\n";
                // res.v.bv = false;
                // return res;
            // }
            // res.v.bv = set.v.iset[idx];
            // return res;
        // }
        else {
            res.v.bv = false;
            std::cout << "Non-matching types in use of ELT, " << itm.t << ", " << set.t << "\n";
            exit(1);
            return res;
        }
    }

    if (fc.fo == formulaoperator::founion || fc.fo == formulaoperator::fointersection
        || fc.fo == formulaoperator::fodupeunion)
    {
        valms set1 = eval(*fc.fcright );
        valms set2 = eval( *fc.fcleft );
        if (set1.t == mtset && set2.t == mtset)
        {
            if (fc.fo == formulaoperator::founion)
                res.seti = new setitrunion(set1.seti,set2.seti);
            else
                if (fc.fo == formulaoperator::fointersection)
                    res.seti = new setitrintersection(set1.seti,set2.seti);
                else
                    res.seti = new setitrdupeunion( set1.seti,set2.seti);
            res.t = mtset;
            res.setsize = res.seti->getsize();

        }
        // else if (set1.t == mtpairset && set2.t == mtpairset)
        // {
            // int L = set1.setsize;
            // int R = set2.setsize;
            // res.t = mtpairset;
            // res.setsize = L <= R ? R : L;
            // res.v.iset = (bool*)malloc((res.setsize*(res.setsize-1))/2*sizeof(bool));

            // if (fc.fo == formulaoperator::founion)
                // for (int i = 0; i < (res.setsize*(res.setsize-1)/2); ++i)
                // {
                    // res.v.iset[i] = (i < (L*(L-1))/2 && set1.v.iset[i]) || (i < (R*(R-1))/2 && set2.v.iset[i]);
                // } else // fointersection
                    // for (int i = 0; i < (res.setsize*(res.setsize-1)/2); ++i)
                    // {
                        // res.v.iset[i] = (i < (L*(L-1))/2 && set1.v.iset[i]) && (i < (R*(R-1))/2 && set2.v.iset[i]);
                    // }
        // }
        else {
            std::cout << "Non-matching types in call to CUP \n";
            res.seti = nullptr;
            res.setsize = 0;
        }
        return res;
    }

    if (quantifierops(fc.fo)) {
        res.v.bv = true;

        res.t = mtbool;
/*        switch (fc.v.qc->secondorder) {
            case true: {
                    valms v = eval(*fc.v.qc->superset);
                    bool responsibletodelete = false;
                    if (v.t == mtcontinuous) {
                        v.v.iv = (int)v.v.dv;
                        v.t = mtdiscrete;
                    }
                    if (v.t == mtdiscrete) {
                        v.seti = new setitrint(v.v.iv);
                        responsibletodelete = true;
                    }

                    std::vector<valms> subset {};
                    // for (int i = 0; i < v.setsize; ++i) {
                        // if (v.v.iset[i])
                            // subset.push_back(i);
                    // }
                    std::vector<int> subsets;
                    int m = lookup_variable(fc.v.qc->name, variables);
                    // variables[m]->qs.v.iset = (bool*)malloc(v.setsize*sizeof(bool));
                    // variables[m]->qs.setsize = v.setsize;
                    // fc.v.qc->qs.v.iset = (bool*)malloc(v.setsize*sizeof(bool));
                    // fc.v.qc->qs.setsize = v.setsize;

                    auto itr = new setitrsubset(v.seti->getitrpos());
                    memset(itr->itrbool->elts,false,itr->itrbool->maxint*sizeof(bool));
                    int numberofsubsets = 1;
                    variables[m]->qs.seti = itr;
                    itr->t = v.seti->t;
                    for (int k = 0; res.v.bv && (k < numberofsubsets); ++k) {
                        // memset(variables[m]->qs.v.iset,false,v.setsize*sizeof(bool));
                        // memset(fc.v.qc->qs.v.iset,false,v.setsize*sizeof(bool));
                        if (fc.fo == formulaoperator::foqexists)
                            res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                        if (fc.fo == formulaoperator::foqforall)
                            res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                    }
                    while (!v.seti->ended())
                    {
                        subset.push_back(v.seti->getnext());
                        for (int i = 0; i < subset.size();++i)
                        {
                            subsets.clear();
                            enumsizedsubsets(0,i,nullptr,0,subset.size(),&subsets);
                            numberofsubsets = subsets.size()/i;
                            for (int k = 0; res.v.bv && (k < numberofsubsets); ++k) {
                                memset(itr->itrbool->elts,false,v.setsize*sizeof(bool));
                                // memset(fc.v.qc->qs.v.iset,false,v.setsize*sizeof(bool));
                                for (int j = 0; (j < i); ++j) {
                                    itr->itrbool->elts[subsets[k*i+j]] = true;
                                    // fc.v.qc->qs.v.iset[subset[subsets[k*i + j]]] = true;
                                }
                                if (fc.fo == formulaoperator::foqexists)
                                    res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                                if (fc.fo == formulaoperator::foqforall)
                                    res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                            }
                        }


                    }

                    if (fc.fo == formulaoperator::foqexists)
                        res.v.bv = !res.v.bv;
                    if (responsibletodelete)
                        delete v.seti;
                    // delete fc.v.qc->qs.v.iset;
                    break;
                }


            // case mtpairset: {

                // valms v = eval(*fc.v.qc->superset);

                // bool responsibletodelete = false;
                // if (v.t == mtcontinuous) {
                    // v.v.iv = (int)v.v.dv;
                    // v.t = mtdiscrete;
                // }
                // if (v.t == mtdiscrete) {
                    // v.t = mtset;
                    // v.setsize = v.v.iv;
                    // v.v.iset = (bool*)malloc((v.setsize*(v.setsize-1)/2) * sizeof(bool));
                    // memset(v.v.iset,true,(v.setsize*(v.setsize-1)/2)*sizeof(bool));
                    // responsibletodelete = true;
                // }

                // std::vector<int> subset {};
                // for (int i = 0; i < (v.setsize*(v.setsize-1)/2); ++i) {
                    // if (v.v.iset[i])
                        // subset.push_back(i);
                // }
                // std::vector<int> subsets;
                // int m = lookup_variable(fc.v.qc->name, variables);
                // variables[m]->qs.v.iset = (bool*)malloc((v.setsize*(v.setsize-1)/2)*sizeof(bool));
                // variables[m]->qs.setsize = v.setsize;
                // fc.v.qc->qs.v.iset = (bool*)malloc(v.setsize*sizeof(bool));
                // fc.v.qc->qs.setsize = v.setsize;
                // for (int i = 0; res.v.bv && (i < subset.size()+1); ++i) {
                    // subsets.clear();
                    // int numberofsubsets;
                    // if (i > 0)
                    // {
                        // enumsizedsubsets(0,i,nullptr,0,subset.size(),&subsets);
                        // numberofsubsets = subsets.size()/i;
                    // }
                    // else
                        // numberofsubsets = 1;
                    // for (int k = 0; res.v.bv && (k < numberofsubsets); ++k) {
                        // memset(variables[m]->qs.v.iset,false,(v.setsize*(v.setsize-1)/2)*sizeof(bool));
                        // memset(fc.v.qc->qs.v.iset,false,v.setsize*sizeof(bool));
                        // for (int j = 0; (j < i); ++j) {
                            // variables[m]->qs.v.iset[subset[subsets[k*i+j]]] = true;
                            // fc.v.qc->qs.v.iset[subset[subsets[k*i + j]]] = true;
                        // }
                        // if (fc.fo == formulaoperator::foqexists)
                            // res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                        // if (fc.fo == formulaoperator::foqforall)
                            // res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                    // }
                // }
                // if (fc.fo == formulaoperator::foqexists)
                    // res.v.bv = !res.v.bv;
                // if (responsibletodelete)
                    // delete v.v.iset;
                // delete fc.v.qc->qs.v.iset;
                // break;
            // }



            case false: {
*/
        valms v = eval(*fc.v.qc->superset);
        auto criterion = fc.v.qc->criterion;
        int maxint = -1;
//                switch (v.t) {
//                    case mtbool: maxint = (int)v.v.bv; break;
//                    case mtdiscrete: maxint = v.v.iv - 1; break;
//                    case mtcontinuous: maxint = (int)v.v.dv - 1; break;
//                }
        // while (!v.seti->ended())
            // v.seti->getnext();
        auto supersetpos = v.seti->getitrpos();
        auto supersetsize = supersetpos->getsize();
        setitrint* ss = nullptr;
//                if (maxint >= 0) {
//                    ss = new setitrint(maxint);
//                    memset(ss->elts,true,(maxint+1)*sizeof(bool));
//                    supersetpos = ss->getitrpos();
//                    supersetsize = maxint;
//                }

        int i = lookup_variable(fc.v.qc->name, variables);
        supersetpos->reset();
        // variables[i]->qs.t = v.seti->t;
        if (fc.fo == formulaoperator::foqexists || fc.fo == formulaoperator::foqforall)
        {
            res.t = mtbool;
            while ((!supersetpos->ended()) && res.v.bv) {
                variables[i]->qs = supersetpos->getnext();
                valms c;
                if (criterion) {
                    c = eval(*criterion);
                    if (c.t == mtbool && !c.v.bv)
                        continue;
                    if (c.t != mtbool) {
                        std::cout << "Quantifier criterion requires boolean value\n";
                        exit(1);
                    }
                }
                // std::cout << "variables t == " << variables[i]->qs.t << ", name == " << fc.v.qc->name <<  std::endl;
                // std::cout << "supersetpos ended == " << supersetpos->ended() << std::endl;
                // for (fc.v.qc->qs.v.iv = 0; (res.v.bv) && fc.v.qc->qs.v.iv < supersetsize; ++fc.v.qc->qs.v.iv) {
                if (fc.fo == formulaoperator::foqexists)
                    res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                if (fc.fo == formulaoperator::foqforall)
                    res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                // std::cout << fc.v.qc->qs.v.iv << " iv \n";
            }
            if (fc.fo == formulaoperator::foqexists)
                res.v.bv = !res.v.bv;
        }
        if (fc.fo == formulaoperator::foqsum || fc.fo == formulaoperator::foqproduct || fc.fo == formulaoperator::foqmin
            || fc.fo == formulaoperator::foqmax || fc.fo == formulaoperator::foqrange
            || fc.fo == formulaoperator::foqaverage)
        {
            res.t = mtcontinuous;
            double min = std::numeric_limits<double>::infinity();
            double max = 0;
            int count = 0;
            if (fc.fo == formulaoperator::foqsum)
                res.v.dv = 0;
            if (fc.fo == formulaoperator::foqproduct)
                res.v.dv = 1;;
            while (!supersetpos->ended() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
            {
                variables[i]->qs = supersetpos->getnext();
                valms c;
                if (criterion) {
                    c = eval(*criterion);
                    if (c.t == mtbool && !c.v.bv)
                        continue;
                    if (c.t != mtbool) {
                        std::cout << "Quantifier criterion requires boolean value\n";
                        exit(1);
                    }
                }
                count++;
                // std::cout << "variables t == " << variables[i]->qs.t << ", name == " << fc.v.qc->name <<  std::endl;
                // std::cout << "supersetpos ended == " << supersetpos->ended() << std::endl;
                // for (fc.v.qc->qs.v.iv = 0; (res.v.bv) && fc.v.qc->qs.v.iv < supersetsize; ++fc.v.qc->qs.v.iv) {
                auto v = eval(*fc.fcright);
                if (fc.fo == formulaoperator::foqrange || fc.fo == formulaoperator::foqmin
                    || fc.fo == formulaoperator::foqmax)
                    res.v.dv = 0;
                if (fc.fo == formulaoperator::foqsum || fc.fo == formulaoperator::foqrange
                    || fc.fo == formulaoperator::foqmin || fc.fo == formulaoperator::foqmax
                    || fc.fo == formulaoperator::foqaverage)
                {
                    switch (v.t)
                    {
                    case mtcontinuous:
                        res.v.dv += v.v.dv;
                        break;
                    case mtdiscrete:
                        res.v.dv += v.v.iv;
                        break;
                    case mtbool:
                        res.v.dv += v.v.bv ? 1 : 0;
                        break;
                    case mtset:
                        res.v.dv += v.seti->getsize();
                        break;
                    }
                }
                if (fc.fo == formulaoperator::foqrange || fc.fo == formulaoperator::foqmin
                    || fc.fo == formulaoperator::foqmax)
                {
                    min = res.v.dv < min ? res.v.dv : min;
                    max = res.v.dv > max ? res.v.dv : max;
                }
                if (fc.fo == formulaoperator::foqproduct)
                {
                    switch (v.t)
                    {
                    case mtcontinuous:
                        res.v.dv *= v.v.dv;
                        break;
                    case mtdiscrete:
                        res.v.dv *= v.v.iv;
                        break;
                    case mtbool:
                        res.v.dv *= v.v.bv ? 1 : 0;
                        break;
                    case mtset:
                        res.v.dv *= v.seti->getsize();
                        break;
                    }
                    // std::cout << fc.v.qc->qs.v.iv << " iv \n";
                }
            }
            if (fc.fo == formulaoperator::foqrange)
                res.v.dv = max - min;
            if (fc.fo == formulaoperator::foqaverage)
                res.v.dv = count > 0 ? res.v.dv / count : 0;
        }
        if (fc.fo == formulaoperator::foqtally || fc.fo == formulaoperator::foqcount)
        {
            res.t = mtdiscrete;
            res.v.iv = 0;
            while (!supersetpos->ended())
            {
                variables[i]->qs = supersetpos->getnext();
                valms c;
                if (criterion) {
                    c = eval(*criterion);
                    if (c.t == mtbool && !c.v.bv)
                        continue;
                    if (c.t != mtbool) {
                        std::cout << "Quantifier criterion requires boolean value\n";
                        exit(1);
                    }
                }
                auto v = eval(*fc.fcright);
                if (fc.fo == formulaoperator::foqcount)
                    switch (v.t)
                    {
                        case mtdiscrete:
                            res.v.iv += v.v.iv != 0 ? 1 : 0;
                            break;
                        case mtbool:
                            res.v.iv += v.v.bv ? 1 : 0;
                            break;
                        case mtcontinuous:
                            res.v.iv += abs(v.v.dv) > ABSCUTOFF ? 1 : 0;
                            break;
                        case mtset:
                        case mttuple:
                            res.v.iv += v.seti->getsize() > 0 ? 1 : 0;
                        break;
                    }
                if (fc.fo == formulaoperator::foqtally)
                    switch (v.t)
                    {
                case mtdiscrete:
                    res.v.iv += v.v.iv;
                        break;
                case mtbool:
                    res.v.iv += v.v.bv ? 1 : 0;
                        break;
                case mtcontinuous:
                    res.v.iv += (int)v.v.dv;
                        break;
                case mtset:
                case mttuple:
                    res.v.iv += v.seti->getsize();
                        break;
                    }
            }

        }

        if (fc.fo == formulaoperator::foqdupeset)
        {
            res.t = mtset;
            std::vector<valms> tot {};
            while (!supersetpos->ended())
            {
                variables[i]->qs = supersetpos->getnext();
                valms c;
                if (criterion) {
                    c = eval(*criterion);
                    if (c.t == mtbool && !c.v.bv)
                        continue;
                    if (c.t != mtbool) {
                        std::cout << "Quantifier criterion requires boolean value\n";
                        exit(1);
                    }
                }
                auto v = eval(*fc.fcright);
                tot.push_back(v);
            }
            res.seti = new setitrmodeone(tot);
        }

        if (fc.fo == formulaoperator::foqset)
        {
            res.t = mtset;
            std::vector<valms> tot {};
            while (!supersetpos->ended())
            {
                variables[i]->qs = supersetpos->getnext();
                valms c;
                if (criterion) {
                    c = eval(*criterion);
                    if (c.t == mtbool && !c.v.bv)
                        continue;
                    if (c.t != mtbool) {
                        std::cout << "Quantifier criterion requires boolean value\n";
                        exit(1);
                    }
                }
                auto v = eval(*fc.fcright);
                bool match = false;
                for (int i = 0; !match && i < tot.size(); i++)
                {
                    switch (tot[i].t)
                    {
                        case mtdiscrete:
                            switch (v.t)
                            {
                            case mtdiscrete:
                                match = match || v.v.iv == tot[i].v.iv;
                                break;
                            case mtbool:
                                match = match || v.v.bv == tot[i].v.iv;
                                break;
                            case mtcontinuous:
                                match = match || abs(v.v.dv - tot[i].v.iv) < ABSCUTOFF;
                                break;
                            }
                        break;
                    case mtbool:
                        switch (v.t)
                        {
                            case mtdiscrete:
                                match = (match || v.v.iv != 0) == tot[i].v.bv;
                                    break;
                            case mtbool:
                                match = match || v.v.bv == tot[i].v.bv;
                                    break;
                            case mtcontinuous:
                                match = match || ((abs(v.v.dv) >= ABSCUTOFF) == tot[i].v.bv);
                                    break;
                        }
                        break;
                    case mtcontinuous:
                        switch (v.t)
                        {
                            case mtdiscrete:
                                match = match || abs(v.v.iv - tot[i].v.dv) < ABSCUTOFF;
                                    break;
                            case mtbool:
                                match = match || (v.v.bv == (abs(tot[i].v.bv) >= ABSCUTOFF));
                                    break;
                            case mtcontinuous:
                                match = (match || abs(v.v.dv - tot[i].v.dv) < ABSCUTOFF);
                                    break;
                        }
                        break;
                    case mtset:
                        if (v.t == mtset || v.t == mttuple)
                        {
                            auto itr1 = tot[i].seti->getitrpos();
                            auto itr2 = v.seti->getitrpos();
                            match = match || (setsubseteq(itr1,itr2) && setsubseteq(itr2,itr1));
                            delete itr1;
                            delete itr2;
                        } else
                        {
                            std::cout << "Mismatched to type set in SET\n";
                        }
                        break;
                    case mttuple:
                        if (v.t == mttuple || v.t == mtset)
                        {
                            auto itr1 = tot[i].seti->getitrpos();
                            auto itr2 = v.seti->getitrpos();
                            match = match || tupleeq(itr1,itr2);
                            delete itr1;
                            delete itr2;
                        } else
                        {
                            std::cout << "Mismatched to type tuple in SET\n";
                        }
                        break;
                    }
                }
                if (!match)
                    tot.push_back(v);
            }

            res.seti = new setitrmodeone(tot);
        }

        if (fc.fo == formulaoperator::foqunion || fc.fo == formulaoperator::foqintersection
            || fc.fo == formulaoperator::foqdupeunion)
        {
            res.t = mtset;
            res.seti = nullptr;
            while (!supersetpos->ended())
            {
                variables[i]->qs = supersetpos->getnext();
                valms c;
                if (criterion) {
                    c = eval(*criterion);
                    if (c.t == mtbool && !c.v.bv)
                        continue;
                    if (c.t != mtbool) {
                        std::cout << "Quantifier criterion requires boolean value\n";
                        exit(1);
                    }
                }
                auto v = eval(*fc.fcright);
                switch (fc.fo)
                {
                case formulaoperator::foqunion:
                    res.seti = res.seti ? new setitrunion( v.seti, res.seti) : v.seti;
                    break;
                case formulaoperator::foqintersection:
                    res.seti = res.seti ? new setitrintersection( v.seti, res.seti) : v.seti;
                    break;
                case formulaoperator::foqdupeunion:
                    res.seti = res.seti ? new setitrdupeunion( v.seti, res.seti) : v.seti;
                    break;
                }
            }
            if (!res.seti)
                res.seti = new setitrint(-1);
        }


        delete ss;
        // break;


            // case mtpair: {
                // valms v = eval(*fc.v.qc->superset);
                // int maxint = -1;
                // switch (v.t) {
                    // case mtbool: maxint = (int)v.v.bv; break;
                    // case mtdiscrete: maxint = v.v.iv; break;
                    // case mtcontinuous: maxint = (int)v.v.dv; break;
                // }
                // auto superset = v.v.iset;
                // auto supersetsize = v.setsize;
                // bool* ss = nullptr;
                // if (maxint >= 0) {
                    // ss = (bool*)malloc((maxint*(maxint-1)/2)*sizeof(bool));
                    // memset(ss,true,(maxint*(maxint-1)/2)*sizeof(bool));
                    // superset = ss;
                    // supersetsize = maxint;
                // }

                // int i = lookup_variable(fc.v.qc->name, variables);
                // int gapidx = v.setsize-1;
                // int idx = gapidx;
                // for (variables[i]->qs.v.ip.i = 0; (res.v.bv) && variables[i]->qs.v.ip.i < supersetsize; ++variables[i]->qs.v.ip.i)
                // {
                    // for (variables[i]->qs.v.ip.j = variables[i]->qs.v.ip.i + 1; (res.v.bv) && variables[i]->qs.v.ip.j < supersetsize; ++variables[i]->qs.v.ip.j) {
                        // for (fc.v.qc->qs.v.iv = 0; (res.v.bv) && fc.v.qc->qs.v.iv < supersetsize; ++fc.v.qc->qs.v.iv) {
                        // if (superset[idx + variables[i]->qs.v.ip.i - variables[i]->qs.v.ip.j]) {
                            // if (fc.fo == formulaoperator::foqexists)
                                // res.v.bv = res.v.bv && !eval(*fc.fcright).v.bv;
                            // if (fc.fo == formulaoperator::foqforall)
                                // res.v.bv = res.v.bv && eval(*fc.fcright).v.bv;
                            // std::cout << fc.v.qc->qs.v.iv << " iv \n";
                        // }
                    // }
                    // --gapidx;
                    // idx = idx + gapidx;
                // }
                // if (fc.fo == formulaoperator::foqexists)
                    // res.v.bv = !res.v.bv;
                // delete ss;
                // break;
            // }




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
               // std::cout << "ps type " << f->v.v.t << " type " << ps.back().t << "seti type " << ps.back().seti->t << "\n";
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
        case mtset: res.v.bv = resright.seti->getsize() == 0;
            return res;
        }
    }


    // switch (res.t)
    // {
    // case mtbool:
    if (booleanops(fc.fo))
    {
        res.t = mtbool;
        switch(resright.t)
        {
        case mtbool:
            if ((fc.fo == formulaoperator::foand && resright.v.bv)
                || (fc.fo == formulaoperator::foor && !resright.v.bv)
                || (fc.fo == formulaoperator::foif && resright.v.bv)
                || (fc.fo == formulaoperator::foimplies && !resright.v.bv)
                || (fc.fo == formulaoperator::foxor)
                || (fc.fo == formulaoperator::foiff))
            {
                valms resleft = eval(*fc.fcleft);
                switch (resleft.t)
                {
                case mtbool: res.v.bv = eval2ary<bool,bool,bool>(resleft.v.bv,resright.v.bv,fc.fo);
                    break;
                case mtdiscrete: res.v.bv = eval2ary<bool,int,bool>(resleft.v.iv,resright.v.bv,fc.fo);
                    break;
                case mtcontinuous: res.v.bv = eval2ary<bool,double,bool>(resleft.v.dv,resright.v.bv,fc.fo);
                    break;
                }
            } else
            {
                switch (fc.fo)
                {
                case formulaoperator::foand: res.v.bv = false; break;
                case formulaoperator::foor: res.v.bv = true; break;
                case formulaoperator::foif: res.v.bv = true; break;
                case formulaoperator::foimplies: res.v.bv = true; break;
                }
            }
            break;
        case mtdiscrete:
            if ((fc.fo == formulaoperator::foand && resright.v.iv)
                || (fc.fo == formulaoperator::foor && !resright.v.iv)
                || (fc.fo == formulaoperator::foif && resright.v.iv)
                || (fc.fo == formulaoperator::foimplies && !resright.v.iv)
                || (fc.fo == formulaoperator::foxor)
                || (fc.fo == formulaoperator::foiff))
            {
                valms resleft = eval(*fc.fcleft);

                switch (resleft.t)
                {
                case mtbool: res.v.bv = eval2ary<bool,bool,int>(resleft.v.bv,resright.v.iv,fc.fo);
                    break;
                case mtdiscrete: res.v.bv = eval2ary<bool,int,int>(resleft.v.iv,resright.v.iv,fc.fo);
                    break;
                case mtcontinuous: res.v.bv = eval2ary<bool,double,int>(resleft.v.dv,resright.v.iv,fc.fo);
                    break;
                }
            } else
            {
                switch (fc.fo)
                {
                case formulaoperator::foand: res.v.bv = false; break;
                case formulaoperator::foor: res.v.bv = true; break;
                case formulaoperator::foif: res.v.bv = true; break;
                case formulaoperator::foimplies: res.v.bv = true; break;
                }
            }

            break;
        case mtcontinuous:
            if ((fc.fo == formulaoperator::foand && resright.v.dv)
                || (fc.fo == formulaoperator::foor && !resright.v.dv)
                || (fc.fo == formulaoperator::foif && resright.v.dv)
                || (fc.fo == formulaoperator::foimplies && !resright.v.dv)
                || (fc.fo == formulaoperator::foxor)
                || (fc.fo == formulaoperator::foiff))
            {
                valms resleft = eval(*fc.fcleft);

                switch (resleft.t)
                {
                case mtbool: res.v.bv = eval2ary<bool,bool,double>(resleft.v.bv,resright.v.dv,fc.fo);
                    break;
                case mtdiscrete: res.v.bv = eval2ary<bool,int,double>(resleft.v.iv,resright.v.dv,fc.fo);
                    break;
                case mtcontinuous: res.v.bv = eval2ary<bool,double,double>(resleft.v.dv,resright.v.dv,fc.fo);
                    break;
                }
            }
            else
                {
                    switch (fc.fo)
                    {
                    case formulaoperator::foand: res.v.bv = false; break;
                    case formulaoperator::foor: res.v.bv = true; break;
                    case formulaoperator::foif: res.v.bv = true; break;
                    case formulaoperator::foimplies: res.v.bv = true; break;
                    }

                }
            break;
        }
        return res;
    }


    valms resleft = eval(*fc.fcleft);

    res.t = fc.v.v.t;
    // if (!booleanops(fc.fo) && (res.t == mtbool))
        // res.t = mtcontinuous;
    if (equalityops(fc.fo))
    {
        res.t = mtbool;
        switch(resleft.t)
        {
            /*
        case mtpair:
            switch (resright.t)
            {
        case mtpair:
            {
                if (fc.fo == formulaoperator::foe || fc.fo == formulaoperator::fone)
                    res.v.bv = (resleft.v.p->i == resright.v.p->i
                                && resleft.v.p->j == resright.v.p->j) != (fc.fo == formulaoperator::fone);
                else
                    if (resleft.v.p->i == resright.v.p->i)
                        res.v.bv = eval2aryeq<valms,valms>(resleft.v.p->j, resright.v.p->j,fc.fo);
                    else
                        res.v.bv = eval2aryeq<valms,valms>(resleft.v.p->i, resright.v.p->i, fc.fo);
                break;
            }
                std::cout << "Error in evalformula::eval comparing int pair to non-int-pair\n";
            }
            break; */
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
        case mtset:
            if (resright.t == mtset || resright.t == mttuple)
            {
                res.v.bv = eval2aryseteq(resleft.seti,resright.seti,fc.fo);
            } else
            {
                std::cout << "Error in evalformula::eval comparing set type to non-set type\n";
            }
            break;
        case mttuple:
            if (resright.t == mttuple)
            {
                res.v.bv = eval2arytupleeq(resleft.seti,resright.seti,fc.fo);
            } else
                if (resright.t == mtset)
                {
                    res.v.bv = eval2aryseteq(resleft.seti,resright.seti,fc.fo);
                }
                else
                {
                    std::cout << "Error in evalformula::eval comparing tuple type to non-tuple type\n";
                }
            break;
        }

        return res;
    }





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
        res = res && isalnum(c);
    if (tok.size() > 0)
        res = res && isalpha(tok[0]);
    return res;
}

inline bool is_literal(std::string tok) {
    if ((tok.size() > 2 && tok[0] == '[' && tok[tok.size()-1] == ']'))
        return true;
    else
        return is_function(tok);
}

inline bool is_variable(std::string tok) {
    bool res = tok.size() > 0 && isalpha(tok[0]);
    for (int i = 1; res && (i < tok.size()); ++i) {
        res = res && isalnum(tok[i]);
    }
    return res;
}

inline std::string get_literal(std::string tok) {
    if (is_literal(tok))
        if (tok.size() > 2 && tok[0] == '[' && tok[tok.size()-1] == ']')
            return tok.substr(1,tok.size()-2);
        else
            return tok;
    return 0;
}


inline bool is_quantifier( std::string tok ) {
    for (auto q : operatorsmap)
        if (tok == q.first)
            return quantifierops(q.second);
    return false;
    // return tok == QUANTIFIER_FORALL || tok == QUANTIFIER_EXISTS;
}




inline std::vector<std::string> Shuntingyardalg( const std::vector<std::string>& components, const std::vector<int>& litnumps) {
    std::vector<std::string> output {};
    std::vector<std::string> operatorstack {};

    std::vector<bool> werevalues {};
    std::vector<int> argcount {};

    int n = 0;
    while( n < components.size()) {
        const std::string tok = components[n++];
        if (is_real(tok) || is_number(tok) || is_truth(tok))
        {
            output.insert(output.begin(),tok);
            if (!werevalues.empty())
            {
                werevalues.resize(werevalues.size() - 1);
                werevalues.push_back(true);
            }
            continue;
        }
        // if (is_number(tok)) {
            // output.insert(output.begin(),tok);
            // if (!werevalues.empty())
            // {
                // werevalues.resize(werevalues.size() - 1);
                // werevalues.push_back(true);
            // }
            // continue;
        // }
        if ((is_operator(tok) && !is_quantifier(tok)) || tok == "IN") {
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


        if (tok != "IN" && (is_literal(tok) || is_quantifier(tok) || is_function(tok) || is_variable(tok)))
        {
            // if (litnumps[get_literal(tok)] > 0)
            // {
            if (n < components.size())
                if (components[n] == "(")
                {
                    operatorstack.push_back(tok);
                    argcount.push_back(0);
                    if (!werevalues.empty())
                        werevalues[werevalues.size() - 1] = true;
                    werevalues.push_back(false);
                } else
                {
                    output.insert(output.begin(),tok);
                    if (!werevalues.empty())
                    {
                        werevalues.resize(werevalues.size() - 1);
                        werevalues.push_back(true);
                    }
                }
            else
            {
                output.insert(output.begin(),tok);
                if (!werevalues.empty())
                {
                    werevalues.resize(werevalues.size() - 1);
                    werevalues.push_back(true);
                }
            }
            // }
            // else
            // output.insert(output.begin(),tok);
            continue;
        }
        if (tok == ",") {
            if (!operatorstack.empty()) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "(") {
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (!operatorstack.empty())
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Mismatched parentheses\n";
                        break;
                    }
                }
            } else
            {
                std::cout << "Mismatched parentheses\n";
            }
            if (!werevalues.empty())
            {
                bool w;
                while (!werevalues.empty())
                {
                    w = werevalues[werevalues.size()-1];
                    werevalues.resize(werevalues.size() - 1);

                    if (w == true)
                    {
                        if (!argcount.empty())
                        {
                            int a = argcount[argcount.size()-1];
                            ++a;
                            argcount[argcount.size()-1] = a;
                        } else
                        {
                            std::cout << "Mismatched argcount\n";
                        }
                    } else
                        break;
                    break;
                }
                // if (w == false)
                    werevalues.push_back(false);
            } else
                werevalues.push_back(false);
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

                if (is_operator(ostok) && !is_quantifier(ostok)) {
                    continue;
                }
                if (is_function(ostok) || is_quantifier(ostok) || is_literal(ostok) || is_variable(ostok))
                {
                    int a = 0;
                    if (!argcount.empty())
                    {
                        a = argcount[argcount.size()-1];
                        argcount.resize(argcount.size()-1);
                    } else
                        std::cout << "Mismatched argcount\n";
                    bool w = false;
                    if (!werevalues.empty())
                    {
                        w = werevalues[werevalues.size()-1];
                        werevalues.resize(werevalues.size()-1);
                    } else
                        std::cout << "Mismatched werevalues (right paren branch)\n";
                    if (w == true)
                        ++a;
                    output.insert(output.begin(), std::to_string(a));
                    output.insert(output.begin(), SHUNTINGYARDVARIABLEARGUMENTKEY);
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
    const std::vector<std::string>& litnames,
    std::vector<qclass*>* variables,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs )
{
    while( pos+1 < q.size()) {
        ++pos;
        std::string tok = q[pos];
        if (is_quantifier(tok)) {

#ifdef QUANTIFIERMODE_PRECEDING
            auto qc = new qclass;
            qc->secondorder = false;
            // qc->eval( q, ++pos);
            int argcnt;
            if (q[++pos] == SHUNTINGYARDVARIABLEARGUMENTKEY)
            {
                argcnt = stoi(q[++pos]);
                if (argcnt != 3 && argcnt != 2)
                    std::cout << "Wrong number (" << argcnt << ") of arguments to a quantifier\n";
            }
            int pos2 = pos+1;
            int quantcount = 0;
            while (pos2+1 <= q.size() && (q[pos2] != "IN" || quantcount >= 1)) {
                quantcount += is_quantifier(q[pos2]) ? 1 : 0;
                quantcount -= q[pos2] == "IN" ? 1 : 0;
                ++pos2;
            }
            if (pos2 >= q.size()) {
                std::cout << "Quantifier not containing an 'IN'\n";
                exit(-1);
            }
            // --pos2;
            qc->superset = parseformulainternal(q, pos2, litnumps,littypes,litnames, variables,fnptrs);
            qc->name = q[++pos2];
            qc->criterion = nullptr;
            variables->push_back(qc);

            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes,litnames, variables,fnptrs);

            // if (pos+1 < pos1)
            if (argcnt == 3)
            {
                qc->criterion = parseformulainternal(q,pos, litnumps, littypes, litnames, variables, fnptrs);
            }
            // qc->qs.t = qc->superset->v.v.seti->t;

            formulaclass* fcleft = nullptr;
            formulaoperator o = lookupoperator(tok);
            formulavalue fv {};
            fv.qc = qc;
            auto fc = fccombine(fv,fcleft,fcright,o);
            pos = pos2;
            return fc;
#else
            // what follows needs to be revisited using argcnt aspect of Shunting Yard formula (variable arguments)
            auto qc = new qclass;
            qc->secondorder = false;
            qc->name = q[++pos];
            // qc->eval( q, ++pos);
            if (q[++pos] != "IN")
            {
                std::cout << "Missing 'IN' in quantifier expression\n";
                exit(-1);
            }

            qc->superset = parseformulainternal(q, pos, litnumps,littypes,variables,fnptrs);
            // qc->qs.t = qc->superset->v.v.seti->t;

            variables->push_back(qc);
            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes,variables,fnptrs);
            formulaclass* fcleft = nullptr;
            formulaoperator o = lookupoperator(tok);
            formulavalue fv {};
            fv.qc = qc;
            auto fc = fccombine(fv,fcleft,fcright,o);
            return fc;
#endif
        }
        if (is_operator(tok))
        {
            formulaoperator o = lookupoperator(tok);
            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes, litnames, variables,fnptrs);
            formulaclass* fcleft = nullptr;
            if (o != formulaoperator::fonot)
            {
                fcleft = parseformulainternal(q,pos,litnumps,littypes,litnames, variables,fnptrs);
            }
            if (fcright)
                return fccombine({},fcleft,fcright,o);
        }
        int v = lookup_variable(tok,*variables);
        if (v >= 0) {

            formulavalue fv {};
            formulaclass* fc;
            if (q[pos+1] == SHUNTINGYARDVARIABLEARGUMENTKEY)
            {
                pos += 2;
                if (q[pos] == "1")
                {
                    fv.v.t = (*variables)[v]->qs.t;
                    fv.v = (*variables)[v]->qs;

                    fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariable);
                    fc->v.vs.name = tok;
                    fc->v.vs.ps.clear();
                    fc->v.vs.ps.push_back(parseformulainternal(q,pos,litnumps,littypes, litnames, variables,fnptrs));
                    fc->v.qc = (*variables)[v];
                } else
                {
                    if (q[pos] != "1")
                        std::cout << "More than one parameter used to index into a set or tuple\n";
                }
            } else {
                fv.v.t = (*variables)[v]->qs.t;
                fv.v = (*variables)[v]->qs;
                fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariable);
                fc->v.vs.name = tok;
                fc->v.vs.ps.clear();
                fc->v.qc = (*variables)[v];
            }

            return fc;
        }
        // if (tok == "V" || tok == "E" || tok == "NE") {
            // formulavalue fv {};
            // fv.v.t = mtset;
            // auto fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariable);
            // fc->v.qc = new qclass;
            // fc->v.qc->name = tok;
            // return fc;
        // }
        if (is_truth(tok))
        {
            formulavalue fv {};
            fv.v.t = mtbool;
            formulaoperator t = lookuptruth(tok);
            fv.v.v.bv = t == formulaoperator::fotrue;
            return fccombine(fv,nullptr,nullptr,t);
        }
        bool literal = false;
        int i;
        std::string potentialliteral {};
        if (is_literal(tok) || is_function(tok))
        {
            potentialliteral = get_literal(tok);
            for (i = 0; i < litnames.size() && !literal; ++i )
                 literal = literal || litnames[i] == potentialliteral;
            --i;
        }

        if (literal)
        {
            formulavalue fv {};
            fv.lit.lname = potentialliteral;
            fv.lit.l = i;
            fv.lit.ps.clear();
            fv.v.t = littypes[fv.lit.l];
            if (litnumps[fv.lit.l] == 0)
                return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
            else
            {
                if (q[pos+1] == SHUNTINGYARDVARIABLEARGUMENTKEY)
                {
                    int argcnt = stoi(q[pos+2]);
                    pos += 2;
                    if (argcnt != litnumps[fv.lit.l])
                    {
                        std::cout << "Literal expects " << litnumps[fv.lit.l] << " parameters, not " << argcnt << "parameters.\n";
                        return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
                    };
                    std::vector<formulaclass*> psrev {};
                    for (int i = 0; i < argcnt; ++i) {
                        psrev.push_back(parseformulainternal(q,pos,litnumps,littypes,litnames, variables,fnptrs));
                    }
                    for (int i = psrev.size()-1; i >= 0; --i)
                        fv.lit.ps.push_back(psrev[i]);
                    return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
                } else
                {
                    std::cout << "Error: parameterized literal has no parameters\n";
                    return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);

                }
            }

        }

        if (is_function(tok)) {
            std::vector<formulaclass*> ps {};
            // double (*f2)(std::vector<double>&);
            int argcnt = 0;
            for (auto f : *fnptrs)
                if (f.first == tok)
                    argcnt = f.second.second;
            std::vector<formulaclass*> psrev {};
            if (q[pos+1] == SHUNTINGYARDVARIABLEARGUMENTKEY)
                if (stoi(q[pos+2]) == argcnt)
                {
                    pos += 2;
                    for (int i = 0; i < argcnt; ++i) {
                        psrev.push_back(parseformulainternal(q,pos,litnumps,littypes,litnames, variables,fnptrs));
                    }
                    for (int i = psrev.size()-1; i >= 0; --i)
                        ps.push_back(psrev[i]);
                    formulavalue fv {};
                    if (auto search = fnptrs->find(tok); search != fnptrs->end())
                    {
                        fv.fns.fn = search->second.first;
                        fv.fns.ps = ps;
                        fv.v.t = mtcontinuous;
                        return fccombine(fv,nullptr,nullptr,formulaoperator::fofunction);
                    } else
                    {
                        std::cout << "Unknown function " << tok << " in parseformula internal\n";
                    }
                } else
                {
                    std::cout << "Error in Shuntingyard around function arguments\n";
                }
            else
            {
                std::cout << "Error in Shuntingyard around function arguments\n";
            }
            formulavalue fv {};
            return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
        }


        if (is_number(tok)) { // integer
            formulavalue fv {};
            fv.v.v.iv = stoi(tok);
            fv.v.t = mtdiscrete;
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }
        if (is_real(tok))
        {
            formulavalue fv {};
            fv.v.v.dv = stof(tok);
            fv.v.t = mtcontinuous;
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }
    }
    std::cout << "Error in parsing formula \n";
    exit(1);
    auto fc = new formulaclass({},nullptr,nullptr,formulaoperator::foconstant);
    return fc;
}


formulaclass* parseformula(
    const std::string& sentence,
    const std::vector<int>& litnumps,
    const std::vector<measuretype>& littypes,
    const std::vector<std::string>& litnames,
    std::vector<qclass*>& variables,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs )
{
    if (sentence != "") {
        variables.clear();
        std::vector<std::string> c = parsecomponents(sentence);
        
        std::vector<std::string> components = Shuntingyardalg(c,litnumps);
        int pos = -1;
        return parseformulainternal( components,pos, litnumps, littypes, litnames, &variables,fnptrs);
    } else {
        auto fc = new formulaclass({},nullptr,nullptr,formulaoperator::foconstant);
        return fc;
    }
}

