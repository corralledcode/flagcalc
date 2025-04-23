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
//#include <oneapi/tbb/detail/_task.h>

#include "feature.h"
#include "graphs.h"

#define QUANTIFIERMODE_PRECEDING

#define SHUNTINGYARDVARIABLEARGUMENTKEY "arg#"
// the random digits below are to prevent any chance occurrence of someone using those function names and it matching
// even though for now function names cannot inlude an underscore
#define SHUNTINGYARDINLINESETKEY "_INLINESET7903822160"
#define SHUNTINGYARDINLINETUPLEKEY "_INLINETUPLE7903822160"
#define SHUNTINGYARDDEREFKEY "_DEREF7903822160"

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

inline bool is_string(const std::string& s)
{
    if (s.size() >= 2)
        if (s[0] == '"' && s[s.size()-1] == '"')
            return true;
    return false;
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
    bool instring = false;
    for (auto ch : str) {
        if (instring)
        {
            instring = ch != '"' || (ch == '"' && partial.size() > 0 && partial[partial.size()-1] == '\\');
            partial += ch;
            if (!instring)
            {
                components.push_back(partial);
                partial = "";
            }
            continue;
        }
        if (ch == ' ') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            continue;
        }
        if (ch == '{')
        {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("{");
            continue;
        }
        if (ch == '}') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("}");
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
                // bool keyword = false;
                // for (auto k : operatorsmap)
                    // keyword = keyword || k.first == components[components.size()-1];
                // if (!keyword && components[components.size()-1] != "(") {
                    if (partial != "") {
                        components.push_back(partial);
                        partial = "";
                    }
                    if (components[components.size()-1] == "(")
                        partial = "-";
                    else
                        components.push_back("-");
                    continue;
                // }
            } else
            {
                partial = "-";
                continue;
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
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("[");
            continue;
        }
        if (ch == ']') {
            if (partial != "") {
                components.push_back(partial);
                partial = "";
            }
            components.push_back("]");
            continue;
        }

        /*
        if (ch == '[') {
            bracketed = true;
        }
        if (ch == ']') {
            bracketed = false;
            partial += ']';
            components.push_back(partial);
            partial = "";
            continue;
        }*/
        if (ch == '<')
        {
            if (partial == "<")
            {
                components.push_back( "<<");
                partial = "";
                continue;
            }
            if (partial != "")
            {
                components.push_back(partial);
            }
            partial = "<";
            continue;
        }
        if (ch == '>')
        {
            if (partial == ">")
            {
                components.push_back( ">>");
                partial = "";
                continue;
            }
            if (partial != "")
            {
                components.push_back(partial);
            }
            partial = ">";
            continue;
        }
        if (ch == '"')
        {
            if (partial != "")
                components.push_back(partial);
            partial = "\"";
            instring = true;
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
            || fo == formulaoperator::fone
            || fo == formulaoperator::fomeet
            || fo == formulaoperator::fodisjoint);
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
            || fo == formulaoperator::foqtuple
            || fo == formulaoperator::foqdupeset
            || fo == formulaoperator::foqunion
            || fo == formulaoperator::foqdupeunion
            || fo == formulaoperator::foqintersection);
}

template<typename T, typename T1, typename T2>
T eval2ary(const T1 in1, const T2 in2, const formulaoperator fo)
{
    T res;
    switch (fo)
    {
    case(formulaoperator::foplus):
        res = in1 + in2;
        break;
    case (formulaoperator::fominus):
        res = in1 - in2;
        break;
    case (formulaoperator::fotimes):
        res = in1 * in2;
        break;
    case(formulaoperator::fodivide):
        res = (T)in1 / (T)in2;
        break;
    case(formulaoperator::foexponent):
        res = pow(in1, in2);
        break;
    case(formulaoperator::fomodulus):
        res = (int)in1 % (int)in2;
        break;
    case(formulaoperator::foand):
        res = in1 && in2;
        break;
    case(formulaoperator::foor):
        res = in1 || in2;
        break;
    case(formulaoperator::foxor):
        res = in1 != in2;
        break;
    case(formulaoperator::foimplies):
        res = (!in1) || in2;
        break;
    case(formulaoperator::foif):
        res = in1 || (!in2);
        break;
    case(formulaoperator::foiff):
        res = in1 == in2;
        break;
    }
    return res;
}

template<typename T1, typename T2>
bool eval2aryeq( const T1 in1, const T2 in2, const formulaoperator fo)
{
    bool res;
    switch (fo) {
    case(formulaoperator::foe):
        res = abs(in1 - in2) < ABSCUTOFF;
        break;
    case(formulaoperator::folte):
        res = in1 <= in2;
        break;
    case(formulaoperator::folt):
        res = in1 < in2;
        break;
    case(formulaoperator::fogte):
        res = in1 >= in2;
        break;
    case (formulaoperator::fogt):
        res = in1 > in2;
        break;
    case(formulaoperator::fone):
        res = abs(in1 - in2) >= ABSCUTOFF;
        break;
    }
    return res;
}


bool setsubseteq( itrpos* in1, itrpos* in2)
{
    bool res = true;
    in1->reset();
    while (!in1->ended() && res)
    {
        auto itm = in1->getnext();
        // res = res && in2->parent->iselt(itm);
        in2->reset();

        bool match = false;
        while (!match && !in2->ended())
        {
            auto itm2 = in2->getnext();
            match = match || mtareequal( itm, itm2);
        }

        /*
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
        }*/
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
    auto abstractsetops = getsetitrops(in1, in2);
    bool res = abstractsetops->boolsetops( fo );
    delete abstractsetops;
    return res;

    /*
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
    return res;*/
}


bool eval2arytupleeq( setitr* in1, setitr* in2, const formulaoperator fo )
{

    auto abstracttupleops = gettupleops(in1, in2);
    bool res = abstracttupleops->boolsetops( fo );
    delete abstracttupleops;
    return res;



/*    bool res;
    auto pos1 = in1->getitrpos();
    auto pos2 = in2->getitrpos();
    if (fo == formulaoperator::foe) {
        res = tupleeq( pos1, pos2);
    } */


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
/*
    if (fo == formulaoperator::fone)
    {
        res = !tupleeq( pos1, pos2);
    }
    return res;
    */
}




valms evalformula::evalpslit( const int idx, namedparams& context, neighborstype* subgraph, params& ps )
{
    std::cout << "Error: evalpslit ancestor abstract called\n";
    valms res;
    res.t = measuretype::mtbool;
    res.v.bv = false;
    return res;
}



valms evalformula::evalvariable( const variablestruct& v, const namedparams& context, const std::vector<int>& vidxin ) {

    // if (v.l < 0)
    //    v.l = lookup_variable(v.name,context);
    int temp = lookup_variable(v.name,context);
    valms res;
    if (temp < 0)
    {
        res.t = mtdiscrete;
        res.v.iv = 0;
        std::cout << "Unknown variable " << v.name << std::endl;
    } else
    {
        if (vidxin.size() == 0)
            res = context[temp].second;
        else {
            if (vidxin.size() != 1)
            {
                std::cout << "Wrong number of parameters passed to de-index a variable\n";
            } else
            {
                int index = vidxin[0];
                auto pos = context[temp].second.seti->getitrpos();
                int i = 0;
                if (!pos->ended())
                    while (!pos->ended() && i++ <= index)
                        res = pos->getnext();
                else
                {
                    std::cout << "Indexing into the empty set\n";
                    res.t = mtdiscrete;
                    res.v.iv = 0;
                }
            }
        }

    }
    return res;
}

valms evalformula::eval( const formulaclass& fc, const namedparams& context) {}


valms evalmformula::evalinternal( const formulaclass& fc, namedparams& context )
{
    valms res;
    if (fc.fo == formulaoperator::foliteral) {
        if (fc.v.lit.ps.empty())
        {
            if (fc.v.lit.l >= 0 && fc.v.lit.l < literals.size()) {
                res = literals[fc.v.lit.l];
            } else {
                if (fc.v.lit.l < 0 && ((int)literals.size() + fc.v.lit.l >= 0))
                {
                    res = literals[literals.size() + fc.v.lit.l];
                }
                else {
                    std::cout << "Error eval'ing formula\n";
                    exit(1);
                    return res;
                }
            }
        } else {
            std::vector<valms> ps {};
            int i = 0;
            for (auto f : fc.v.lit.ps) {
                ps.push_back(evalinternal(*f, context));
                // std::cout << "ps type " << f->v.v.t << " type " << ps.back().t << "seti type " << ps.back().seti->t << "\n";
                // std::cout << fc.v.lit.ps[i++]->v.qc->qs.t << "\n";
            }
            neighborstype* subgraph {};
            if (fc.v.subgraph)
            {
//                subgraph = ps[0].v.nsv;
                subgraph = ps[0].v.nsv;
                // ps.erase(ps.begin());
                ps.erase(ps.begin());
            }
            res = evalpslit(fc.v.lit.l, context, subgraph, ps);
        }
        return res;
    }

    switch (fc.fo) {
    case formulaoperator::foconstant:
        {
            res.t = fc.v.v.t;
            switch (res.t)
            {
            case mtbool: res.v.bv = fc.v.v.v.bv;
                return res;
            case mtdiscrete: res.v.iv = fc.v.v.v.iv;
                return res;
            case mtcontinuous: res.v.dv = fc.v.v.v.dv;
                return res;
            case mtset:
            case mttuple:
                // std::vector<valms> tot {};
                // for (int i = 0; i < fc.v.ss.elts.size(); ++i)
                    // tot.push_back(evalinternal(*fc.v.ss.elts[i]));
                // res.seti = new setitrmodeone(tot);
                res.seti = new setitrformulae(rec, idx, fc.v.ss.elts, context ); // doesn't compute in time, that is, misses variables in a quantifier
                return res;
            case mtstring:
                res.v.rv = fc.v.v.v.rv;
                return res;
            case mtgraph:
                res.v.nsv = fc.v.v.v.nsv;
                return res;
            }
            res.t = mtbool;
            res.v.bv = false;
            return res;
        }

    case formulaoperator::fofunction: {
            bool found = false;
            double (*fn)(std::vector<double>&) = fc.v.fns.fn;

            std::vector<double> ps;
            ps.clear();
            for (auto f : fc.v.fns.ps) {
                auto a = evalinternal(*f,context);
                valms r;
                mtconverttocontinuous(a,r.v.dv);
/*                switch (a.t)
                {
                case mtbool: r.v.dv = (double)a.v.bv;
                    break;
                case mtdiscrete: r.v.dv = (double)a.v.iv;
                    break;
                case mtcontinuous: r.v.dv = a.v.dv;
                    break;
                }*/
                ps.push_back(r.v.dv);
            }
            res.v.dv = fn(ps);
            res.t = measuretype::mtcontinuous;
            return res;
        }

    case formulaoperator::foderef:
        {
            auto v = evalinternal(*fc.fcright,context);
            if (v.t != measuretype::mtset && v.t != measuretype::mttuple)
                std::cout << "Non-set or tuple variable being dereferenced\n";
            auto pos = v.seti->getitrpos();
            int i = 0;
            valms res;
            auto idx = evalinternal(*fc.fcleft,context);
            switch (idx.t) {
            case mtbool: idx.v.iv = idx.v.bv ? 1 : 0;
                break;
            case mtcontinuous:
                    idx.v.iv = (int)(idx.v.dv);
                break;
            }
            if (!pos->ended())
                while (i++ <= idx.v.iv && !pos->ended())
                    res = pos->getnext();
            else {
                std::cout << "Dereferencing beyond end of set\n";
                res.t = measuretype::mtdiscrete;
                res.v.iv = 0;
            }
            if (i <= idx.v.iv)
            {
                std::cout << "Dereferencing beyond end of set\n";
                res.t = measuretype::mtdiscrete;
                res.v.iv = 0;
            }
            delete pos;
            return res;
        }
    case formulaoperator::fovariable: {
            if (fc.v.vs.ps.empty())
            {
                std::vector<int> ps {};
                res = evalvariable(fc.v.vs, context, ps);
            } else {
                std::vector<int> ps {};
                for (auto f : fc.v.vs.ps) {
                    ps.push_back(evalinternal(*f, context).v.iv);
                    // std::cout << "ps type " << f->v.v.t << " type " << ps.back().t << "seti type " << ps.back().seti->t << "\n";
                }
                res = evalvariable(fc.v.vs, context,ps);
            }
            return res;
        }

    case (formulaoperator::foelt):
        {
            res.t = measuretype::mtbool;
            valms set = evalinternal(*fc.fcright,context );
            valms itm = evalinternal( *fc.fcleft,context );


            if (set.t == mtset || set.t == mttuple)
            {
//                res.v.bv = set.seti->iselt(itm);
                auto pos = set.seti->getitrpos();
                pos->reset();
                bool match = false;
                while ( !match && !pos->ended())
                {
                    auto v = pos->getnext();
//                    match = match || mtareequal(v,itm);
                    match = match || mtareequalgenerous(itm,v);

/*
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
                            }*/
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
            } else {
                res.v.bv = false;
                std::cout << "Non-matching types in use of ELT, " << itm.t << ", " << set.t << "\n";
                exit(1);
                return res;
            }
        }
    case formulaoperator::founion:
    case formulaoperator::fointersection:
    case formulaoperator::fodupeunion:
    case formulaoperator::fosetminus:
    case formulaoperator::fosetxor:
        {
            valms set1 = evalinternal(*fc.fcleft, context );
            valms set2 = evalinternal( *fc.fcright, context );
            if (set2.t == mtset || set2.t == mttuple)
                switch (set1.t) {
                    case mtset: {
                        auto abstractsetops = getsetitrops(set1.seti, set2.seti);
                        res.seti = abstractsetops->setops(fc.fo);
                        res.t = measuretype::mtset;
                        delete abstractsetops;
                        break;}
                    case mttuple: {
                        auto abstracttupleops = gettupleops(set1.seti, set2.seti);
                        res.seti = abstracttupleops->setops(fc.fo);
                        res.t = measuretype::mttuple;
                        delete abstracttupleops;
                        break;}
                    default:
                        std::cout << "Non-matching types in call to CUP, CAP, CUPD, SETMINUS, or SETXOR\n";
                        res.seti = nullptr;
                        break;
                }
            else {
                std::cout << "Non-matching types in call to CUP, CAP, CUPD, SETMINUS, or SETXOR\n";
                res.seti = nullptr;
            }
/*            if ((set1.t == mtset || set1.t == mttuple) && (set2.t == mtset || set2.t == mttuple))
            {
                auto abstractsetops = getsetitrops(set1.seti, set2.seti);
                res.seti = abstractsetops->setops(fc.fo);


                switch (fc.fo)
                {
                case formulaoperator::founion:
                    res.seti = new setitrunion(set1.seti,set2.seti);
                    break;
                case formulaoperator::fointersection:
                    res.seti = new setitrintersection(set1.seti,set2.seti);
                    break;
                case formulaoperator::fodupeunion:
                    res.seti = new setitrdupeunion( set1.seti,set2.seti);
                    break;
                case formulaoperator::fosetminus:
                    res.seti = new setitrsetminus( set1.seti,set2.seti);
                    break;
                case formulaoperator::fosetxor:
                    res.seti = new setitrsetxor( set1.seti,set2.seti);
                    break;
                }*/

            return res;
        }
    case formulaoperator::fotrue:
        {
            res.t = measuretype::mtbool;
            res.v.bv = true;
            return res;
        }
    case formulaoperator::fofalse:
        {
            res.t = measuretype::mtbool;
            res.v.bv = false;
            return res;
        }
    }

    if (fc.fo == formulaoperator::fonaming)
    {
        valms v = evalinternal(*fc.fcright->boundvariable->superset, context);
        context.push_back({fc.fcright->boundvariable->name,v});
        res = evalinternal(*fc.fcright, context);
        context.resize(context.size()-1);
        return res;
    }

    // if (quantifierops.find(fc.fo)->second) {
    if (quantifierops(fc.fo)) {
        res.v.bv = true;

        res.t = mtbool;
        valms v = evalinternal(*fc.fcright->boundvariable->superset, context);
        // valms v = evalinternal(*fc.v.qc->superset);
        bool needtodeletevseti = false;
        if (v.t == mtdiscrete || v.t == mtcontinuous)
        {
            if (v.t == mtcontinuous)
                v.v.iv = (int)v.v.dv;
            if (v.v.iv >= 0)
                v.seti = new setitrint(v.v.iv-1);
            else
                v.seti = new setitrint(-1);
            v.t = mtset;
            needtodeletevseti = true;
        } else
        if (v.t == mtbool)
            std::cout << "Cannot use mtbool for quantifier superset\n";
        auto criterion = fc.v.qc->criterion;
        int maxint = -1;
        auto supersetpos = v.seti->getitrpos();
        setitrint* ss = nullptr;
        context.push_back({fc.fcright->boundvariable->name,fc.fcright->boundvariable->qs});
        const int i = context.size()-1;
        supersetpos->reset();
        switch (fc.fo)
        {
        case (formulaoperator::foqexists):
        case (formulaoperator::foqforall):
            {
                while (!supersetpos->ended() && res.v.bv) {
                    context[i].second = supersetpos->getnext();
                    valms c;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t == mtbool && !c.v.bv)
                            continue;
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (fc.fo == formulaoperator::foqexists)
                        res.v.bv = res.v.bv && !evalinternal(*fc.fcright, context).v.bv;
                    else //if (fc.fo == formulaoperator::foqforall)
                        res.v.bv = res.v.bv && evalinternal(*fc.fcright, context).v.bv;
                    // std::cout << fc.v.qc->qs.v.iv << " iv \n";
                }
                if (fc.fo == formulaoperator::foqexists)
                    res.v.bv = !res.v.bv;
                break;
            }
        case (formulaoperator::foqsum):
        case (formulaoperator::foqproduct):
        case (formulaoperator::foqmin):
        case (formulaoperator::foqmax):
        case (formulaoperator::foqrange):
        case (formulaoperator::foqaverage):
            {
                res.t = mtcontinuous;
                double min = std::numeric_limits<double>::infinity();
                double max = -std::numeric_limits<double>::infinity();
                int count = 0; // this is not foqcount but rather to be used to find average
                if (fc.fo == formulaoperator::foqsum)
                    res.v.dv = 0;
                else if (fc.fo == formulaoperator::foqproduct)
                    res.v.dv = 1;;
                while (!supersetpos->ended() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                {
                    context[i].second = supersetpos->getnext();
                    valms c;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t == mtbool && !c.v.bv)
                            continue;
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    count++;  // this is not foqcount but rather to be used to find average
                    // std::cout << "variables t == " << variables[i]->qs.t << ", name == " << fc.v.qc->name <<  std::endl;
                    // std::cout << "supersetpos ended == " << supersetpos->ended() << std::endl;
                    // for (fc.v.qc->qs.v.iv = 0; (res.v.bv) && fc.v.qc->qs.v.iv < supersetsize; ++fc.v.qc->qs.v.iv) {
                    auto v = evalinternal(*fc.fcright, context);
                    if (fc.fo != formulaoperator::foqproduct)   // formulaoperator::foqsum || fc.fo == formulaoperator::foqrange
                        // || fc.fo == formulaoperator::foqmin || fc.fo == formulaoperator::foqmax
                            // || fc.fo == formulaoperator::foqaverage)
                    {
                        if (fc.fo == formulaoperator::foqrange || fc.fo == formulaoperator::foqmin
                            || fc.fo == formulaoperator::foqmax)
                            res.v.dv = 0;
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

                        if (fc.fo == formulaoperator::foqrange || fc.fo == formulaoperator::foqmin
                            || fc.fo == formulaoperator::foqmax)
                        {
                            if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                min = res.v.dv;
                            else
                                min = res.v.dv < min ? res.v.dv : min;
                            if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                max = res.v.dv;
                            else
                                max = res.v.dv > max ? res.v.dv : max;
                        }
                    } else
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
                            case mttuple:
                                res.v.dv *= v.seti->getsize();
                                break;
                            }
                            // std::cout << fc.v.qc->qs.v.iv << " iv \n";
                        }

                }
                switch (fc.fo)
                {
                case(formulaoperator::foqrange):
                    res.v.dv = max - min;
                    break;
                case (formulaoperator::foqaverage):
                    res.v.dv = count > 0 ? res.v.dv / count : 0;
                    break;
                case (formulaoperator::foqmax):
                    res.v.dv = max;
                    break;
                case (formulaoperator::foqmin):
                    res.v.dv = min;
                    break;
                }
                break;
            }
        case formulaoperator::foqtally:
        case formulaoperator::foqcount:
            {
                res.t = mtdiscrete;
                res.v.iv = 0;
                while (!supersetpos->ended())
                {
                    context[i].second = supersetpos->getnext();
                    // variables[i]->qs = supersetpos->getnext();
                    valms c;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t == mtbool && !c.v.bv)
                            continue;
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    auto v = evalinternal(*fc.fcright, context);
                    if (fc.fo == formulaoperator::foqcount) {
                        valms tmp;
                        mtconverttobool(v,tmp.v.bv);
                        if (tmp.v.bv)
                            ++res.v.iv;
                    } else  if (fc.fo == formulaoperator::foqtally)
                    {
                        valms tmp;
                        mtconverttodiscrete(v, tmp.v.iv);
                        res.v.iv += tmp.v.iv;
                    }

                }
                break;
            }

        case (formulaoperator::foqdupeset):
        case (formulaoperator::foqtuple):
            {
                res.t = fc.fo == formulaoperator::foqdupeset ? mtset : mttuple;
                std::vector<valms> tot {};
                while (!supersetpos->ended())
                {
                    context[i].second = supersetpos->getnext();
                    // variables[i]->qs = supersetpos->getnext();
                    valms c;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t == mtbool && !c.v.bv)
                            continue;
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    auto v = evalinternal(*fc.fcright, context);
                    tot.push_back(v);
                }
                res.seti = new setitrmodeone(tot);
                break;
            }
        case (formulaoperator::foqset):
            {
                res.t = mtset;
                std::vector<valms> tot {};
                while (!supersetpos->ended())
                {
                    context[i].second = supersetpos->getnext();
                    // variables[i]->qs = supersetpos->getnext();
                    valms c;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t == mtbool && !c.v.bv)
                            continue;
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    auto v = evalinternal(*fc.fcright, context);
                    bool match = false;
                    for (int i = 0; !match && i < tot.size(); i++)
                    {
                        match = match || mtareequal(tot[i], v);

                       /*
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
                        }*/
                    }
                    if (!match)
                        tot.push_back(v);
                }

                res.seti = new setitrmodeone(tot);
                break;
            }


        case (formulaoperator::foqunion):
        case (formulaoperator::foqintersection):
        case (formulaoperator::foqdupeunion):
            {
                res.t = mtset;
                res.seti = nullptr;
                std::vector<setitr*> composite {};
                while (!supersetpos->ended())
                {
                    context[i].second = supersetpos->getnext();
                    // variables[i]->qs = supersetpos->getnext();
                    valms c;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t == mtbool && !c.v.bv)
                            continue;
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    valms tempv;
                    valms outv;
                    tempv = evalinternal(*fc.fcright, context);
                    mtconverttoset(tempv,outv.seti);
                    composite.push_back(outv.seti);
                }

                auto abstractpluralsetops = getsetitrpluralops(composite);

                res.seti = abstractpluralsetops->setops(fc.fo);


/*                    switch (fc.fo)
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
                    } */
                }
                if (!res.seti)
                    res.seti = new setitrint(-1);
                break;
            }

        delete ss;

        if (needtodeletevseti)
            delete v.seti;
        context.resize(i);

        return res;
    }


    if (fc.fo == formulaoperator::foswitch)
    {
        valms resleft = evalinternal(*fc.fcleft, context);
        bool b = false;
        switch (resleft.t)
        {
        case mtbool: b = resleft.v.bv; break;
        case mtcontinuous: b = !(abs(resleft.v.dv) < ABSCUTOFF); break;
        case mtdiscrete: b = resleft.v.iv != 0; break;
        case mtset:
        case mttuple:
            auto itr = resleft.seti->getitrpos();
            b = !itr->ended();
            delete itr;
            break;
        // case string:
        }
        if (b)
        {
            if (fc.fcright->fo  == formulaoperator::focases)
                res = evalinternal(*fc.fcright->fcleft, context);
            else
            {
                std::cout << "Expecting ':' after '?'\n";
                valms v;
                v.t = mtbool;
                v.v.bv = false;
                res = v;
            }
        } else
        {
            if (fc.fcright->fo  == formulaoperator::focases)
                res = evalinternal(*fc.fcright->fcright, context);
            else
            {
                std::cout << "Expecting ':' after '?'\n";
                valms v;
                v.t = mtbool;
                v.v.bv = false;
                res = v;
            }
        }
        return res;
    }


    valms resright = evalinternal(*fc.fcright, context);

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
    // if (booleanops.find(fc.fo)->second)
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
                valms resleft = evalinternal(*fc.fcleft, context);
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
                valms resleft = evalinternal(*fc.fcleft, context);

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
                valms resleft = evalinternal(*fc.fcleft, context);

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

    valms resleft = evalinternal(*fc.fcleft, context);

    res.t = fc.v.v.t;
    if (equalityops(fc.fo))
    // if (equalityops.find(fc.fo)->second)
    {
        res.t = mtbool;
        switch(resleft.t)
        {
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

inline valms evalmformula::eval( const formulaclass& fc, const namedparams& context )
{
    // if (!fc.v.subgraph) {
    if (idx >= 0)
    {
        literals.resize(rec->literals.size());
        for (int i = 0; i < rec->literals.size(); ++i)
            literals[i] = rec->literals[i][idx];
    } else
    {
        literals.clear();
    }
    // } else
    // { // subgraph case
        // literals.clear();
    // }
    namedparams contextlocal = context;
    return evalinternal( fc, contextlocal );
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
        res = res && (isalnum(c) || c == '_');
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
        res = res && (isalnum(tok[i]) || tok[i] == '_');
    }
    return res;
}

inline std::string get_literal(std::string tok) {
    if (is_literal(tok))
        if (tok.size() > 2 && tok[0] == '[' && tok[tok.size()-1] == ']')
            return tok.substr(1,tok.size()-2);
        else
            return tok;
    return "";
}

inline bool is_quantifier( std::string tok ) {
    for (auto q : operatorsmap)
        if (tok == q.first)
            return quantifierops(q.second);
            // return quantifierops.find(q.second)->second;
    return false;
}

inline bool is_naming(std::string tok)
{
    for (auto q : operatorsmap)
        if (tok == q.first)
            return q.second == formulaoperator::fonaming;
    return false;
}






inline std::vector<std::string> Shuntingyardalg( const std::vector<std::string>& components, const std::vector<int>& litnumps) {
    std::vector<std::string> output {};
    std::vector<std::string> operatorstack {};

    std::vector<bool> werevalues {};
    std::vector<int> argcount {};


    int n = 0;
    while( n < components.size()) {
        const std::string tok = components[n++];
        if (is_real(tok) || is_number(tok) || is_truth(tok) || is_string(tok))
        {
            output.insert(output.begin(),tok);
            if (!werevalues.empty())
            {
                werevalues.resize(werevalues.size() - 1);
                werevalues.push_back(true);
            }
            continue;
        }
        if (is_operator(tok) && !is_quantifier(tok) && !is_naming(tok)) {
            if (operatorstack.size() >= 1) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while ((ostok != "(") && (ostok != "{") && ostok != "<<" && ostok != "[" && (operatorprecedence(ostok,tok) >= 0)) {
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
        if (is_literal(tok) || is_quantifier(tok) || is_function(tok) || is_variable(tok) || is_naming(tok))
        {
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
            continue;
        }

        if (tok == ",") {
            if (!operatorstack.empty()) {
                std::string ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "(" && ostok != "{" && ostok != "<<" && ostok != "[") {
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (!operatorstack.empty())
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Mismatched parentheses, curley braces, brackets, or tuple brackets\n";
                        break;
                    }
                }
            } else
            {
                std::cout << "Mismatched parentheses, curley braces, brackets, or tuple brackets (loc 2)\n";
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

        if (tok == "[") {
            operatorstack.push_back(SHUNTINGYARDDEREFKEY);
            operatorstack.push_back(tok);
            argcount.push_back(0);
            if (!werevalues.empty())
                werevalues[werevalues.size() - 1] = true;
            werevalues.push_back(false);
            continue;
        }

        if (tok == "{")
        {
            operatorstack.push_back(SHUNTINGYARDINLINESETKEY);
            operatorstack.push_back(tok);
            argcount.push_back(0);
            if (!werevalues.empty())
                werevalues[werevalues.size() - 1] = true;
            werevalues.push_back(false);
            continue;
        }

        if (tok == "}")
        {
            std::string ostok;
            if (operatorstack.size() >= 1)
            {
                ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "{") {
                    if (operatorstack.empty()) {
                        std::cout << "Error mismatched curley braces (loc 1)\n";
                        return output;
                    }
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (operatorstack.size() >= 1)
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Error mismatched curley braces (loc 2)\n";
                        return output;
                    }
                }
            }
            if (operatorstack.empty() || operatorstack[operatorstack.size()-1] != "{") {
                std::cout << "Error mismatched curley braces (loc 3)\n";
                return output;
            }
            operatorstack.resize(operatorstack.size()-1);


            if (operatorstack.size() > 0) {
                ostok = operatorstack[operatorstack.size()-1];

                // if (is_operator(ostok) && !is_quantifier(ostok)) {
                    // continue;
                // } else
                    if (ostok == SHUNTINGYARDINLINESETKEY)
                    // if (is_function(ostok) || is_quantifier(ostok) || is_literal(ostok) || is_variable(ostok))
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
                        // continue;
                    } else
                    {
                        std::cout << "Error in inline set\n";
                    }

            }
            if (n < components.size())
            {
                if (components[n] == "[") {
                    // operatorstack.push_back(tok);
                    operatorstack.push_back(SHUNTINGYARDDEREFKEY);
                    argcount.push_back(0);
                    if (!werevalues.empty())
                        werevalues[werevalues.size() - 1] = true;
                    werevalues.push_back(false);
                    // operatorstack.push_back("(");
                    // ++n;
                    continue;
                }
            }


            continue;
        }


        if (tok == "<<")
        {
            operatorstack.push_back(SHUNTINGYARDINLINETUPLEKEY);
            operatorstack.push_back(tok);
            argcount.push_back(0);
            if (!werevalues.empty())
                werevalues[werevalues.size() - 1] = true;
            werevalues.push_back(false);
            continue;
        }

        if (tok == ">>")
        {
            std::string ostok;
            if (operatorstack.size() >= 1)
            {
                ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "<<") {
                    if (operatorstack.empty()) {
                        std::cout << "Error mismatched tuple braces (loc 1)\n";
                        return output;
                    }
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (operatorstack.size() >= 1)
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Error mismatched tuple braces (loc 2)\n";
                        return output;
                    }
                }
            }
            if (operatorstack.empty() || operatorstack[operatorstack.size()-1] != "<<") {
                std::cout << "Error mismatched tuple braces (loc 3)\n";
                return output;
            }
            operatorstack.resize(operatorstack.size()-1);


            if (operatorstack.size() > 0) {
                ostok = operatorstack[operatorstack.size()-1];

                // if (is_operator(ostok) && !is_quantifier(ostok)) {
                    // continue;
                // } else
                    if (ostok == SHUNTINGYARDINLINETUPLEKEY)
                    // if (is_function(ostok) || is_quantifier(ostok) || is_literal(ostok) || is_variable(ostok))
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
                        // continue;
                    } else
                    {
                        std::cout << "Error in inline tuple\n";
                    }

            }
            if (n < components.size())
            {
                if (components[n] == "[") {
                    // operatorstack.push_back(tok);
                    operatorstack.push_back(SHUNTINGYARDDEREFKEY);
                    argcount.push_back(0);
                    if (!werevalues.empty())
                        werevalues[werevalues.size() - 1] = true;
                    werevalues.push_back(false);
                    // operatorstack.push_back("(");
                    // ++n;
                    continue;
                }
            }


            continue;
        }


        if (tok == "(") {
            operatorstack.push_back(tok);
            continue;
        }

        if (tok == "]")
        {
            std::string ostok;
            if (operatorstack.size() >= 1)
            {
                ostok = operatorstack[operatorstack.size()-1];
                while (ostok != "[") {
                    if (operatorstack.empty()) {
                        std::cout << "Error mismatched brackets (loc 1)\n";
                        return output;
                    }
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (operatorstack.size() >= 1)
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Error mismatched brackets (loc 2)\n";
                        return output;
                    }
                }
            }
            if (operatorstack.empty() || operatorstack[operatorstack.size()-1] != "[") {
                std::cout << "Error mismatched tuple brackets (loc 3)\n";
                return output;
            }
            operatorstack.resize(operatorstack.size()-1);


            if (operatorstack.size() > 0) {
                ostok = operatorstack[operatorstack.size()-1];

                // if (is_operator(ostok) && !is_quantifier(ostok)) {
                    // continue;
                // } else
                    if (ostok == SHUNTINGYARDDEREFKEY)
                    // if (is_function(ostok) || is_quantifier(ostok) || is_literal(ostok) || is_variable(ostok))
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
                            std::cout << "Mismatched werevalues (right bracket branch)\n";
                        if (w == true)
                            ++a;

                        output.insert(output.begin(), std::to_string(a));
                        output.insert(output.begin(), SHUNTINGYARDVARIABLEARGUMENTKEY);
                        output.insert(output.begin(),ostok);
                        operatorstack.resize(operatorstack.size()-1);
                        // continue;
                    } else
                    {
                        std::cout << "Error in brackets deref\n";
                    }

            }
            if (n < components.size())
            {
                if (components[n] == "[") {
                    // operatorstack.push_back(tok);
                    operatorstack.push_back(SHUNTINGYARDDEREFKEY);
                    argcount.push_back(0);
                    if (!werevalues.empty())
                        werevalues[werevalues.size() - 1] = true;
                    werevalues.push_back(false);
                    // operatorstack.push_back("(");
                    // ++n;
                    continue;
                }
            }


            continue;
        }




        if (tok == ")") {

            std::string ostok;
            if (!operatorstack.empty())
            {
                ostok = operatorstack[operatorstack.size()-1];
                while ((tok == ")" && ostok != "(") || (tok == "]" && ostok != "[")) {
                    if (operatorstack.empty()) {
                        std::cout << "Error mismatched parentheses or brackets (loc 1)\n";
                        return output;
                    }
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (!operatorstack.empty())
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Error mismatched parentheses or brackets (loc 2)\n";
                        for (auto o : output)
                            std::cout << o << ", ";
                        std::cout << "\n";
                        return output;
                    }
                }
            }
            if (operatorstack.empty() || (tok == ")" && operatorstack[operatorstack.size()-1] != "(") || (tok == "]" && operatorstack[operatorstack.size()-1] != "[")) {
                std::cout << "Error mismatched parentheses or brackets (loc 3)\n";
                for (auto t : operatorstack) {
                    std::cout << t << ", ";
                }
                std::cout << std::endl;

                for (auto o : output)
                    std::cout << o << ", ";
                std::cout << "\n";

                return output;
            }
            operatorstack.resize(operatorstack.size()-1);

            if (operatorstack.size() > 0)
            {
                ostok = operatorstack[operatorstack.size()-1];
                if (is_operator(ostok) && !is_quantifier(ostok) && !is_naming(ostok)) {
                    // continue;
                } else
                    if (is_function(ostok) || is_quantifier(ostok) || is_literal(ostok) || is_variable(ostok)
                        || is_naming(ostok) || ostok == SHUNTINGYARDDEREFKEY)
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
                            std::cout << "Mismatched werevalues (right paren/bracket branch)\n";
                        if (w == true)
                            ++a;

                        output.insert(output.begin(), std::to_string(a));
                        output.insert(output.begin(), SHUNTINGYARDVARIABLEARGUMENTKEY);
                        output.insert(output.begin(),ostok);
                        operatorstack.resize(operatorstack.size()-1);
                    }
                // continue;
            }

            if (n < components.size())
            {
                if (components[n] == "[" && tok == "]") {
                    // operatorstack.push_back(tok);
                    operatorstack.push_back(SHUNTINGYARDDEREFKEY);
                    argcount.push_back(0);
                    if (!werevalues.empty())
                        werevalues[werevalues.size() - 1] = true;
                    werevalues.push_back(false);
                    // operatorstack.push_back("(");
                    // ++n;
                    continue;
                }
            }
            continue;
        }

    }


    while (operatorstack.size()> 0) {
        std::string ostok = operatorstack[operatorstack.size()-1];
        if (ostok == "(" || ostok == "{" || ostok == "<<" || ostok == "[") {
            std::cout << "Error mismatched parentheses, curley braces, brackets, or tuple brackets: " << ostok << ", (loc 4)\n";

            for (auto t : operatorstack) {
                std::cout << t << ", ";
            }
            std::cout << std::endl;

            for (auto o : output)
                std::cout << o << ", ";
            std::cout << "\n";

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
    namedparams& ps,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs )
{
    while( pos+1 < q.size()) {
        ++pos;
        std::string tok = q[pos];

        if (tok == SHUNTINGYARDDEREFKEY) {
            ++pos; // skip over arg#
            int argcount = stoi(q[++pos]);
            if (argcount != 1) {
                std::cout << "No support for dereferencing a set or tuple with other than one variable input\n";
                exit(1);
            }
            formulavalue fv {};

            auto fcleft = parseformulainternal(q,pos,litnumps,littypes, litnames, ps, fnptrs);
            auto fcright = parseformulainternal(q,pos,litnumps,littypes, litnames, ps,  fnptrs);

            auto fc = fccombine(fv,fcleft,fcright, formulaoperator::foderef);

            return fc;
        }

        if (tok == SHUNTINGYARDINLINESETKEY || tok == SHUNTINGYARDINLINETUPLEKEY)
        {

            formulavalue fv {};
            fv.v.t = tok == SHUNTINGYARDINLINESETKEY ? mtset : mttuple;
            int argcount = 0;
            if (pos + 2 < q.size() && q[++pos] == SHUNTINGYARDVARIABLEARGUMENTKEY)
            {
                argcount = stoi(q[++pos]);
            }

            fv.ss.elts.clear();
            fv.ss.elts.resize(argcount);
            for (int i = 0; i < argcount; ++i)
                fv.ss.elts[argcount - i - 1] = parseformulainternal(q,pos,litnumps,littypes,litnames, ps,  fnptrs);

            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);

        }

        if (is_naming(tok))
        {
            auto qc = new qclass;
            int argcnt;
            if (pos+1 < q.size() && q[++pos] == SHUNTINGYARDVARIABLEARGUMENTKEY)
            {
                argcnt = stoi(q[++pos]);
                if (argcnt != 2)
                    std::cout << "Wrong number (" << argcnt << ") of arguments to a 'NAMING'\n";
            }
            int pos2 = pos+1;
            int namingcount = 0;
            std::string AStok = "AS";
            for (auto s : operatorsmap)
                if (s.second == formulaoperator::foas)
                {
                    AStok = s.first;
                    break;
                }
            while (pos2+1 <= q.size() && (q[pos2] != AStok || namingcount >= 1)) {
                namingcount += is_naming(q[pos2]) ? 1 : 0;
                namingcount -= q[pos2] == AStok ? 1 : 0;
                ++pos2;
            }
            if (pos2 >= q.size()) {
                std::cout << "'Naming' not containing an 'AS'\n";
                exit(1);
            }

            qc->superset = parseformulainternal(q, pos2, litnumps,littypes,litnames, ps, fnptrs);
            qc->name = q[++pos2];
            qc->criterion = nullptr;

            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs);
            fcright->boundvariable = qc;

            // if (argcnt == 3)
            // {
                // qc->criterion = parseformulainternal(q,pos, litnumps, littypes, litnames, ps,  fnptrs);
            // }

            formulaclass* fcleft = nullptr;
            formulaoperator o = lookupoperator(tok);
            formulavalue fv {};
            fv.qc = qc;
            auto fc = fccombine(fv,fcleft,fcright,o);

            pos = pos2;
            return fc;

        }

        if (is_quantifier(tok)) {

#ifdef QUANTIFIERMODE_PRECEDING
            auto qc = new qclass;
            qc->secondorder = false;
            // qc->eval( q, ++pos);
            int argcnt;
            if (pos+1 < q.size() && q[++pos] == SHUNTINGYARDVARIABLEARGUMENTKEY)
            {
                argcnt = stoi(q[++pos]);
                if (argcnt != 3 && argcnt != 2)
                    std::cout << "Wrong number (" << argcnt << ") of arguments to a quantifier\n";
            }
            std::string INtok = "IN";
            for (auto s : operatorsmap)
                if (s.second == formulaoperator::foin)
                {
                    INtok = s.first;
                    break;
                }
            int pos2 = pos+1;
            int quantcount = 0;
            while (pos2+1 <= q.size() && (q[pos2] != INtok || quantcount >= 1)) {
                quantcount += is_quantifier(q[pos2]) ? 1 : 0;
                quantcount -= q[pos2] == INtok ? 1 : 0;
                ++pos2;
            }
            if (pos2 >= q.size()) {
                std::cout << "Quantifier not containing an 'IN'\n";
                exit(1);
            }
            qc->superset = parseformulainternal(q, pos2, litnumps,littypes,litnames, ps, fnptrs);
            qc->name = q[++pos2];
            qc->criterion = nullptr;

            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs);
            fcright->boundvariable = qc;

            if (argcnt == 3)
            {
                qc->criterion = parseformulainternal(q,pos, litnumps, littypes, litnames, ps,  fnptrs);
            }

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
            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes, litnames, ps, fnptrs);
            formulaclass* fcleft = nullptr;
            if (o != formulaoperator::fonot)
            {
                fcleft = parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs);
            }
            if (fcright)
                return fccombine({},fcleft,fcright,o);
        }

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
            {
                if (pos+1 < q.size() && q[pos+1] == SHUNTINGYARDVARIABLEARGUMENTKEY)
                {
                    int argcnt = stoi(q[pos+2]);
                    pos += 2;
                    fv.subgraph = false;
                    if (argcnt == 1)
                    {
                        fv.subgraph = true;
                        formulaclass* subps = parseformulainternal( q, pos, litnumps, littypes, litnames, ps, fnptrs);
                        fv.lit.ps.push_back(subps);
                        return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
                    } else {
                        std::cout << "Literal expects " << litnumps[fv.lit.l] << " parameters, not " << argcnt << "parameters.\n";
                        return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
                    }
                }

                return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
            } else
            {
                if (pos+1 < q.size() && q[pos+1] == SHUNTINGYARDVARIABLEARGUMENTKEY)
                {
                    int argcnt = stoi(q[pos+2]);
                    pos += 2;

                    fv.subgraph = false;
                    if (argcnt != litnumps[fv.lit.l])
                    {
                        if (argcnt == litnumps[fv.lit.l]+1)
                        {
                            fv.subgraph = true;
                        } else {
                            std::cout << "Literal expects " << litnumps[fv.lit.l] << " parameters, not " << argcnt << "parameters.\n";
                            return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
                        }
                    };

                    std::vector<formulaclass*> psrev {};
                    for (int i = 0; i < argcnt; ++i) {
                        psrev.push_back(parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs));
                    }
                    for (int i = psrev.size()-1; i >= 0; --i)
                        fv.lit.ps.push_back(psrev[i]); // could add here support for named parameters

                    return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
                } else
                {
                    std::cout << "Error: parameterized literal has no parameters\n";
                    return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);

                }
            }

        }

        if (is_function(tok))
        {
            std::vector<formulaclass*> psoffunction {};
            // double (*f2)(std::vector<double>&);
            int argcnt = 0;
            bool found = false;
            for (auto f : *fnptrs)
            {
                found = f.first == tok;
                if (found)
                {
                    argcnt = f.second.second;
                    break;
                }
            }
            if (found) {
                std::vector<formulaclass*> psrev {};
                if (pos + 1 < q.size() && q[pos+1] == SHUNTINGYARDVARIABLEARGUMENTKEY)
                    if (stoi(q[pos+2]) == argcnt)
                    {
                        pos += 2;
                        for (int i = 0; i < argcnt; ++i) {
                            psrev.push_back(parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs));
                        }
                        for (int i = psrev.size()-1; i >= 0; --i)
                            psoffunction.push_back(psrev[i]);
                        formulavalue fv {};
                        if (auto search = fnptrs->find(tok); search != fnptrs->end())
                        {
                            fv.fns.fn = search->second.first;
                            fv.fns.ps = psoffunction;
                            fv.v.t = mtcontinuous;
                            return fccombine(fv,nullptr,nullptr,formulaoperator::fofunction);
                        } else
                        {
                            std::cout << "Unknown function " << tok << " in parseformula internal\n";
                        }
                    } else
                    {
                        std::cout << "Incorrect number of function arguments (" << stoi(q[pos+2]) << ") passed to function " << tok << " expecting " << argcnt << ".\n";
                        exit(1);
                    }
                else
                {
                    std::cout << "Error in Shuntingyard around function arguments\n";
                    exit(1);
                }
                formulavalue fv {};
                return fccombine(fv,nullptr,nullptr,formulaoperator::foliteral);
            }
        }

        if (is_number(tok)) { // integer
            formulavalue fv {};
            fv.v.v.iv = stoi(tok);
            fv.v.t = mtdiscrete;
            fv.ss.elts.clear();
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }
        if (is_real(tok))
        {
            formulavalue fv {};
            fv.v.v.dv = stof(tok);
            fv.v.t = mtcontinuous;
            fv.ss.elts.clear();
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }
        if (is_string(tok))
        {
            formulavalue fv {};
            std::string tok2 = tok.substr(1,tok.size()-2);
            fv.v.v.rv = new std::string(tok2);
            fv.v.t = mtstring;
            fv.ss.elts.clear();
            return fccombine(fv,nullptr,nullptr,formulaoperator::foconstant);
        }

        // conclude it must be a variable

        if (is_variable(tok))
        {
            formulavalue fv {};
            formulaclass* fc;
            if (pos+1 < q.size() && q[pos+1] == SHUNTINGYARDVARIABLEARGUMENTKEY)
            {
                pos += 2;
                if (q[pos] == "1")
                {
                    fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariable);
                    fc->v.vs.name = tok;
                    fc->v.vs.l = -1;
                    fc->v.vs.ps.clear();
                    fc->v.vs.ps.push_back(parseformulainternal(q,pos,litnumps,littypes, litnames, ps, fnptrs));
                } else
                {
                    if (q[pos] != "1")
                        std::cout << "Less or more than one parameter used to index into a set or tuple\n";
                }
            } else {
                fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariable);
                fc->v.vs.name = tok;
                fc->v.vs.l = -1;
                fc->v.vs.ps.clear();
            }
            return fc;
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
    namedparams& nps,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs )
{
    if (sentence != "") {
        // variables.clear();
        std::vector<std::string> c = parsecomponents(sentence);
        
        std::vector<std::string> components = Shuntingyardalg(c,litnumps);
        int pos = -1;
        return parseformulainternal( components,pos, litnumps, littypes, litnames, nps,fnptrs);
    } else {
        auto fc = new formulaclass({},nullptr,nullptr,formulaoperator::foconstant);
        return fc;
    }
}

inline bool searchfcforvariable( formulaclass* fc, std::vector<std::string> bound)
{
    if (!fc)
        return false;
    if (quantifierops(fc->fo))
    // if (quantifierops.find(fc->fo)->second)
    {
        bound.push_back(fc->v.qc->name);
        if (searchfcforvariable(fc->v.qc->superset,bound))
            return true;
        if (searchfcforvariable(fc->fcright, bound) || searchfcforvariable(fc->fcleft, bound))
            return true;
    }
    if (fc->fo == formulaoperator::fovariable)
    {
        for (auto s : bound)
            if (fc->v.vs.name == s)
                return false;
        return true;
    }
    if (fc->fcleft && searchfcforvariable(fc->fcleft,bound))
        return true;
    if (fc->fcright && searchfcforvariable(fc->fcright,bound))
        return true;
    if (fc->fo == formulaoperator::fofunction)
        for (auto p : fc->v.fns.ps)
            if (searchfcforvariable(p,bound))
                return true;
    if (fc->fo == formulaoperator::foconstant)
        for (auto p : fc->v.ss.elts)
            if (searchfcforvariable(p,bound))
                return true;
    if (fc->fo == formulaoperator::foliteral)
        for (auto p : fc->v.lit.ps)
            if (searchfcforvariable(p,bound))
                return true;
    return false;
}

