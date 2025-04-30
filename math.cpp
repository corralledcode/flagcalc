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
#define SHUNTINGYARDVARIABLEARGUMENTENDKEY "_VARIABLEARGUMENTENDKEY7903822160"

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


inline std::map<formulaoperator,bool> booleanopslookup {
                            {formulaoperator::foqexists,false},
                            {formulaoperator::foqforall,false},
                            {formulaoperator::foqsum,false},
                            {formulaoperator::foqproduct,false},
                            {formulaoperator::foqmin,false},
                            {formulaoperator::foqmax,false},
                            {formulaoperator::foqrange,false},
                            {formulaoperator::foqaverage,false},
                            {formulaoperator::foqtally,false},
                            {formulaoperator::foqcount,false},
                            {formulaoperator::foqset,false},
                            {formulaoperator::foqdupeset,false},
                            {formulaoperator::foqtuple,false},
                            {formulaoperator::foqunion,false},
                            {formulaoperator::foqdupeunion,false},
                            {formulaoperator::foqintersection,false},
                            {formulaoperator::foqmedian,false},
                            {formulaoperator::foqmode,false},
                            {formulaoperator::fonaming,false},
                            {formulaoperator::foexponent,false},
                            {formulaoperator::fotimes,false},
                            {formulaoperator::fodivide,false},
                            {formulaoperator::fomodulus,false},
                            {formulaoperator::foplus,false},
                            {formulaoperator::fominus,false},
                            {formulaoperator::foe,false},
                            {formulaoperator::folte,false},
                            {formulaoperator::folt,false},
                            {formulaoperator::fogte,false},
                            {formulaoperator::fogt,false},
                            {formulaoperator::fone,false},
                            {formulaoperator::foelt,false},
                            {formulaoperator::fonot,false},
                            {formulaoperator::foand,true},
                            {formulaoperator::foor,true},
                            {formulaoperator::foxor,true},
                            {formulaoperator::foimplies,true},
                            {formulaoperator::foiff,true},
                            {formulaoperator::foif,true},
                            {formulaoperator::founion,false},
                            {formulaoperator::fodupeunion,false},
                            {formulaoperator::fointersection,false},
                            {formulaoperator::foswitch,false},
                            {formulaoperator::focases, false},
                            {formulaoperator::foin, false},
                            {formulaoperator::foas, false},
                            {formulaoperator::fosetminus, false},
                            {formulaoperator::fosetxor, false},
                            {formulaoperator::fomeet, false},
                            {formulaoperator::fodisjoint, false},
                            {formulaoperator::fothreaded, false},
                            {formulaoperator::forsort, false},
                            {formulaoperator::forpartition, false}
};


inline bool booleanops( const formulaoperator fo)
{
    // return booleanopslookup[fo];
    return (fo == formulaoperator::foand
            || fo == formulaoperator::foor
            || fo == formulaoperator::foimplies
            || fo == formulaoperator::foiff
            || fo == formulaoperator::foxor
            || fo == formulaoperator::foif);
}

inline std::map<formulaoperator,bool> equalityopslookup {
                            {formulaoperator::foqexists,false},
                            {formulaoperator::foqforall,false},
                            {formulaoperator::foqsum,false},
                            {formulaoperator::foqproduct,false},
                            {formulaoperator::foqmin,false},
                            {formulaoperator::foqmax,false},
                            {formulaoperator::foqrange,false},
                            {formulaoperator::foqaverage,false},
                            {formulaoperator::foqtally,false},
                            {formulaoperator::foqcount,false},
                            {formulaoperator::foqset,false},
                            {formulaoperator::foqdupeset,false},
                            {formulaoperator::foqtuple,false},
                            {formulaoperator::foqunion,false},
                            {formulaoperator::foqdupeunion,false},
                            {formulaoperator::foqintersection,false},
                            {formulaoperator::foqmedian,false},
                            {formulaoperator::foqmode,false},
                            {formulaoperator::fonaming,false},
                            {formulaoperator::foexponent,false},
                            {formulaoperator::fotimes,false},
                            {formulaoperator::fodivide,false},
                            {formulaoperator::fomodulus,false},
                            {formulaoperator::foplus,false},
                            {formulaoperator::fominus,false},
                            {formulaoperator::foe,true},
                            {formulaoperator::folte,true},
                            {formulaoperator::folt,true},
                            {formulaoperator::fogte,true},
                            {formulaoperator::fogt,true},
                            {formulaoperator::fone,true},
                            {formulaoperator::foelt,false},
                            {formulaoperator::fonot,false},
                            {formulaoperator::foand,false},
                            {formulaoperator::foor,false},
                            {formulaoperator::foxor,false},
                            {formulaoperator::foimplies,false},
                            {formulaoperator::foiff,false},
                            {formulaoperator::foif,false},
                            {formulaoperator::founion,false},
                            {formulaoperator::fodupeunion,false},
                            {formulaoperator::fointersection,false},
                            {formulaoperator::foswitch,false},
                            {formulaoperator::focases, false},
                            {formulaoperator::foin, false},
                            {formulaoperator::foas, false},
                            {formulaoperator::fosetminus, false},
                            {formulaoperator::fosetxor, false},
                            {formulaoperator::fomeet, true},
                            {formulaoperator::fodisjoint, true},
                            {formulaoperator::fothreaded, false},
                            {formulaoperator::forsort, false},
                            {formulaoperator::forpartition, false}
};


inline bool equalityops( const formulaoperator fo)
{
    // return equalityopslookup[fo];
    return (fo == formulaoperator::folte
            || fo == formulaoperator::folt
            || fo == formulaoperator::foe
            || fo == formulaoperator::fogt
            || fo == formulaoperator::fogte
            || fo == formulaoperator::fone
            || fo == formulaoperator::fomeet
            || fo == formulaoperator::fodisjoint);
}

inline std::map<formulaoperator,bool> quantifieropslookup {
                            {formulaoperator::foqexists,true},
                            {formulaoperator::foqforall,true},
                            {formulaoperator::foqsum,true},
                            {formulaoperator::foqproduct,true},
                            {formulaoperator::foqmin,true},
                            {formulaoperator::foqmax,true},
                            {formulaoperator::foqrange,true},
                            {formulaoperator::foqaverage,true},
                            {formulaoperator::foqtally,true},
                            {formulaoperator::foqcount,true},
                            {formulaoperator::foqset,true},
                            {formulaoperator::foqdupeset,true},
                            {formulaoperator::foqtuple,true},
                            {formulaoperator::foqunion,true},
                            {formulaoperator::foqdupeunion,true},
                            {formulaoperator::foqintersection,true},
                            {formulaoperator::foqmedian,true},
                            {formulaoperator::foqmode,true},
                            {formulaoperator::fonaming,false},
                            {formulaoperator::foexponent,false},
                            {formulaoperator::fotimes,false},
                            {formulaoperator::fodivide,false},
                            {formulaoperator::fomodulus,false},
                            {formulaoperator::foplus,false},
                            {formulaoperator::fominus,false},
                            {formulaoperator::foe,false},
                            {formulaoperator::folte,false},
                            {formulaoperator::folt,false},
                            {formulaoperator::fogte,false},
                            {formulaoperator::fogt,false},
                            {formulaoperator::fone,false},
                            {formulaoperator::foelt,false},
                            {formulaoperator::fonot,false},
                            {formulaoperator::foand,false},
                            {formulaoperator::foor,false},
                            {formulaoperator::foxor,false},
                            {formulaoperator::foimplies,false},
                            {formulaoperator::foiff,false},
                            {formulaoperator::foif,false},
                            {formulaoperator::founion,false},
                            {formulaoperator::fodupeunion,false},
                            {formulaoperator::fointersection,false},
                            {formulaoperator::foswitch,false},
                            {formulaoperator::focases, false},
                            {formulaoperator::foin, false},
                            {formulaoperator::foas, false},
                            {formulaoperator::fosetminus, false},
                            {formulaoperator::fosetxor, false},
                            {formulaoperator::fomeet, false},
                            {formulaoperator::fodisjoint, false},
                            {formulaoperator::fothreaded, false},
                            {formulaoperator::forsort, false},
                            {formulaoperator::forpartition, false}
};



inline bool quantifierops( const formulaoperator fo )
{
    // return quantifieropslookup[fo];
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
            || fo == formulaoperator::foqintersection
            || fo == formulaoperator::foqmedian
            || fo == formulaoperator::foqmode);
}

inline std::map<formulaoperator,bool> relationalopslookup {
                            {formulaoperator::foqexists,false},
                            {formulaoperator::foqforall,false},
                            {formulaoperator::foqsum,false},
                            {formulaoperator::foqproduct,false},
                            {formulaoperator::foqmin,false},
                            {formulaoperator::foqmax,false},
                            {formulaoperator::foqrange,false},
                            {formulaoperator::foqaverage,false},
                            {formulaoperator::foqtally,false},
                            {formulaoperator::foqcount,false},
                            {formulaoperator::foqset,false},
                            {formulaoperator::foqdupeset,false},
                            {formulaoperator::foqtuple,false},
                            {formulaoperator::foqunion,false},
                            {formulaoperator::foqdupeunion,false},
                            {formulaoperator::foqintersection,false},
                            {formulaoperator::foqmedian,false},
                            {formulaoperator::foqmode,false},
                            {formulaoperator::fonaming,false},
                            {formulaoperator::foexponent,false},
                            {formulaoperator::fotimes,false},
                            {formulaoperator::fodivide,false},
                            {formulaoperator::fomodulus,false},
                            {formulaoperator::foplus,false},
                            {formulaoperator::fominus,false},
                            {formulaoperator::foe,false},
                            {formulaoperator::folte,false},
                            {formulaoperator::folt,false},
                            {formulaoperator::fogte,false},
                            {formulaoperator::fogt,false},
                            {formulaoperator::fone,false},
                            {formulaoperator::foelt,false},
                            {formulaoperator::fonot,false},
                            {formulaoperator::foand,false},
                            {formulaoperator::foor,false},
                            {formulaoperator::foxor,false},
                            {formulaoperator::foimplies,false},
                            {formulaoperator::foiff,false},
                            {formulaoperator::foif,false},
                            {formulaoperator::founion,false},
                            {formulaoperator::fodupeunion,false},
                            {formulaoperator::fointersection,false},
                            {formulaoperator::foswitch,false},
                            {formulaoperator::focases, false},
                            {formulaoperator::foin, false},
                            {formulaoperator::foas, false},
                            {formulaoperator::fosetminus, false},
                            {formulaoperator::fosetxor, false},
                            {formulaoperator::fomeet, false},
                            {formulaoperator::fodisjoint, false},
                            {formulaoperator::fothreaded, false},
                            {formulaoperator::forsort, true},
                            {formulaoperator::forpartition, true}
};


inline bool relationalops( const formulaoperator fo )
{
    // return relationalopslookup[fo];
    return (fo == formulaoperator::forsort
            || fo == formulaoperator::forpartition);
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



valms evalformula::evalvariable( variablestruct& v, const namedparams& context, const std::vector<int>& vidxin ) {

    // int temp = lookup_variable(v.name,context);
    valms res;
    res = context[v.l].second;
    return res;
}

valms evalformula::evalvariablederef( variablestruct& v, const namedparams& context, const std::vector<int>& vidxin ) {

    valms res;
    int index = vidxin[0];
    auto pos = context[v.l].second.seti->getitrpos();
    pos->reset();
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

    return res;
}

void evalformula::preprocessbindvariablenames( formulaclass* fc, namedparams& context )
{
    if (fc)
    {
        if (fc->fo == formulaoperator::fovariable || fc->fo == formulaoperator::fovariablederef)
        {
            if (fc->v.vs.l < 0)
            {
                if (context.size()>0)
                    if (context[context.size()-1].first == fc->v.vs.name)
                        fc->v.vs.l = context.size()-1;
                    else
                    {
                        bool found = false;
                        int j;
                        for (j = context.size()-2; j >= 0 && !found; j--)
                            found = context[j].first == fc->v.vs.name;
                        if (!found)
                            std::cout << "Unknown variable name " << fc->v.vs.name << " (preprocessbindvariables)\n";
                        else
                            fc->v.vs.l = j+1;
                    }
                else
                {
                    std::cout << "Unknown variable name " << fc->v.vs.name << "\n";
                }
            }
        } else if (quantifierops(fc->fo) || fc->fo == formulaoperator::fonaming || relationalops(fc->fo))
        {
            valms v;
            // int startingsize = context.size();
            namedparams contexttemp = context;
            for (int i = 0; i < fc->fcright->boundvariables.size(); i++)
            {
                preprocessbindvariablenames(fc->fcright->boundvariables[i]->superset, context);
                preprocessbindvariablenames(fc->fcright->boundvariables[i]->alias, context);
                contexttemp.push_back({fc->fcright->boundvariables[i]->name,v });
            }
            preprocessbindvariablenames(fc->fcright->criterion, contexttemp);
            preprocessbindvariablenames(fc->fcright,contexttemp);
            // context.resize(startingsize);
        } else
        {
            preprocessbindvariablenames(fc->fcright,context);
            preprocessbindvariablenames(fc->fcleft,context);
            preprocessbindvariablenames(fc->criterion,context);
            preprocessbindvariablenames(fc->v.criterion,context);
            for (int i = 0; i < fc->boundvariables.size(); ++i)
            {
                preprocessbindvariablenames(fc->boundvariables[i]->alias,context);
                preprocessbindvariablenames(fc->boundvariables[i]->superset,context);
                preprocessbindvariablenames(fc->boundvariables[i]->value,context);
            }
            for (int i = 0; i < fc->v.fns.ps.size(); ++i)
                preprocessbindvariablenames(fc->v.fns.ps[i],context);
            for (int i = 0; i < fc->v.lit.ps.size(); ++i)
                preprocessbindvariablenames(fc->v.lit.ps[i],context);
        }
    }
}


int evalmformula::partitionforsort( std::vector<int> &arr, int start, int end, formulaclass* fc, namedparams& context, std::vector<valms>* v ) {
    int pivot = arr[start];
    int count = 0;

    for (int i = start+1;i <= end; i++) {
        context[context.size()-2].second = (*v)[arr[i]];
        context[context.size()-1].second = (*v)[pivot];
        if (!evalinternal(*fc,context).v.bv)
            count++;
    }

    int pivotIndex = start + count;
    std::swap(arr[pivotIndex],arr[start]);

    int i = start;
    int j = end;
    while (i < pivotIndex && j > pivotIndex) {
        context[context.size()-2].second = (*v)[arr[i]];
        context[context.size()-1].second = (*v)[pivot];
        while (!evalinternal(*fc,context).v.bv)
        {
            context[context.size()-2].second = (*v)[arr[i]];
            context[context.size()-1].second = (*v)[pivot];
            i++;
        }
        context[context.size()-2].second = (*v)[arr[j]];
        context[context.size()-1].second = (*v)[pivot];
        while (evalinternal(*fc,context).v.bv)
        {
            context[context.size()-2].second = (*v)[arr[j]];
            context[context.size()-1].second = (*v)[pivot];
            j--;
        }
        if (i < pivotIndex && j > pivotIndex) {
            std::swap(arr[i++],arr[j--]);
        }
    }
    return pivotIndex;
}

void evalmformula::quickSort( std::vector<int> &arr, int start, int end, formulaclass* fc, namedparams& context, std::vector<valms>* v ) {

    if (start >= end)
        return;

    int p = partitionforsort(arr,start,end,fc,context,v);

    quickSort(arr, start, p-1,fc,context,v);
    quickSort(arr, p+1, end, fc,context,v);
}

void evalmformula::threadevalcriterion(formulaclass* fc, formulaclass* criterion, namedparams* context, bool* c, valms* res)
{
    *c = evalinternal(*criterion, *context).v.bv;
    if (*c)
        *res = this->evalinternal(*fc, *context);
}

void evalmformula::threadeval(formulaclass* fc, namedparams* context, valms* res)
{
    *res = this->evalinternal(*fc, *context);
}

void evalmformula::partitionmerge( formulaclass* fc, namedparams* context, std::vector<std::vector<valms>>* v1, std::vector<std::vector<valms>>* v2, std::vector<std::pair<int,int>>* a )
{
    auto contextidxA = context->size()-1;
    auto contextidxB = context->size()-2;
    if (v2->size() == 0)
        return;
    while (true)
    {
        int i;
        bool found = false;
        (*context)[contextidxB].second = (*v2)[v2->size()-1][0];
        for (i = 0; i < v1->size() && !found; ++i)
        {
            (*context)[contextidxA].second = (*v1)[i][0];
            for (int l = 0; l < a->size(); ++l) {
                valms v = evalinternal(*fc->boundvariables[(*a)[l].second]->alias, *context);
                (*context)[(*a)[l].first].second = v;
            }
            found = evalinternal(*fc, *context).v.bv;
        }
        int k;
        if (found)
        {
            k = i-1;
            for (auto b : (*v2)[v2->size()-1])
                (*v1)[k].push_back(b);
        } else
            (*v1).push_back((*v2)[v2->size()-1]);
        v2->resize(v2->size()-1);
        if (v2->size() <= 0)
            break;
        for (int l = 0; l < a->size(); ++l) {
            valms v = evalinternal(*fc->boundvariables[(*a)[l].second]->alias, *context);
            (*context)[(*a)[l].first].second = v;
        }
    }
}


valms evalformula::eval( formulaclass& fc, namedparams& context) {}


valms evalmformula::evalinternal( formulaclass& fc, namedparams& context )
{
    valms res;
    if (fc.fo == formulaoperator::foliteral) {
        if (fc.v.lit.ps.empty())
        {
            // if (fc.v.lit.l >= 0 && fc.v.lit.l < literals.size()) {
                res = literals[fc.v.lit.l];
            // } else {
                // if (fc.v.lit.l < 0 && ((int)literals.size() + fc.v.lit.l >= 0))
                // {
                    // res = literals[literals.size() + fc.v.lit.l];
                // }
                // else {
                    // std::cout << "Error eval'ing formula\n";
                    // exit(1);
                    // return res;
                // }
            // }
        } else {
            std::vector<valms> ps {};
            int i = 0;
            for (auto f : fc.v.lit.ps) {
                ps.push_back(evalinternal(*f, context));
            }
            neighborstype* subgraph {};
            if (fc.v.subgraph)
            {
                subgraph = ps[0].v.nsv;
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
    case formulaoperator::fovariable:
        {
            std::vector<int> ps {};
            res = evalvariable(fc.v.vs, context, ps);
            return res;
        }

    case formulaoperator::fovariablederef:
        {
            std::vector<int> ps {};
            for (auto f : fc.v.vs.ps) {
                ps.push_back(evalinternal(*f, context).v.iv);
                // std::cout << "ps type " << f->v.v.t << " type " << ps.back().t << "seti type " << ps.back().seti->t << "\n";
            }
            res = evalvariable(fc.v.vs, context,ps);
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
                    std::cout << "Non-matching types in call to CUP, CAP, CUPD, CROSS, SETMINUS, or SETXOR\n";
                    res.seti = nullptr;
                    break;
                }
            else {
                std::cout << "Non-matching types in call to CUP, CAP, CUPD, CROSS, SETMINUS, or SETXOR\n";
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
        for (int i = 0; i < fc.fcright->boundvariables.size(); ++i) {
            valms v = evalinternal(*fc.fcright->boundvariables[i]->alias, context);
            context.push_back({fc.fcright->boundvariables[i]->name,v});
        }
        res = evalinternal(*fc.fcright, context);
        context.resize(context.size()-fc.fcright->boundvariables.size());
        return res;
    }

    // if (quantifierops.find(fc.fo)->second) {

    if (quantifierops(fc.fo))
    {
        std::vector<itrpos*> supersetpos {};
        std::vector<int> i {};
        std::vector<setitrint*> ss {};
        std::vector<valms> vv {};
        int originalcontextsize = context.size();
        auto criterion = fc.fcright->criterion;
        bool vacuouslytrue = false;
        std::vector<bool> needtodeletevseti {};
        needtodeletevseti.resize(fc.fcright->boundvariables.size());
        for (int j = 0; j < fc.fcright->boundvariables.size(); ++j) {
            valms v;
            if (fc.fcright->boundvariables[j]->superset) {
                v = evalinternal(*fc.fcright->boundvariables[j]->superset, context);
                if (v.t == mtdiscrete || v.t == mtcontinuous)
                {
                    if (v.t == mtcontinuous)
                        v.v.iv = (int)v.v.dv;
                    if (v.v.iv >= 0)
                        v.seti = new setitrint(v.v.iv-1);
                    else
                        v.seti = new setitrint(-1);
                    v.t = mtset;
                    needtodeletevseti[j] = true;
                } else {
                    if (v.t == mtbool)
                        std::cout << "Cannot use mtbool for quantifier superset\n";
                    needtodeletevseti[j] = false;
                }

                supersetpos.push_back(v.seti->getitrpos());
                ss.push_back(nullptr);
                context.push_back({fc.fcright->boundvariables[j]->name,fc.fcright->boundvariables[j]->qs});
                i.push_back( context.size()-1 );
                supersetpos[supersetpos.size()-1]->reset();
                if (supersetpos[supersetpos.size()-1]->ended()) {
                    vacuouslytrue = true;
                } else
                    context[i[i.size()-1]].second = supersetpos[supersetpos.size()-1]->getnext();
                vv.push_back(v);
            }
        }
        std::vector<std::pair<int,int>> a {};
        for (int j = 0; j < fc.fcright->boundvariables.size(); ++j)
        {
            if (fc.fcright->boundvariables[j]->alias) {
                valms v = evalinternal(*fc.fcright->boundvariables[j]->alias, context);
                context.push_back({fc.fcright->boundvariables[j]->name,v});
                a.push_back({context.size()-1,j});
            }
        }
        /* THE FOLLOWING CODE THOUGH ELEGANT SLOWS DOWN BY A THREE-FOLD FACTOR
                int k = 0;
                double min;
                double max;
                int count;
                std::vector<valms> tot;
                std::vector<setitr*> composite;
                switch (fc.fo)
                {
                case formulaoperator::foqexists:
                case formulaoperator::foqforall:
                    {
                        res.v.bv = true;
                        res.t = mtbool;
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
                        min = std::numeric_limits<double>::infinity();
                        max = -std::numeric_limits<double>::infinity();
                        count = 0; // this is not foqcount but rather to be used to find average
                        if (fc.fo == formulaoperator::foqsum)
                            res.v.dv = 0;
                        else if (fc.fo == formulaoperator::foqproduct)
                            res.v.dv = 1;
                        break;
                    }
                case formulaoperator::foqtally:
                case formulaoperator::foqcount:
                    {
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        break;
                    }
                case (formulaoperator::foqdupeset):
                case (formulaoperator::foqset):
                case (formulaoperator::foqtuple):
                    {
                        res.t = fc.fo == formulaoperator::foqtuple ? mttuple : mtset;
                        tot.clear();
                        break;
                    }
                case (formulaoperator::foqunion):
                case (formulaoperator::foqintersection):
                case (formulaoperator::foqdupeunion):
                    {
                        res.t = mtset;
                        res.seti = nullptr;
                        composite.clear();
                        break;
                    }
                }

                if (vacuouslytrue)
                    k = supersetpos.size();
                bool done = false;
                while (k < supersetpos.size() && !done) {
                    switch (fc.fo)
                    {
                        case formulaoperator::foqexists:
                        case formulaoperator::foqforall:
                            {
                                done = !res.v.bv;
                                break;
                            }
                        case formulaoperator::foqsum:
                        case formulaoperator::foqproduct:
                        case formulaoperator::foqmin:
                        case formulaoperator::foqmax:
                        case formulaoperator::foqrange:
                        case formulaoperator::foqaverage:
                            {
                                done = fc.fo == formulaoperator::foqproduct && res.v.dv == 0;
                                break;
                            }
                    }
                    valms c;
                    c.v.bv = true;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (c.v.bv)
                    {
                        auto ei = evalinternal(*fc.fcright, context);
                        switch (fc.fo)
                        {
                            case formulaoperator::foqexists:
                            case formulaoperator::foqforall:
                                {
                                    if (fc.fo == formulaoperator::foqexists)
                                        res.v.bv = res.v.bv && !ei.v.bv;
                                    else
                                        res.v.bv = res.v.bv && ei.v.bv;
                                    break;
                                }
                            case formulaoperator::foqsum:
                            case formulaoperator::foqproduct:
                            case formulaoperator::foqmin:
                            case formulaoperator::foqmax:
                            case formulaoperator::foqrange:
                            case formulaoperator::foqaverage:
                                {
                                    count++;  // this is not foqcount but rather to be used to find average
                                    auto v = ei;
                                    if (fc.fo != formulaoperator::foqproduct)
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
                                    }
                                    break;
                                }
                            case formulaoperator::foqtally:
                                {
                                    auto v = ei;
                                    valms tmp;
                                    mtconverttodiscrete(v, tmp.v.iv);
                                    res.v.iv += tmp.v.iv;
                                    break;
                                }
                            case formulaoperator::foqcount:
                                {
                                    auto v = ei;
                                    valms tmp;
                                    mtconverttobool(v,tmp.v.bv);
                                    if (tmp.v.bv)
                                        ++res.v.iv;
                                    break;
                                }
                            case formulaoperator::foqdupeset:
                            case formulaoperator::foqtuple:
                                {
                                    auto v = ei;
                                    tot.push_back(v);
                                    break;
                                }
                            case formulaoperator::foqset:
                                {
                                    auto v = ei;
                                    bool match = false;
                                    for (int i = 0; !match && i < tot.size(); i++)
                                    {
                                        match = match || mtareequal(tot[i], v);
                                    }
                                    if (!match)
                                        tot.push_back(v);
                                    break;
                                }
                            case formulaoperator::foqunion:
                            case formulaoperator::foqintersection:
                            case formulaoperator::foqdupeunion:
                                {
                                    valms tempv;
                                    valms outv;
                                    tempv = ei;
                                    mtconverttoset(tempv,outv.seti);
                                    composite.push_back(outv.seti);
                                    break;
                                }

                        }
                    }
                    quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                }
                switch (fc.fo)
                {
                case formulaoperator::foqexists:
                    {
                        res.v.bv = !res.v.bv;
                        break;
                    }
                case formulaoperator::foqrange:
                    {
                        res.v.dv = max - min;
                        break;
                    }
                case formulaoperator::foqaverage:
                    {
                        res.v.dv = count > 0 ? res.v.dv / count : 0;
                        break;
                    }
                case formulaoperator::foqmax:
                    {
                        res.v.dv = max;
                        break;
                    }
                case formulaoperator::foqmin:
                    {
                        res.v.dv = min;
                        break;
                    }
                case formulaoperator::foqdupeset:
                case formulaoperator::foqtuple:
                case formulaoperator::foqset:
                    {
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqunion:
                case formulaoperator::foqintersection:
                case formulaoperator::foqdupeunion:
                    {
                        auto abstractpluralsetops = getsetitrpluralops(composite);
                        res.seti = abstractpluralsetops->setops(fc.fo);
                        if (!res.seti)
                            res.seti = new setitrint(-1);
                        break;
                    }

                }
                for (int k = originalcontextsize; k < ss.size(); ++k) {
                    delete ss[k-originalcontextsize];
                    if (needtodeletevseti[k - originalcontextsize])
                        delete vv[k-originalcontextsize].seti;
                }
                context.resize(originalcontextsize);
                return res;
        */

        if (!fc.threaded)
        {
            if (criterion) // QUANTIFIER: case of not threaded, with criterion
            {
                switch (fc.fo)
                {
                case formulaoperator::foqexists:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && res.v.bv) {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                                res.v.bv = res.v.bv && !evalinternal(*fc.fcright, context).v.bv;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.v.bv = !res.v.bv;
                        break;
                    }

                case formulaoperator::foqforall:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && res.v.bv) {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                                res.v.bv = res.v.bv && evalinternal(*fc.fcright, context).v.bv;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        break;
                    }
                case formulaoperator::foqsum:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;

                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                if (fc.fo != formulaoperator::foqproduct)   // formulaoperator::foqsum || fc.fo == formulaoperator::foqrange
                                    // || fc.fo == formulaoperator::foqmin || fc.fo == formulaoperator::foqmax
                                        // || fc.fo == formulaoperator::foqaverage)
                                {
                                    switch (v.t) // headache maintaining code, but this slows down due to switch'ing on each iterant
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

                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                        }
                        break;
                    }
                case formulaoperator::foqproduct:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 1;;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(res.v.dv == 0))
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
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
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                        }
                        break;
                    }
                case formulaoperator::foqmin:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        double min = std::numeric_limits<double>::infinity();

                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                switch (v.t)
                                {
                                case mtcontinuous:
                                    res.v.dv = v.v.dv;
                                    break;
                                case mtdiscrete:
                                    res.v.dv = v.v.iv;
                                    break;
                                case mtbool:
                                    res.v.dv = v.v.bv ? 1 : 0;
                                    break;
                                case mtset:
                                    res.v.dv = v.seti->getsize();
                                    break;
                                }

                                if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                    min = res.v.dv;
                                else
                                    min = res.v.dv < min ? res.v.dv : min;
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                        }
                        res.v.dv = min;
                        break;
                    }
                case formulaoperator::foqmax:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        double max = -std::numeric_limits<double>::infinity();

                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                switch (v.t)
                                {
                                case mtcontinuous:
                                    res.v.dv = v.v.dv;
                                    break;
                                case mtdiscrete:
                                    res.v.dv = v.v.iv;
                                    break;
                                case mtbool:
                                    res.v.dv = v.v.bv ? 1 : 0;
                                    break;
                                case mtset:
                                    res.v.dv = v.seti->getsize();
                                    break;
                                }
                                if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                    max = res.v.dv;
                                else
                                    max = res.v.dv > max ? res.v.dv : max;
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                        }
                        res.v.dv = max;
                        break;
                    }
                case formulaoperator::foqrange:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        double min = std::numeric_limits<double>::infinity();
                        double max = -std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                switch (v.t)
                                {
                                case mtcontinuous:
                                    res.v.dv = v.v.dv;
                                    break;
                                case mtdiscrete:
                                    res.v.dv = v.v.iv;
                                    break;
                                case mtbool:
                                    res.v.dv = v.v.bv ? 1 : 0;
                                    break;
                                case mtset:
                                    res.v.dv = v.seti->getsize();
                                    break;
                                }

                                if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                    min = res.v.dv;
                                else
                                    min = res.v.dv < min ? res.v.dv : min;
                                if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                    max = res.v.dv;
                                else
                                    max = res.v.dv > max ? res.v.dv : max;
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.v.dv = max - min;
                        break;
                    }
                case formulaoperator::foqaverage:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0.0;
                        int count = 0; // this is not foqcount but rather to be used to find average
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                count++;  // this is not foqcount but rather to be used to find average
                                auto v = evalinternal(*fc.fcright, context);

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
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.v.dv = count > 0 ? res.v.dv / count : 0;
                        break;
                    }
                case formulaoperator::foqtally:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size()) {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                valms tmp;
                                mtconverttodiscrete(v, tmp.v.iv);
                                res.v.iv += tmp.v.iv;
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        break;
                    }
                case formulaoperator::foqcount:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size()) {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                valms tmp;
                                mtconverttobool(v,tmp.v.bv);
                                if (tmp.v.bv)
                                    ++res.v.iv;
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        break;
                    }
                case formulaoperator::foqdupeset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                tot.push_back(v);
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqtuple:
                    {
                        int k = 0;
                        res.t = mttuple;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                tot.push_back(v);
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                auto v = evalinternal(*fc.fcright, context);
                                bool match = false;
                                for (int i = 0; !match && i < tot.size(); i++)
                                {
                                    match = match || mtareequal(tot[i], v);
                                }
                                if (!match)
                                    tot.push_back(v);
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqunion:
                case formulaoperator::foqintersection:
                case formulaoperator::foqdupeunion:
                    {
                        int k = 0;
                        res.t = mtset;
                        res.seti = nullptr;
                        std::vector<setitr*> composite {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            valms c = evalinternal(*criterion, context);
                            if (c.v.bv)
                            {
                                valms tempv;
                                valms outv;
                                tempv = evalinternal(*fc.fcright, context);
                                mtconverttoset(tempv,outv.seti);
                                composite.push_back(outv.seti);
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }

                        auto abstractpluralsetops = getsetitrpluralops(composite);

                        res.seti = abstractpluralsetops->setops(fc.fo);

                        if (!res.seti)
                            res.seti = new setitrint(-1);
                        break;
                    }
                case formulaoperator::foqmedian:
                    {
                        std::cout << "No support yet for MEDIAN\n";
                        exit(1);
                        break;
                    }
                case formulaoperator::foqmode:
                    {
                        std::cout << "No support yet for MODE\n";
                        exit(1);
                        break;
                    }
                }

            } else // QUANTIFIER: case of not threaded, no criterion
            {
                switch (fc.fo)
                {
                case formulaoperator::foqexists:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && res.v.bv) {
                            res.v.bv = res.v.bv && !evalinternal(*fc.fcright, context).v.bv;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.v.bv = !res.v.bv;
                        break;
                    }

                case formulaoperator::foqforall:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && res.v.bv) {
                            res.v.bv = res.v.bv && evalinternal(*fc.fcright, context).v.bv;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        break;
                    }
                case formulaoperator::foqsum:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;

                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            if (fc.fo != formulaoperator::foqproduct)   // formulaoperator::foqsum || fc.fo == formulaoperator::foqrange
                                // || fc.fo == formulaoperator::foqmin || fc.fo == formulaoperator::foqmax
                                    // || fc.fo == formulaoperator::foqaverage)
                            {
                                double out;
                                mtconverttocontinuous(v,out);
                                res.v.dv += out;
                            }
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                        }
                        break;
                    }
                case formulaoperator::foqproduct:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 1;;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(res.v.dv == 0))
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            double out;
                            mtconverttocontinuous(v,out);
                            res.v.dv *= out;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        break;
                    }
                case formulaoperator::foqmin:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        double min = std::numeric_limits<double>::infinity();

                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            mtconverttocontinuous(v,res.v.dv);
                            if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                min = res.v.dv;
                            else
                                min = res.v.dv < min ? res.v.dv : min;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                        }
                        res.v.dv = min;
                        break;
                    }
                case formulaoperator::foqmax:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        double max = -std::numeric_limits<double>::infinity();

                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            mtconverttocontinuous(v,res.v.dv);
                            if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                max = res.v.dv;
                            else
                                max = res.v.dv > max ? res.v.dv : max;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                        }
                        res.v.dv = max;
                        break;
                    }
                case formulaoperator::foqrange:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        double min = std::numeric_limits<double>::infinity();
                        double max = -std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            mtconverttocontinuous(v,res.v.dv);
                            if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                min = res.v.dv;
                            else
                                min = res.v.dv < min ? res.v.dv : min;
                            if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                max = res.v.dv;
                            else
                                max = res.v.dv > max ? res.v.dv : max;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.v.dv = max - min;
                        break;
                    }
                case formulaoperator::foqaverage:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0.0;
                        int count = 0; // this is not foqcount but rather to be used to find average
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            count++;  // this is not foqcount but rather to be used to find average
                            auto v = evalinternal(*fc.fcright, context);
                            double out;
                            mtconverttocontinuous(v,out);
                            res.v.dv += out;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.v.dv = count > 0 ? res.v.dv / count : 0;
                        break;
                    }
                case formulaoperator::foqtally:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size()) {
                            auto v = evalinternal(*fc.fcright, context);
                            valms tmp;
                            mtconverttodiscrete(v, tmp.v.iv);
                            res.v.iv += tmp.v.iv;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        break;
                    }
                case formulaoperator::foqcount:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size()) {
                            auto v = evalinternal(*fc.fcright, context);
                            valms tmp;
                            mtconverttobool(v,tmp.v.bv);
                            if (tmp.v.bv)
                                ++res.v.iv;
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        break;
                    }
                case formulaoperator::foqdupeset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            tot.push_back(v);
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqtuple:
                    {
                        int k = 0;
                        res.t = mttuple;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            tot.push_back(v);
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            auto v = evalinternal(*fc.fcright, context);
                            bool match = false;
                            for (int i = 0; !match && i < tot.size(); i++)
                            {
                                match = match || mtareequal(tot[i], v);
                            }
                            if (!match)
                                tot.push_back(v);
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqunion:
                case formulaoperator::foqintersection:
                case formulaoperator::foqdupeunion:
                    {
                        int k = 0;
                        res.t = mtset;
                        res.seti = nullptr;
                        std::vector<setitr*> composite {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        while (k < supersetpos.size())
                        {
                            valms tempv;
                            valms outv;
                            tempv = evalinternal(*fc.fcright, context);
                            mtconverttoset(tempv,outv.seti);
                            composite.push_back(outv.seti);
                            quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                        }

                        auto abstractpluralsetops = getsetitrpluralops(composite);

                        res.seti = abstractpluralsetops->setops(fc.fo);

                        if (!res.seti)
                            res.seti = new setitrint(-1);
                        break;
                    }
                case formulaoperator::foqmedian:
                    {
                        std::cout << "No support yet for MEDIAN\n";
                        exit(1);
                        break;
                    }
                case formulaoperator::foqmode:
                    {
                        std::cout << "No support yet for MODE\n";
                        exit(1);
                        break;
                    }
                }
            }


        } else // case of threaded quantifier
        {
            if (criterion) // QUANTIFIER: case of threaded and with criterion
            {
                switch (fc.fo)
                {
                case formulaoperator::foqexists:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size() && res.v.bv)
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                // contexts[pos].clear();
                                // for (auto c : context)
                                // {
                                    // contexts[pos].push_back(c);
                                // }
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m) {
                                ress[m].v.bv = true;
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            }
                            for (int m = 0; m < pos ; ++m)
                            {
                                t[m].get();
                                res.v.bv = (!c[m] || !ress[m].v.bv) && res.v.bv;
                            }
                        }
                        res.v.bv = !res.v.bv;
                        break;
                    }

                case formulaoperator::foqforall:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size() && res.v.bv)
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m) {
                                ress[m].v.bv = true;
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            }
                            for (int m = 0; m < pos ; ++m)
                            {
                                t[m].get();
                                res.v.bv = (!c[m] || ress[m].v.bv) && res.v.bv;
                            }
                        }
                        break;
                    }
                case formulaoperator::foqsum:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    double out;
                                    mtconverttocontinuous(ress[m],out);
                                    res.v.dv += out;
                                }
                        }
                        break;
                    }
                case formulaoperator::foqproduct:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 1;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size() && res.v.dv != 0)
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    double out;
                                    mtconverttocontinuous(ress[m],out);
                                    res.v.dv *= out;
                                }
                        }
                        break;
                    }
                case formulaoperator::foqmin:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        double min = std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    mtconverttocontinuous(ress[m],res.v.dv);
                                    if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                        min = res.v.dv;
                                    else
                                        min = res.v.dv < min ? res.v.dv : min;
                                }
                        }
                        res.v.dv = min;
                        break;
                    }
                case formulaoperator::foqmax:
                    {
                       int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        double max = -std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    mtconverttocontinuous(ress[m],res.v.dv);
                                    if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                        max = res.v.dv;
                                    else
                                        max = res.v.dv > max ? res.v.dv : max;
                                }
                        }
                        res.v.dv = max;
                        break;
                    }
                case formulaoperator::foqrange:
                    {
                       int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        double min = std::numeric_limits<double>::infinity();
                        double max = -std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    mtconverttocontinuous(ress[m],res.v.dv);
                                    if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                        min = res.v.dv;
                                    else
                                        min = res.v.dv < min ? res.v.dv : min;
                                    if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                        max = res.v.dv;
                                    else
                                        max = res.v.dv > max ? res.v.dv : max;
                                }
                        }
                        res.v.dv = max - min;
                        break;
                    }
                case formulaoperator::foqaverage:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        int count = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    ++count;
                                    double out;
                                    mtconverttocontinuous(ress[m],out);
                                    res.v.dv += out;
                                }
                        }
                        res.v.dv = count > 0 ? res.v.dv/count : 0;
                        break;
                    }
                case formulaoperator::foqtally:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    int tmp;
                                    mtconverttodiscrete(ress[m],tmp);
                                    res.v.iv += tmp;
                                }
                        }
                        break;
                    }
                case formulaoperator::foqcount:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    bool tmp;
                                    mtconverttobool(ress[m],tmp);
                                    if (tmp)
                                        ++res.v.iv;
                                }
                        }
                        break;
                    }
                case formulaoperator::foqdupeset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                    tot.push_back(ress[m]);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqtuple:
                    {
                        int k = 0;
                        res.t = mttuple;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }
                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                    tot.push_back(ress[m]);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }
                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m]) {
                                    bool match = false;
                                    for (int i = 0; !match && i < tot.size(); i++)
                                        match = match || mtareequal(tot[i], ress[m]);
                                    if (!match)
                                        tot.push_back(ress[m]);
                                }
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqunion:
                case formulaoperator::foqintersection:
                case formulaoperator::foqdupeunion:
                    {
                        int k = 0;
                        res.t = mtset;
                        res.seti = nullptr;
                        std::vector<setitr*> composite {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            bool c[pos];
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadevalcriterion,this,fc.fcright,criterion,&contexts[m],&c[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                if (c[m])
                                {
                                    setitr* out;
                                    mtconverttoset(ress[m],out);
                                    composite.push_back(out);
                                }
                        }
                        auto abstractpluralsetops = getsetitrpluralops(composite);
                        res.seti = abstractpluralsetops->setops(fc.fo);

                        if (!res.seti)
                            res.seti = new setitrint(-1);

                        break;
                    }
                case formulaoperator::foqmedian:
                    {
                        std::cout << "No support yet for MEDIAN\n";
                        exit(1);
                        break;
                    }
                case formulaoperator::foqmode:
                    {
                        std::cout << "No support yet for MODE\n";
                        exit(1);
                        break;
                    }
                }

            } else { // QUANTIFIER: case of threaded and no criterion
                switch (fc.fo)
                {
                case formulaoperator::foqexists:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size() && res.v.bv)
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m) {
                                ress[m].v.bv = true;
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            }
                            for (int m = 0; m < pos ; ++m)
                            {
                                t[m].get();
                                res.v.bv = !ress[m].v.bv && res.v.bv;
                            }
                        }
                        res.v.bv = !res.v.bv;
                        break;
                    }

                case formulaoperator::foqforall:
                    {
                        int k = 0;
                        res.v.bv = true;
                        res.t = mtbool;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size() && res.v.bv)
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m) {
                                ress[m].v.bv = true;
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            }
                            for (int m = 0; m < pos ; ++m)
                            {
                                t[m].get();
                                res.v.bv = ress[m].v.bv && res.v.bv;
                            }
                        }
                        break;
                    }
                case formulaoperator::foqsum:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos; ++m)
                                t[m].get();

                            for (int m = 0; m < pos ; ++m)
                            {
                                double out;
                                mtconverttocontinuous(ress[m],out);
                                res.v.dv += out;
                            }
                        }
                        break;
                    }
                case formulaoperator::foqproduct:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 1;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size() && res.v.dv != 0)
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                double out;
                                mtconverttocontinuous(ress[m],out);
                                res.v.dv *= out;
                            }
                        }
                        break;
                    }
                case formulaoperator::foqmin:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        double min = std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                mtconverttocontinuous(ress[m],res.v.dv);
                                if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                    min = res.v.dv;
                                else
                                    min = res.v.dv < min ? res.v.dv : min;
                            }
                        }
                        res.v.dv = min;
                        break;
                    }
                case formulaoperator::foqmax:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        double max = -std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                mtconverttocontinuous(ress[m],res.v.dv);
                                if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                    max = res.v.dv;
                                else
                                    max = res.v.dv > max ? res.v.dv : max;
                            }
                        }
                        res.v.dv = max;
                        break;
                    }
                case formulaoperator::foqrange:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        double min = std::numeric_limits<double>::infinity();
                        double max = -std::numeric_limits<double>::infinity();
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                mtconverttocontinuous(ress[m],res.v.dv);
                                if (min == std::numeric_limits<double>::infinity() || min == -std::numeric_limits<double>::infinity())
                                    min = res.v.dv;
                                else
                                    min = res.v.dv < min ? res.v.dv : min;
                                if (max == -std::numeric_limits<double>::infinity() || max == std::numeric_limits<double>::infinity())
                                    max = res.v.dv;
                                else
                                    max = res.v.dv > max ? res.v.dv : max;
                            }
                        }
                        res.v.dv = max - min;
                        break;
                    }
                case formulaoperator::foqaverage:
                    {
                        int k = 0;
                        res.t = mtcontinuous;
                        res.v.dv = 0;
                        int count = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                ++count;
                                double out;
                                mtconverttocontinuous(ress[m],out);
                                res.v.dv += out;
                            }
                        }
                        res.v.dv = count > 0 ? res.v.dv/count : 0;
                        break;
                    }
                case formulaoperator::foqtally:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                int out;
                                mtconverttodiscrete(ress[m],out);
                                res.v.iv += out;
                            }
                        }
                        break;
                    }
                case formulaoperator::foqcount:
                    {
                        int k = 0;
                        res.t = mtdiscrete;
                        res.v.iv = 0;
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                bool out;
                                mtconverttobool(ress[m],out);
                                if (out)
                                    ++res.v.iv;
                            }
                        }
                        break;
                    }
                case formulaoperator::foqdupeset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                tot.push_back(ress[m]);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqtuple:
                    {
                        int k = 0;
                        res.t = mttuple;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                                tot.push_back(ress[m]);
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqset:
                    {
                        int k = 0;
                        res.t = mtset;
                        std::vector<valms> tot {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m) {
                                bool match = false;
                                for (int i = 0; !match && i < tot.size(); i++)
                                {
                                    match = match || mtareequal(tot[i], ress[m]);
                                }
                                if (!match)
                                    tot.push_back(ress[m]);
                            }
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                case formulaoperator::foqunion:
                case formulaoperator::foqintersection:
                case formulaoperator::foqdupeunion:
                    {
                        int k = 0;
                        res.t = mtset;
                        res.seti = nullptr;
                        std::vector<setitr*> composite {};
                        if (vacuouslytrue)
                            k = supersetpos.size();
                        std::vector<namedparams> contexts {};
                        contexts.resize(thread_count);
                        while (k < supersetpos.size())
                        {
                            int pos = 0;
                            while (pos < thread_count && k < supersetpos.size())
                            {
                                contexts[pos] = context;
                                quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                                pos++;
                            }

                            std::vector<std::future<void>> t;
                            t.resize(pos);
                            std::vector<valms> ress;
                            ress.resize(pos);
                            for (int m = 0; m < pos; ++m)
                                t[m] = std::async(&evalmformula::threadeval,this,fc.fcright,&contexts[m],&ress[m]);
                            for (int m = 0; m < pos ; ++m)
                                t[m].get();
                            for (int m = 0; m < pos ; ++m)
                            {
                                valms outv;
                                mtconverttoset(ress[m],outv.seti);
                                composite.push_back(outv.seti);
                            }
                        }
                        auto abstractpluralsetops = getsetitrpluralops(composite);
                        res.seti = abstractpluralsetops->setops(fc.fo);

                        if (!res.seti)
                            res.seti = new setitrint(-1);

                        break;
                    }
                case formulaoperator::foqmedian:
                    {
                        std::cout << "No support yet for MEDIAN\n";
                        exit(1);
                        break;
                    }
                case formulaoperator::foqmode:
                    {
                        std::cout << "No support yet for MODE\n";
                        exit(1);
                        break;
                    }
                }
            }
        }

        /* REPLACED WITH MORE EFFICIENT (ALBEIT MORE LONG) CODE

        switch (fc.fo) {
            case (formulaoperator::foqexists):
            case (formulaoperator::foqforall):
            {
                int k = 0;
                res.v.bv = true;
                res.t = mtbool;
                if (vacuouslytrue)
                    k = supersetpos.size();
                while (k < supersetpos.size() && res.v.bv) {

                    valms c;
                    c.v.bv = true;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (c.v.bv)
                        if (fc.fo == formulaoperator::foqexists)
                            res.v.bv = res.v.bv && !evalinternal(*fc.fcright, context).v.bv;
                        else
                            res.v.bv = res.v.bv && evalinternal(*fc.fcright, context).v.bv;
                    // std::cout << fc.v.qc->qs.v.iv << " iv \n";

                    quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

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
            case (formulaoperator::foqaverage): {
                int k = 0;
                res.t = mtcontinuous;
                double min = std::numeric_limits<double>::infinity();
                double max = -std::numeric_limits<double>::infinity();
                int count = 0; // this is not foqcount but rather to be used to find average
                if (fc.fo == formulaoperator::foqsum)
                    res.v.dv = 0;
                else if (fc.fo == formulaoperator::foqproduct)
                    res.v.dv = 1;;

                if (vacuouslytrue)
                    k = supersetpos.size();
                while (k < supersetpos.size() && !(fc.fo == formulaoperator::foqproduct && res.v.dv == 0))
                {
                    valms c;
                    c.t = mtbool;
                    c.v.bv = true;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (c.v.bv)
                    {
                        count++;  // this is not foqcount but rather to be used to find average
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
                    quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

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
                int k = 0;
                res.t = mtdiscrete;
                res.v.iv = 0;

                if (vacuouslytrue)
                    k = supersetpos.size();
                while (k < supersetpos.size()) {
                    valms c;
                    c.v.bv = true;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (c.v.bv)
                    {
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
                    quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                }
                break;
            }

            case (formulaoperator::foqdupeset):
            case (formulaoperator::foqtuple):
            {
                int k = 0;
                res.t = fc.fo == formulaoperator::foqdupeset ? mtset : mttuple;
                std::vector<valms> tot {};
                if (vacuouslytrue)
                    k = supersetpos.size();
                while (k < supersetpos.size())
                {
                    valms c;
                    c.v.bv = true;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (c.v.bv)
                    {
                        auto v = evalinternal(*fc.fcright, context);
                        tot.push_back(v);
                    }
                    quantifiermultipleadvance(fc,supersetpos,k,context,i,a);

                }
                res.seti = new setitrmodeone(tot);
                break;
            }
            case (formulaoperator::foqset):
            {
                int k = 0;
                res.t = mtset;
                std::vector<valms> tot {};
                if (vacuouslytrue)
                    k = supersetpos.size();
                while (k < supersetpos.size())
                {
                    valms c;
                    c.v.bv = true;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (c.v.bv)
                    {
                        auto v = evalinternal(*fc.fcright, context);
                        bool match = false;
                        for (int i = 0; !match && i < tot.size(); i++)
                        {
                            match = match || mtareequal(tot[i], v);
                        }
                        if (!match)
                            tot.push_back(v);
                    }
                    quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                }
                res.seti = new setitrmodeone(tot);
                break;
            }


            case (formulaoperator::foqunion):
            case (formulaoperator::foqintersection):
            case (formulaoperator::foqdupeunion): {
                int k = 0;
                res.t = mtset;
                res.seti = nullptr;
                std::vector<setitr*> composite {};
                if (vacuouslytrue)
                    k = supersetpos.size();
                while (k < supersetpos.size())
                {
                    valms c;
                    c.v.bv = true;
                    if (criterion) {
                        c = evalinternal(*criterion, context);
                        if (c.t != mtbool) {
                            std::cout << "Quantifier criterion requires boolean value\n";
                            exit(1);
                        }
                    }
                    if (c.v.bv)
                    {
                        valms tempv;
                        valms outv;
                        tempv = evalinternal(*fc.fcright, context);
                        mtconverttoset(tempv,outv.seti);
                        composite.push_back(outv.seti);
                    }
                    quantifiermultipleadvance(fc,supersetpos,k,context,i,a);
                }

                auto abstractpluralsetops = getsetitrpluralops(composite);

                res.seti = abstractpluralsetops->setops(fc.fo);

                if (!res.seti)
                    res.seti = new setitrint(-1);
                break;
            }
        } */
        for (int k = originalcontextsize; k < ss.size(); ++k) {
            delete ss[k-originalcontextsize];
            if (needtodeletevseti[k - originalcontextsize])
                delete vv[k-originalcontextsize].seti;
        }
        context.resize(originalcontextsize);

        return res;
    }



    if (relationalops(fc.fo))
    {
        std::vector<itrpos*> supersetpos {};
        std::vector<int> i {};
        std::vector<setitrint*> ss {};
        std::vector<valms> vv {};
        int originalcontextsize = context.size();
        auto criterion = fc.fcright->criterion;
        bool vacuouslytrue = false;
        std::vector<bool> needtodeletevseti {};
        needtodeletevseti.resize(fc.fcright->boundvariables.size());
        for (int j = 0; j < fc.fcright->boundvariables.size(); ++j) {
            valms v;
            if (fc.fcright->boundvariables[j]->superset) {
                v = evalinternal(*fc.fcright->boundvariables[j]->superset, context);
                if (v.t == mtdiscrete || v.t == mtcontinuous)
                {
                    if (v.t == mtcontinuous)
                        v.v.iv = (int)v.v.dv;
                    if (v.v.iv >= 0)
                        v.seti = new setitrint(v.v.iv-1);
                    else
                        v.seti = new setitrint(-1);
                    v.t = mtset;
                    needtodeletevseti[j] = true;
                } else {
                    if (v.t == mtbool)
                        std::cout << "Cannot use mtbool for quantifier superset\n";
                    needtodeletevseti[j] = false;
                }

                supersetpos.push_back(v.seti->getitrpos());
                ss.push_back(nullptr);
                context.push_back({fc.fcright->boundvariables[j]->name,fc.fcright->boundvariables[j]->qs});
                i.push_back( context.size()-1 );
                supersetpos[supersetpos.size()-1]->reset();
                if (supersetpos[supersetpos.size()-1]->ended()) {
                    vacuouslytrue = true;
                } else
                    context[i[i.size()-1]].second = supersetpos[supersetpos.size()-1]->getnext();
                vv.push_back(v);
            }
        }
        std::vector<std::pair<int,int>> a {};
        for (int j = 0; j < fc.fcright->boundvariables.size(); ++j)
        {
            if (fc.fcright->boundvariables[j]->alias) {
                valms v = evalinternal(*fc.fcright->boundvariables[j]->alias, context);
                context.push_back({fc.fcright->boundvariables[j]->name,v});
                a.push_back({context.size()-1,j});
            }
        }


        if (!fc.threaded)
            if (!criterion) // RELATIONAL: case of not threaded, no criterion
            {
                switch (fc.fo)
                {
                case formulaoperator::forpartition:
                    {
                        std::vector<std::vector<valms>> tot {};
                        res.t = mtset;

                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "PARTITION relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        if (!vacuouslytrue)
                            while (true)
                            {
                                bool found = false;
                                int i;
                                for (i = 0; i < tot.size() && !found; i++)
                                {
                                    context[contextidxA].second = tot[i][0];
                                    found = evalinternal(*fc.fcright, context).v.bv;
                                }
                                if (found)
                                    tot[i-1].push_back(context[contextidxB].second);
                                else
                                {
                                    tot.resize(tot.size()+1);
                                    tot[tot.size()-1].clear();
                                    tot[tot.size()-1].push_back(context[contextidxB].second);
                                }
                                if (supersetpos[supersetpos.size()-2]->ended())
                                    break;
                                context[contextidxB].second = supersetpos[supersetpos.size()-2]->getnext();
                                for (int j = 0; j < a.size(); ++j) {
                                    valms v = evalinternal(*fc.fcright->boundvariables[a[j].second]->alias, context);
                                    context[a[j].first].second = v;
                                }
                            }
                        std::vector<valms> c {};
                        for (auto a : tot)
                        {
                            auto b = new setitrmodeone(a );
                            valms r;
                            r.t = mtset;
                            r.seti = b;
                            c.push_back(r);
                        }
                        res.seti = new setitrmodeone(c);
                        break;
                    }
                case formulaoperator::forsort:
                    {
                        res.t = mttuple;
                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "SORT relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        std::vector<valms> v {};
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        std::vector<int> arr;
                        if (!vacuouslytrue)
                            while (true)
                            {
                                v.push_back(context[contextidxA].second);
                                if (supersetpos[supersetpos.size()-1]->ended())
                                    break;
                                context[contextidxA].second = supersetpos[supersetpos.size()-1]->getnext();
                                for (int j = 0; j < a.size(); ++j) {
                                    valms v = evalinternal(*fc.fcright->boundvariables[a[j].second]->alias, context);
                                    context[a[j].first].second = v;
                                }
                            }
                        arr.resize(v.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            arr[i] = i;
                        }
                        quickSort(arr,0,arr.size()-1,fc.fcright,context,&v);

                        std::vector<valms> tot;
                        tot.resize(arr.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            tot[i] = v[arr[i]];
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                }
            } else // RELATIONAL: case of criterion, not threaded
            {
                switch (fc.fo)
                {
                case formulaoperator::forpartition:
                    {
                        std::vector<std::vector<valms>> tot {};
                        res.t = mtset;

                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "PARTITION relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        if (!vacuouslytrue)
                            while (true)
                            {
                                bool found = false;
                                int i;
                                auto c = evalinternal(*criterion, context);
                                if (c.v.bv)
                                {
                                    for (i = 0; i < tot.size() && !found; i++)
                                    {
                                        context[contextidxA].second = tot[i][0];
                                        found = evalinternal(*fc.fcright, context).v.bv;
                                    }
                                    if (found)
                                        tot[i-1].push_back(context[contextidxB].second);
                                    else
                                    {
                                        tot.resize(tot.size()+1);
                                        tot[tot.size()-1].clear();
                                        tot[tot.size()-1].push_back(context[contextidxB].second);
                                    }
                                }
                                if (supersetpos[supersetpos.size()-2]->ended())
                                    break;
                                context[contextidxB].second = supersetpos[supersetpos.size()-2]->getnext();
                                for (int j = 0; j < a.size(); ++j) {
                                    valms v = evalinternal(*fc.fcright->boundvariables[a[j].second]->alias, context);
                                    context[a[j].first].second = v;
                                }
                            }
                        std::vector<valms> c {};
                        for (auto a : tot)
                        {
                            auto b = new setitrmodeone(a );
                            valms r;
                            r.t = mtset;
                            r.seti = b;
                            c.push_back(r);
                        }
                        res.seti = new setitrmodeone(c);
                        break;
                    }
                case formulaoperator::forsort:
                    {
                        res.t = mttuple;
                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "SORT relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        std::vector<valms> v {};
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        std::vector<int> arr;
                        if (!vacuouslytrue)
                            while (true)
                            {
                                auto c = evalinternal(*criterion, context);
                                if (c.v.bv)
                                {
                                    v.push_back(context[contextidxA].second);
                                }
                                if (supersetpos[supersetpos.size()-1]->ended())
                                    break;
                                context[contextidxA].second = supersetpos[supersetpos.size()-1]->getnext();
                                for (int j = 0; j < a.size(); ++j) {
                                    valms v = evalinternal(*fc.fcright->boundvariables[a[j].second]->alias, context);
                                    context[a[j].first].second = v;
                                }

                            }
                        arr.resize(v.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            arr[i] = i;
                        }
                        quickSort(arr,0,arr.size()-1,fc.fcright,context,&v);

                        std::vector<valms> tot;
                        tot.resize(arr.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            tot[i] = v[arr[i]];
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                }
            }
        else {
            if (!criterion) // RELATIONAL: case of threaded, no criterion
            {
                switch (fc.fo)
                {
                case formulaoperator::forpartition:
                    {
                        std::vector<std::vector<valms>> tot {};
                        res.t = mtset;

                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "PARTITION relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        std::vector<std::vector<std::vector<valms>>> v {};
                        if (!vacuouslytrue)
                        {
                            std::vector<namedparams> contexts;
                            contexts.resize(thread_count);
                            for (int m = 0; m < thread_count; ++m)
                            {
                                // contexts[m].clear();
                                contexts[m] = context;
                            }
                            // for (auto c : context)
                                // for (int m = 0; m < thread_count; ++m)
                                    // contexts[m].push_back(c);

                            v.push_back({{context[context.size()-1].second}});
                            const int j = supersetpos.size()-1;
                            while (!supersetpos[supersetpos.size()-1]->ended())
                            {
                                v.push_back({{supersetpos[j]->getnext()}});
                                // v.resize(v.size()+1);
                                // int i = v.size()-1;
                                // v[i].clear();
                                // v[i].push_back({supersetpos[supersetpos.size()-1]->getnext()});
                            }
                            while (v.size() > 1)
                            {
                                const int pos = thread_count <= ceil((v.size()-1)/2.0) ? thread_count-1 : ceil((v.size()-1)/2.0);
                                std::vector<std::future<void>> t;
                                t.resize(pos+1);
                                for (int m = 0; m < pos; ++m) {
                                    const int i = m;
                                    const int j = pos + m;
                                    t[m] = std::async(&evalmformula::partitionmerge,this,fc.fcright,&contexts[m],&v[i],&v[j],&a);
                                }
                                if (v.size() % 2 != 0)
                                    t[pos] = std::async(&evalmformula::partitionmerge,this,fc.fcright,&contexts[pos],&v[0],&v[v.size()-1],&a);
                                if (v.size() % 2 != 0)
                                    for (int m = 0; m < pos+1 ; ++m)
                                        t[m].get();
                                else
                                    for (int m = 0; m < pos; ++m)
                                        t[m].get();
                                v.resize(pos);
                                // for (int m = pos-1; m >= 0; --m)
                                    // v.erase(v.begin()+2*m+1);
                            }
                        } else
                            v.push_back({});
                        std::vector<valms> c {};
                        for (auto a : v[0])
                        {
                            auto b = new setitrmodeone(a );
                            valms r;
                            r.t = mtset;
                            r.seti = b;
                            c.push_back(r);
                        }
                        res.seti = new setitrmodeone(c);
                        break;
                    }
                case formulaoperator::forsort:
                    {
                        res.t = mttuple;
                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "SORT relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        std::vector<valms> v {};
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        std::vector<int> arr;
                        if (!vacuouslytrue)
                            while (true)
                            {
                                v.push_back(context[contextidxA].second);
                                if (supersetpos[supersetpos.size()-1]->ended())
                                    break;
                                context[contextidxA].second = supersetpos[supersetpos.size()-1]->getnext();
                                for (int j = 0; j < a.size(); ++j) {
                                    valms v = evalinternal(*fc.fcright->boundvariables[a[j].second]->alias, context);
                                    context[a[j].first].second = v;
                                }
                            }
                        arr.resize(v.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            arr[i] = i;
                        }
                        quickSort(arr,0,arr.size()-1,fc.fcright,context,&v);

                        std::vector<valms> tot;
                        tot.resize(arr.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            tot[i] = v[arr[i]];
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                }
            } else // RELATIONAL: case of threaded, criterion
            {
                switch (fc.fo)
                {
                case formulaoperator::forpartition:
                    {
                        std::vector<std::vector<valms>> tot {};
                        res.t = mtset;

                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "PARTITION relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        std::vector<std::vector<std::vector<valms>>> v {};
                        if (!vacuouslytrue)
                        {
                            std::vector<namedparams> contexts;
                            contexts.resize(thread_count);
                            for (int m = 0; m < thread_count; ++m)
                                contexts[m].clear();
                            for (auto c : context)
                                for (int m = 0; m < thread_count; ++m)
                                    contexts[m].push_back(c);

                            v.resize(1);
                            v[0].push_back({context[context.size()-1].second});
                            while (!supersetpos[supersetpos.size()-1]->ended())
                            {
                                v.resize(v.size()+1);
                                int i = v.size()-1;
                                v[i].clear();
                                v[i].push_back({supersetpos[supersetpos.size()-1]->getnext()});
                                context[context.size()-2] = {fc.fcright->boundvariables[0]->name,v[i][v[i].size()-1][0]};
                                context[context.size()-1] = {fc.fcright->boundvariables[1]->name,v[i][v[i].size()-1][0]};
                                if (!evalinternal(*criterion,context).v.bv)
                                    v[i].resize(v[i].size()-1);
                            }
                            while (v.size() >= 2)
                            {
                                int pos = thread_count <= ceil((v.size()-1)/2.0) ? thread_count : ceil((v.size()-1)/2.0);
                                std::vector<std::future<void>> t;
                                t.resize(pos);
                                for (int m = 0; m < pos; ++m) {
                                    int i = 2*m;
                                    int j = 2*m+1;
                                    t[m] = std::async(&evalmformula::partitionmerge,this,fc.fcright,&contexts[m],&v[i],&v[j],&a);
                                }
                                for (int m = 0; m < pos ; ++m)
                                    t[m].get();
                                for (int m = pos-1; m >= 0; --m)
                                    v.erase(v.begin()+2*m+1);
                            }
                        } else
                            v.push_back({});
                        std::vector<valms> c {};
                        for (auto a : v[0])
                        {
                            auto b = new setitrmodeone(a );
                            valms r;
                            r.t = mtset;
                            r.seti = b;
                            c.push_back(r);
                        }
                        res.seti = new setitrmodeone(c);
                        break;
                    }
                case formulaoperator::forsort:
                    {
                        res.t = mttuple;
                        if (fc.fcright->boundvariables.size() != 2)
                        {
                            std::cout << "SORT relational quantifier requires exactly two variables\n";
                            exit(1);
                        }
                        std::vector<valms> v {};
                        int contextidxB = context.size()-2;
                        int contextidxA = context.size()-1;
                        std::vector<int> arr;
                        if (!vacuouslytrue)
                            while (true)
                            {
                                auto c = evalinternal(*criterion, context);
                                if (c.v.bv)
                                {
                                    v.push_back(context[contextidxA].second);
                                }
                                if (supersetpos[supersetpos.size()-1]->ended())
                                    break;
                                context[contextidxA].second = supersetpos[supersetpos.size()-1]->getnext();
                                for (int j = 0; j < a.size(); ++j) {
                                    valms v = evalinternal(*fc.fcright->boundvariables[a[j].second]->alias, context);
                                    context[a[j].first].second = v;
                                }

                            }
                        arr.resize(v.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            arr[i] = i;
                        }
                        quickSort(arr,0,arr.size()-1,fc.fcright,context,&v);

                        std::vector<valms> tot;
                        tot.resize(arr.size());
                        for (int i = 0; i < arr.size(); i++)
                        {
                            tot[i] = v[arr[i]];
                        }
                        res.seti = new setitrmodeone(tot);
                        break;
                    }
                }
            }


        }

        for (int k = originalcontextsize; k < ss.size(); ++k) {
            delete ss[k-originalcontextsize];
            if (needtodeletevseti[k - originalcontextsize])
                delete vv[k-originalcontextsize].seti;
        }
        context.resize(originalcontextsize);
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

void evalmformula::quantifiermultipleadvance( formulaclass& fc, std::vector<itrpos*> &supersetpos, int &k, std::vector<std::pair<std::string,valms>> &context, std::vector<int> &i, std::vector<std::pair<int,int>> &a )
{
    while (k < supersetpos.size() && supersetpos[k]->ended())
        ++k;
    if (k < supersetpos.size())
    {
        context[i[k]].second = supersetpos[k]->getnext();
        for (int l = 0; l < k; ++l) {
            supersetpos[l]->reset();
            context[i[l]].second = supersetpos[l]->getnext();
        }
        for (int j = 0; j < a.size(); ++j)
        {
            valms v = evalinternal(*fc.fcright->boundvariables[a[j].second]->alias, context);
            context[a[j].first].second = v;
        }
        k = 0;
    }
}


inline valms evalmformula::eval( formulaclass& fc, namedparams& context )
{
    if (idx >= 0)
    {
        literals.resize(rec->literals.size());
        for (int i = 0; i < rec->literals.size(); ++i)
            literals[i] = rec->literals[i][idx];
    } else
    {
        literals.clear();
    }

    // namedparams contextlocal {};
    // for (int i = 0; i < context.size(); ++i)
    // {
        // contextlocal.push_back(context[i]);
        // bindvariablenames(&fc,contextlocal);
    // }

    // auto contextlocal = context;
    preprocessbindvariablenames(&fc,context);

    return evalinternal( fc, context );
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
    // if ((tok.size() > 2 && tok[0] == '[' && tok[tok.size()-1] == ']'))
        // return true;
    // else
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
        // if (tok.size() > 2 && tok[0] == '[' && tok[tok.size()-1] == ']')
            // return tok.substr(1,tok.size()-2);
        // else
            return tok;
    return "";
}

inline bool is_quantifier( std::string tok ) {
    for (auto q : operatorsmap)
        if (tok == q.first)
            return quantifierops(q.second);
    return false;
}

inline bool is_relational( std::string tok ) {
    for (auto q : operatorsmap)
        if (tok == q.first)
            return relationalops(q.second);
    return false;
}


inline bool is_naming(std::string tok)
{
    for (auto q : operatorsmap)
        if (tok == q.first)
            return q.second == formulaoperator::fonaming;
    return false;
}

inline bool is_threaded(std::string tok)
{
    for (auto q : operatorsmap)
        if (tok == q.first)
            return q.second == formulaoperator::fothreaded;
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
        if (is_operator(tok) && !is_quantifier(tok) && !is_naming(tok) && !is_relational(tok)) {
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
        if (is_literal(tok) || is_quantifier(tok) || is_function(tok) || is_variable(tok) || is_naming(tok)
            || is_threaded(tok) || is_relational(tok))
        {
            if (n < components.size())
            {
                if (components[n] == "(" || is_threaded(components[n]))
                {
                    if (is_quantifier(tok) || is_naming(tok) || is_relational(tok))
                        output.insert(output.begin(),SHUNTINGYARDVARIABLEARGUMENTENDKEY);
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
            } else
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
                while (ostok != "(") {
                    if (operatorstack.empty()) {
                        std::cout << "Error mismatched parentheses (loc 1)\n";
                        return output;
                    }
                    output.insert(output.begin(),ostok);
                    operatorstack.resize(operatorstack.size()-1);
                    if (!operatorstack.empty())
                        ostok = operatorstack[operatorstack.size()-1];
                    else
                    {
                        std::cout << "Error mismatched parentheses (loc 2)\n";
                        for (auto o : output)
                            std::cout << o << ", ";
                        std::cout << "\n";
                        return output;
                    }
                }
            }
            if (operatorstack.empty() || operatorstack[operatorstack.size()-1] != "(") {
                std::cout << "Error mismatched parentheses (loc 3)\n";
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
                if (is_operator(ostok) && !is_quantifier(ostok) && !is_naming(ostok) && !is_relational(ostok)) {
                    // continue;
                } else
                    if (is_function(ostok) || is_quantifier(ostok) || is_literal(ostok) || is_variable(ostok)
                        || is_naming(ostok) || is_relational(ostok) || ostok == SHUNTINGYARDDEREFKEY)
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
            std::vector<qclass*> qcs {};
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
                namingcount += (is_quantifier(q[pos2]) || is_naming(q[pos2]) || is_relational(q[pos2])) ? 1 : 0;
                namingcount -= q[pos2] == SHUNTINGYARDVARIABLEARGUMENTENDKEY ? 1 : 0;
                ++pos2;
            }
            if (pos2 >= q.size() && namingcount == 0) {
                std::cout << "'Naming' not containing an 'AS'\n";
                exit(1);
            }

            qc->superset = nullptr;
            qc->alias = parseformulainternal(q, pos2, litnumps,littypes,litnames, ps, fnptrs);
            while (q[pos2+1] == SHUNTINGYARDVARIABLEARGUMENTENDKEY)
                ++pos2;
            qc->name = q[++pos2];
            qcs.push_back(qc);

            formulaclass* fcright;
            fcright = parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs);
            fcright->boundvariables = qcs;

            formulaclass* fcleft = nullptr;
            formulaoperator o = lookupoperator(tok);
            formulavalue fv {};
            fv.qcs = qcs;
            auto fc = fccombine(fv,fcleft,fcright,o);

            pos = pos2;
            return fc;

        }

        if (is_threaded(tok))
        {
            formulavalue fv {};
            auto fc = parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs);
            if (!quantifierops(fc->fo) && !relationalops(fc->fo))
            {
                std::cout << "THREADED keyword must be followed by a quantifier\n";
                fc = nullptr;
            }
            fc->threaded = true;
            return fc;
        }
        if (is_quantifier(tok) || is_relational(tok)) {

            std::vector<qclass*> qcs {};
            int argcnt;
            if (pos+1 < q.size() && q[++pos] == SHUNTINGYARDVARIABLEARGUMENTKEY)
            {
                argcnt = stoi(q[++pos]);
                if (argcnt < 2)
                    std::cout << "Wrong number (" << argcnt << ") of arguments to a quantifier or relational quantifier\n";
            }
            std::string INtok = "IN";
            for (auto s : operatorsmap)
                if (s.second == formulaoperator::foin)
                {
                    INtok = s.first;
                    break;
                }
            std::string AStok = "AS";
            for (auto s : operatorsmap)
                if (s.second == formulaoperator::foas)
                {
                    AStok = s.first;
                    break;
                }
            int pos2 = pos+1;
            int lastpos2 = pos2;
            int quantcount = 0;
            int INcount = 0;
            while (pos2 < q.size() && q[pos2] != SHUNTINGYARDVARIABLEARGUMENTENDKEY && quantcount >= 0) {
                while (pos2+1 <= q.size() && (q[pos2] != INtok || quantcount >= 1) && quantcount >= 0) {
                    quantcount += (is_naming(q[pos2]) || is_quantifier(q[pos2]) || is_relational(q[pos2])) ? 1 : 0;
                    quantcount -= q[pos2] == SHUNTINGYARDVARIABLEARGUMENTENDKEY ? 1 : 0;
                    ++pos2;
                }

                if (pos2 < q.size() && q[pos2] == INtok && quantcount == 0) {
                    INcount++;
                    auto qc = new qclass;
                    qc->superset = parseformulainternal(q, pos2, litnumps,littypes,litnames, ps, fnptrs);
                    qc->alias = nullptr;
                    while (q[pos2+1] == SHUNTINGYARDVARIABLEARGUMENTENDKEY)
                        ++pos2;
                    qc->name = q[++pos2];
                    qcs.push_back(qc);
                    if (is_relational(tok))
                    {
                        INcount++;
                        auto qc2 = new qclass;
                        qc2->superset = qc->superset;
                        qc2->alias = nullptr;
                        qc2->name = q[++pos2];
                        qcs.push_back(qc2);
                    }
                    lastpos2 = pos2;
                    pos2++;
                }
            }

            if (pos2 >= q.size() && qcs.size() == 0) {
                std::cout << "Quantifier not containing an 'IN'\n";
                exit(1);
            }

            int pos3 = pos+1;
            int lastpos3 = pos3;
            int namingcount = 0;
            int AScount = 0;

            while (pos3 < q.size() && q[pos3] != SHUNTINGYARDVARIABLEARGUMENTENDKEY && namingcount >= 0) {
                while (pos3+1 <= q.size() && (q[pos3] != AStok || namingcount >= 1) && namingcount >= 0) {
                    namingcount += (is_naming(q[pos3]) || is_quantifier(q[pos3]) || is_relational(q[pos3])) ? 1 : 0;
                    namingcount -= q[pos3] == SHUNTINGYARDVARIABLEARGUMENTENDKEY ? 1 : 0;
                    ++pos3;
                }

                if (pos3 < q.size() && q[pos3] == AStok && namingcount == 0) {
                    AScount++;
                    auto qc = new qclass;
                    qc->superset = nullptr;
                    qc->alias = parseformulainternal(q, pos3, litnumps,littypes,litnames, ps, fnptrs);
                    while (q[pos3+1] == SHUNTINGYARDVARIABLEARGUMENTENDKEY)
                        ++pos3;
                    qc->name = q[++pos3];
                    qcs.push_back(qc);
                    lastpos3 = pos3;
                    pos3++;
                }
            }

            formulaclass* fcright = parseformulainternal(q,pos,litnumps,littypes,litnames, ps, fnptrs);
            fcright->criterion = nullptr;
            fcright->boundvariables = qcs;

            if (argcnt > (INcount + AScount + 1))
            {
                fcright->criterion = parseformulainternal(q,pos, litnumps, littypes, litnames, ps,  fnptrs);
            }

            formulaclass* fcleft = nullptr;
            formulaoperator o = lookupoperator(tok);
            formulavalue fv {};
            fv.qcs = qcs;

            auto fc = fccombine(fv,fcleft,fcright,o);
            fc->boundvariables = qcs;

            pos = pos > (lastpos2 > lastpos3 ? lastpos2 : lastpos3) ? pos : (lastpos2 > lastpos3 ? lastpos2 : lastpos3);
            return fc;
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
                    fc = fccombine(fv,nullptr,nullptr,formulaoperator::fovariablederef);
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

    std::cout << "Error in parsing formula\n";
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
    {
        for (int i = 0; i < fc->v.qcs.size(); ++i) {
            bound.push_back(fc->v.qcs[i]->name);
            if (searchfcforvariable(fc->v.qcs[i]->superset,bound))
                return true;
            if (searchfcforvariable(fc->fcright, bound) || searchfcforvariable(fc->fcleft, bound))
                return true;
        }
    }
    if (fc->fo == formulaoperator::fovariable || fc->fo == formulaoperator::fovariablederef)
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

