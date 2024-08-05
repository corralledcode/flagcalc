//
// Created by peterglenn on 7/15/24.
//

#ifndef MATH_H
#define MATH_H

#include <map>
#include <string>
#include "mathfn.h"
#include <cmath>
#include <functional>

#include "graphs.h"

class qclass;
class evalformula;

inline bool is_number(const std::string& s);


enum class logicalconnective {lcand, lcor};

struct logicalsentence {
    int item {};
    std::vector<logicalsentence> ls {};
    logicalconnective lc {logicalconnective::lcand};
    bool negated {false};
};

bool evalsentence( logicalsentence ls, std::vector<bool> literals );

logicalsentence lscombine( const logicalsentence ls1, const logicalsentence ls2, const logicalconnective lc );


enum class formulaoperator
{foliteral,fofunction, foconstant, foqforall, foqexists, foplus, fominus, fotimes, fodivide, foexponent,
folte, folt, foe,fone,fogte,fogt,
foand,foor,fonot,fotrue,fofalse,fovariable};

inline std::map<std::string,formulaoperator> operatorsmap
    {{"^",formulaoperator::foexponent},
        {"*",formulaoperator::fotimes},
        {"/",formulaoperator::fodivide},
        {"+",formulaoperator::foplus},
        {"-",formulaoperator::fominus},
        {"AND",formulaoperator::foand},
        {"OR",formulaoperator::foor},
        {"NOT",formulaoperator::fonot},
        {"&&",formulaoperator::foand},
        {"||",formulaoperator::foor},
        {"!",formulaoperator::fonot},
        {"==",formulaoperator::foe},
        {"<=",formulaoperator::folte},
        {"<",formulaoperator::folt},
        {">=",formulaoperator::fogte},
        {">",formulaoperator::fogt},
        {"!=",formulaoperator::fone},
        {"FORALL",formulaoperator::foqforall},
        {"EXISTS",formulaoperator::foqexists}};


std::vector<std::string> parsecomponents( std::string str);

logicalsentence parsesentenceinternal (std::vector<std::string> components, int nesting );

logicalsentence parsesentence( std::string sentence );

class formulaclass;

struct fnstruct {
    double (*fn)(std::vector<double>&);
    std::vector<formulaclass*> ps;
};

struct litstruct
{
    int l;
    std::vector<formulaclass*> ps;
};

union vals
{
    bool bv;
    double dv;
    int iv;
    bool* iset;
    // std::pair<int,int> pv {};
};

enum measuretype { mtbool, mtdiscrete, mtcontinuous, mtpair, mtset, mtpairset };


struct valms
{
    measuretype t;
    vals v;
    int setsize;
};

inline bool operator==(const valms& a1, const valms& a2)
{
    if (a1.t != a2.t)
        return false;

    switch (a1.t) {
    case measuretype::mtcontinuous:
        return a1.v.dv == a2.v.dv;
    case measuretype::mtdiscrete:
        return a1.v.iv == a2.v.iv;
    case measuretype::mtbool:
        return a1.v.bv == a2.v.bv;
    }
}




inline bool operator<(const valms& a1, const valms& a2)
{
    if (a1.t < a2.t)
        return true;
    if (a1.t == a2.t)
        switch (a1.t) {
        case measuretype::mtcontinuous:
            return a1.v.dv < a2.v.dv;
        case measuretype::mtdiscrete:
            return a1.v.iv < a2.v.iv;
        case measuretype::mtbool:
            return a1.v.bv < a2.v.bv;
        }
    return false;
}

//enum qtype { qtitem, qtpair, qtsubset, qtsubsetpair};


struct qstruct {
    std::string name;
    bool* subset;
    int n;
    int m; // for pairs
    measuretype t;
};


class qclass {
public:
    std::string name;
    valms qs;
    formulaclass* superset;
    void eval( const std::vector<std::string>& q, int& pos) {
        name = q[pos];
        ++pos;
        if (q[pos] == "SUBSETEQ") {
            qs.t = mtset;
        } else
            if (q[pos] == "SUBSETEQP") {
                qs.t = mtpairset;
            } else
                if (q[pos] == "IN") {
                    qs.t = mtdiscrete;
                } else
                    if (q[pos] == "INP") {
                        qs.t = mtpair;
                    } else
                        std::cout << "unknown quantifier variable";
                }
};

inline int lookup_variable( const std::string& tok, std::vector<qclass*>& variables) {
    bool found = false;
    int i = 0;
    while (!found && i < variables.size()) {
        found = variables[i]->name == tok;
        ++i;
    }
    if (!found)
        return -1;
    return i-1;
}


struct formulavalue {
    vals v;
    measuretype t;
    litstruct lit;
    fnstruct fns;
    qclass* qc;
};

class formulaclass {
public:
    formulavalue v;
    formulaclass* fcleft;
    formulaclass* fcright;
    formulaoperator fo;
    formulaclass(formulavalue vin, formulaclass* fcleftin, formulaclass* fcrightin, formulaoperator foin)
        : v{vin}, fcleft(fcleftin), fcright(fcrightin), fo(foin) {}
    ~formulaclass() {
        delete fcleft;
        delete fcright;
        if (fo == formulaoperator::fofunction) {
            for (auto fnf : v.fns.ps)
                delete fnf;
        }
        if (fo == formulaoperator::fovariable) {
            if (v.t == mtset || v.t == mtpairset)
                delete v.v.iset;
        }
    }

};



formulaclass* fccombine( const formulavalue& item, formulaclass* fc1, formulaclass* fc2, formulaoperator fo );


inline std::map<formulaoperator,int> precedencemap {
                            {formulaoperator::foqexists,0},
                            {formulaoperator::foqforall,0},
                            {formulaoperator::foexponent,1},
                            {formulaoperator::fotimes,2},
                            {formulaoperator::fodivide,2},
                            {formulaoperator::foplus,3},
                            {formulaoperator::fominus,3},
                            {formulaoperator::foe,4},
                            {formulaoperator::folte,4},
                            {formulaoperator::folt,4},
                            {formulaoperator::fogte,4},
                            {formulaoperator::fogt,4},
                            {formulaoperator::fone,4},
                            {formulaoperator::fonot,5},
                            {formulaoperator::foand,6},
                            {formulaoperator::foor,6}};





formulaclass* parseformula(
    const std::string& sentence,
    const std::vector<int>& litnumps,
    const std::vector<measuretype>& littypes,
    std::vector<qclass*>& variables,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs  );


class evalformula
{


public:
    graphtype* g {};
    std::vector<valms>* literals {};
    std::vector<measuretype>* littypes {};
    std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>*fnptrs = &global_fnptrs;
    //std::function<void()>* populatevariablesbound {};
    std::vector<qclass*> variables {};

    virtual valms evalpslit( const int idx, std::vector<valms>& psin );
    virtual valms evalvariable( std::string& vname );
    virtual valms eval( formulaclass& fc );
    evalformula();

    ~evalformula()
    {
        for (auto q : variables)
        {
            // if (!(q->name == "V" || q->name == "E" || q->name == "NE"))
                if (q->qs.t == mtset)
                    delete q->qs.v.iset;
            delete q;
        }
    }

};



#endif //MATH_H
