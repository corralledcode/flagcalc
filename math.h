//
// Created by peterglenn on 7/15/24.
//

#ifndef MATH_H
#define MATH_H

#include <map>
#include <string>
#include "mathfn.h"
#include <cmath>

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
{foliteral,fofunction, foconstant, foplus, fominus, fotimes, fodivide, foexponent,
folte, folt, foe,fone,fogte,fogt,
foand,foor,fonot,fotrue,fofalse};

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
        {"!=",formulaoperator::fone}};


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
};

enum measuretype { mtbool, mtdiscrete, mtcontinuous };


struct valms
{
    measuretype t;
    vals v;
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


struct formulavalue {
    vals v;
    measuretype t;
    litstruct lit;
    fnstruct fns;
};

class formulaclass {
public:
    const formulavalue v;
    const formulaclass* fcleft;
    const formulaclass* fcright;
    const formulaoperator fo;
    formulaclass(const formulavalue vin, const formulaclass* fcleftin, const formulaclass* fcrightin, const formulaoperator foin)
        : v{vin}, fcleft(fcleftin), fcright(fcrightin), fo(foin) {}
    ~formulaclass() {
        delete fcleft;
        delete fcright;
        if (fo == formulaoperator::fofunction) {
            for (auto fnf : v.fns.ps)
                delete fnf;
        }
    }

};



formulaclass* fccombine( const formulavalue& item, const formulaclass* fc1, const formulaclass* fc2, const formulaoperator fo );


inline std::map<formulaoperator,int> precedencemap {
                            {formulaoperator::foexponent,0},
                            {formulaoperator::fotimes,1},
                            {formulaoperator::fodivide,1},
                            {formulaoperator::foplus,2},
                            {formulaoperator::fominus,2},
                            {formulaoperator::foe,3},
                            {formulaoperator::folte,3},
                            {formulaoperator::folt,3},
                            {formulaoperator::fogte,3},
                            {formulaoperator::fogt,3},
                            {formulaoperator::fone,3},
                            {formulaoperator::fonot,4},
                            {formulaoperator::foand,5},
                            {formulaoperator::foor,5}};





formulaclass* parseformula(
    const std::string& sentence,
    const std::vector<int>& litnumps,
    const std::vector<measuretype>& littypes,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs  );


class evalformula
{


public:
    std::vector<valms>* literals {};
    std::vector<measuretype>* littypes {};
    std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>*fnptrs = &global_fnptrs;

    virtual valms evalpslit( const int idx, std::vector<valms>& psin );
    virtual valms eval( const formulaclass& fc);
    evalformula();

};



#endif //MATH_H
