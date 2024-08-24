//
// Created by peterglenn on 7/15/24.
//

#ifndef MATH_H
#define MATH_H

#include <map>
#include <string>
#include "mathfn.h"
#include <cmath>
#include <cstring>
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
folte, folt, foe,fone,fogte,fogt,founion, fointersection, foelt,
foand,foor,foxor,fonot,foimplies,foiff,foif,fotrue,fofalse,fovariable};

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
        {"XOR",formulaoperator::foxor},
        {"!",formulaoperator::fonot},
        {"IMPLIES",formulaoperator::foimplies},
        {"IFF",formulaoperator::foiff},
        {"IF",formulaoperator::foif},
        {"==",formulaoperator::foe},
        {"<=",formulaoperator::folte},
        {"<",formulaoperator::folt},
        {">=",formulaoperator::fogte},
        {">",formulaoperator::fogt},
        {"!=",formulaoperator::fone},
        {"FORALL",formulaoperator::foqforall},
        {"EXISTS",formulaoperator::foqexists},
        {"CUP",formulaoperator::founion},
        {"CAP",formulaoperator::fointersection},
        {"ELT",formulaoperator::foelt}};


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

struct intpair
{
    int i;
    int j;
};

struct valspair;

union vals
{
    bool bv;
    double dv;
    int iv;
    // bool* iset;
    // intpair ip;
    valspair* p;
    // std::pair<int,int> pv {};
};

enum measuretype { mtbool, mtdiscrete, mtcontinuous, mtpair, mtset, mtpairset };

class setitr;
class itrpos;

struct valms
{
    measuretype t;
    vals v;
    int setsize;
    setitr* seti;
};

struct valspair
{
    valms i;
    valms j;
};

class setitr;

class itrpos;

class setitr
{
protected:
    std::vector<itrpos*> itrs {};
public:
    std::vector<valms> totality {};

    itrpos* getitrpos();

    int pos = 0;
    virtual int getsize() {return 0;}
    measuretype t = mtdiscrete;

    virtual void reset() {pos = 0;}
    virtual bool ended() {return true;}
    virtual valms getnext()
    {
        if (++pos >= totality.size())
            totality.push_back({});
        return totality[pos];
    }
    ~setitr()
    {
        for (itrpos* itr : itrs)
            delete itr;
        itrs.clear();
    }
};

class itrpos
{
public:
    setitr* parent;
    int pos = 0;
    bool ended()
    {
        return (pos >= parent->getsize());
    }
    void reset() {pos = 0;}
    int getsize() {return parent->getsize();}
    valms getnext()
    {
        valms res;
        while (pos >= parent->totality.size() && !parent->ended())
        {
            while (parent->pos < parent->totality.size())
            {
                parent->getnext();
                if (parent->ended())
                    break;
            }
            if (!parent->ended())
                parent->getnext();
        }
        if (pos < parent->totality.size())
            return parent->totality[pos++];
        ++pos;
        return {};
    }
    itrpos(setitr* parentin) : parent{parentin} {}
};

inline itrpos* setitr::getitrpos()
{
    itrs.push_back(new itrpos(this));
    return itrs.back();
}



template<typename T>
class setitrfactory
{
protected:
    std::vector<T*> setitrs {};
    public:
    T* getsetitr()
    {
        setitrs.push_back(new T);
        return setitrs.back();
    };
    ~setitrfactory()
    {
        for (auto s : setitrs)
            delete s;
    }
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
    case measuretype::mtpair:
        return a1.v.p->i == a2.v.p->i && a1.v.p->j == a2.v.p->j;
    case measuretype::mtset:
        {
            bool match = true;
            a1.seti->reset();
            a2.seti->reset();
            while (match && !a1.seti->ended())
            {
                valms v = a1.seti->getnext();
                match = match && !a2.seti->ended() && v == a2.seti->getnext();
            }
            return match;
        }
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
    case measuretype::mtset: {}
        // to do...
        }
    return false;
}

inline bool operator>(const valms& a1, const valms& a2)
{
    return a2 < a1;
}


inline bool operator<=(const valms& a1, const valms& a2)
{
    if (a1.t < a2.t)
        return true;
    if (a1 == a2)
        return true;
    if (a1.t == a2.t)
        switch (a1.t) {
    case measuretype::mtcontinuous:
        return a1.v.dv < a2.v.dv;
    case measuretype::mtdiscrete:
        return a1.v.iv < a2.v.iv;
    case measuretype::mtbool:
        return a1.v.bv < a2.v.bv;
    case measuretype::mtset: {}
            // to do...
        }
    return false;
}
inline bool operator>=(const valms& a1, const valms& a2)
{
    return a2 <= a1;
}




class setitrmodeone : public setitr
{
    public:

    bool computed = false;
    virtual void compute() {computed = true;};
    int getsize() override
    {
        if (!computed)
        {
            compute();
            computed = true;
        }
        return totality.size();
    }
    bool ended() override
    {
        return pos >= getsize();
    }
    valms getnext() override
    {
        if (!computed)
        {
            compute();
            computed = true;
            pos = 0;
        }
        if (!ended())
        {
            return totality[pos++];
        }
        return {};
    }
};

class setitrunion : public setitrmodeone
{
    public:
    setitr* setA;
    setitr* setB;
    void compute() override
    {
        auto Aitr = setA->getitrpos();
        auto Bitr = setB->getitrpos();
        pos = 0;
        totality.clear();
        while (!Aitr->ended())
        {
            totality.push_back(Aitr->getnext());
        }
        while (!Bitr->ended())
        {
            bool found = false;
            valms v = Bitr->getnext();
            for (int i = 0; !found && i < totality.size(); i++)
                found = found || v == totality[i];
            if (!found)
                totality.push_back(v);
        }
        computed = true;
        reset();
        // delete Aitr;
        // delete Bitr;
    }

    setitrunion(setitr* a, setitr* b) : setA{a}, setB{b} {};

};

class setitrintersection : public setitrmodeone
{
public:
    setitr* setA {};
    setitr* setB {};
    void compute() override
    {
        auto Aitr = setA->getitrpos();
        auto Bitr = setB->getitrpos();
        pos = 0;
        std::vector<valms> temp {};
        totality.clear();
        while (!Aitr->ended())
        {
            temp.push_back(Aitr->getnext());
        }
        while (!Bitr->ended())
        {
            bool found = false;
            valms v = Bitr->getnext();
            for (int i = 0; !found && i < temp.size(); i++)
                found = found || v == temp[i];
            if (found)
                totality.push_back(v);
        }
        computed = true;
        reset();
        // delete Aitr;
        // delete Bitr;
    }

    setitrintersection(setitr* a, setitr* b) : setA{a}, setB{b} {};

};

class setitrbool : public setitrmodeone
{
public:
    int maxint;
    bool* elts = nullptr;
    int relpos = 0;

    void compute() override
    {
        totality.clear();
        for (int i = 0; i < maxint; ++i)
        {
            if (elts[i])
            {
                valms v;
                v.t = measuretype::mtdiscrete;
                v.v.iv = i;
                totality.push_back(v);
            }
        }
        computed = true;
    }
    void setmaxint( const int maxintin )
    {
        maxint = maxintin;
        if (maxint > 0)
        {
            elts = (bool*)malloc(maxint*sizeof(bool));
            memset(elts, true, maxint*sizeof(bool));
        }
        else
            elts = nullptr;
        computed = false;
        reset();
    }

    setitrbool(const int maxintin)
    {
        setmaxint(maxintin);
        t = mtdiscrete;
        totality.clear();
    };
    setitrbool()
    {
        setmaxint(0);
        t = mtdiscrete;
    };

    ~setitrbool() {
        delete elts;
    }
};


class setitrsubset : public setitrbool {
public:
    itrpos* superset {};
    void setsuperset( itrpos* supersetposin )
    {
        superset = supersetposin;
        setmaxint(superset->getsize());
        reset();
    }
    valms getnext() override
    {
        valms r = setitrbool::getnext();
        valms res;
        if (r.v.iv >= 0)
            res = totality[r.v.iv];
        return res;
        // may add error catching code here
    }
    void reset() override
    {
        superset->reset();
        setitrbool::reset();
    }
    bool ended() override
    {
        return setitrbool::ended(); // || superset->ended();
    }
    setitrsubset(itrpos* supersetin) : superset{supersetin}, setitrbool(supersetin ? supersetin->getsize() : 0)
    {
        t = superset->parent->t;
    };
    setitrsubset() : superset{}, setitrbool(0)
    {
        t = mtdiscrete;
    };

};

class setitrint : public setitrmodeone
{
    public:
    int maxint = 0;
    bool* elts = nullptr;
    void compute() override
    {
        totality.clear();
        for (int i = 0; i < maxint; ++i)
        {
            if (elts[i])
            {
                valms v;
                v.t = measuretype::mtdiscrete;
                v.v.iv = i;
                totality.push_back(v);
            }
        }
        computed = true;
    }
    void setmaxint( const int maxintin )
    {
        delete elts;
        maxint = maxintin;
        if (maxint > 0)
        {
            elts = (bool*)malloc(maxint*sizeof(bool));
            memset(elts, true, maxint*sizeof(bool));
        } else
            elts = nullptr;
        computed = false;
        reset();
    }
    setitrint(const int maxintin)
    {
        setmaxint(maxintin);
        t = mtdiscrete;
    }
    setitrint()
    {
        setmaxint(0);
        t = mtdiscrete;
    }
    ~setitrint() {
        delete elts;
    }

};

class setitrset : public setitrmodeone
{
    public:
    setitrset( const std::vector<valms>& totalityin)
    {
        totality = totalityin;
        computed = true;
        pos = 0;
    }
    setitrset()
    {
        totality.clear();
        computed = true;
        pos = 0;
    }
};

class setitrinitialsegment : public setitr
{
public:

};

class qclass {
public:
    std::string name;
    valms qs;
    formulaclass* superset;
    bool secondorder = false;
    void eval( const std::vector<std::string>& q, int& pos)
    {
        name = q[pos];
        ++pos;
        if (q[pos] == "SUBSETEQ") {
            qs.t = mtset;
            secondorder = true;
        } else
            // if (q[pos] == "SUBSETEQP") {
                // qs.t = mtpairset;
                    // } else
                        if (q[pos] == "IN") {
                            // qs.t = mtdiscrete;
                            secondorder = false;
                        }
        // else
        // if (q[pos] == "INP") {
        // qs.t = mtpair;
        // }
                        else
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
    valms v;
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
            if (v.v.t == mtset || v.v.t == mtpairset)
                delete v.v.seti;
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
                            {formulaoperator::foelt,4},
                            {formulaoperator::fonot,5},
                            {formulaoperator::foand,6},
                            {formulaoperator::foor,6},
                            {formulaoperator::foxor,6},
                            {formulaoperator::foimplies,6},
                            {formulaoperator::foiff,6},
                            {formulaoperator::foif,6},
                            {formulaoperator::founion,7},
                            {formulaoperator::fointersection,7}};





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
                // if (q->qs.t == mtset)
                    // delete q->qs.seti;
            delete q;
        }
    }

};



#endif //MATH_H
