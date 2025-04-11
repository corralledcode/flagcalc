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
#include <regex>

#include "graphio.h"

#include "graphs.h"

#define ABSCUTOFF 0.000001

class qclass;
class evalformula;

bool is_number(const std::string& s);
bool is_real(const std::string& s);


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
{foliteral,fofunction, foconstant, foderef,
    foqforall, foqexists,
    foplus, fominus, fotimes, fodivide, foexponent, fomodulus,
    folte, folt, foe, fone, fogte, fogt, founion, fodupeunion, fointersection, foelt,
    foand,foor,foxor,fonot,foimplies,foiff,foif,fotrue,fofalse,fovariable,
    foqsum, foqproduct, foqmin, foqmax, foqaverage, foqrange,
    foqtally, foqcount, foqset, foqdupeset, foqunion, foqdupeunion, foqintersection,
    foswitch, focases, foin, fonaming, foas,
    fosetminus, fosetxor, fomeet, fodisjoint};

inline const std::map<std::string,formulaoperator> operatorsmap
    {{"^",formulaoperator::foexponent},
        {"*",formulaoperator::fotimes},
        {"/",formulaoperator::fodivide},
        {"+",formulaoperator::foplus},
        {"-",formulaoperator::fominus},
        {"%",formulaoperator::fomodulus},
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
        {"SET",formulaoperator::foqset},
        {"SETD",formulaoperator::foqdupeset},
        {"FORALL",formulaoperator::foqforall},
        {"EXISTS",formulaoperator::foqexists},
        {"CUP",formulaoperator::founion},
        {"CUPD",formulaoperator::fodupeunion},
        {"CAP",formulaoperator::fointersection},
        {"ELT",formulaoperator::foelt},
        {"SUM",formulaoperator::foqsum},
        {"MIN",formulaoperator::foqmin},
        {"MAX",formulaoperator::foqmax},
        {"PRODUCT",formulaoperator::foqproduct},
        {"AVERAGE",formulaoperator::foqaverage},
        {"RANGE",formulaoperator::foqrange},
        {"TALLY",formulaoperator::foqtally},
        {"COUNT",formulaoperator::foqcount},
        {"BIGCUP",formulaoperator::foqunion},
        {"BIGCUPD",formulaoperator::foqdupeunion},
        {"BIGCAP", formulaoperator::foqintersection},
        {"?", formulaoperator::foswitch},
        {":", formulaoperator::focases},
        {"IN", formulaoperator::foin},
        {"NAMING", formulaoperator::fonaming},
        {"AS", formulaoperator::foas},
        {"SETMINUS", formulaoperator::fosetminus},
        {"SETXOR", formulaoperator::fosetxor},
        {"MEET", formulaoperator::fomeet},
        {"DISJOINT", formulaoperator::fodisjoint}};


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
    std::string lname;
    std::vector<formulaclass*> ps;
};

struct variablestruct
{
    int l;
    std::string name;
    std::vector<formulaclass*> ps;
};

struct setstruct
{
    std::vector<formulaclass*> elts;
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
    neighborstype* nsv; // which has g built in
    std::string* rv;
};

enum measuretype { mtbool, mtdiscrete, mtcontinuous, mtset, mttuple, mtstring, mtgraph };

inline std::map<measuretype,std::string> measuretypenames {
    {mtbool, "mtbool"},
    {mtdiscrete, "mtdiscrete"},
    {mtcontinuous, "mtcontinuous"},
    {mtset, "mtset"},
    {mttuple, "mttuple"},
    {mtstring, "mtstring"},
    {mtgraph, "mtgraph"}};


class setitr;
class itrpos;

bool setsubseteq( itrpos* in1, itrpos* in2);
bool tupleeq( itrpos* in1, itrpos* in2);

struct valms
{
    measuretype t = mtbool;
    vals v;
    setitr* seti;
//    valms() : valms(mtbool,{},nullptr) {};
//    valms( measuretype tin, vals vin, setitr* setiin );
};

void mtconvertboolto( const bool vin, valms& vout );
void mtconvertdiscreteto( const int vin, valms& vout );
void mtconvertcontinuousto( const double vin, valms& vout );
void mtconvertsetto( setitr* vin, valms& vout );
void mtconverttupleto( setitr* vin, valms& vout );
void mtconvertstringto( std::string* vin, valms& vout );
void mtconvertgraphto( neighborstype* vin, valms& vout );
void mtconverttobool( const valms& vin, bool& vout );
void mtconverttodiscrete( const valms& vin, int& vout );
void mtconverttocontinuous( const valms& vin, double& vout );
void mtconverttoset( const valms& vin, setitr*& vout );
void mtconverttotuple( const valms& vin, setitr*& vout );
void mtconverttostring( const valms& vin, std::string*& vout );
void mtconverttograph( const valms& vin, neighborstype*& vout );

void mtconverttype1( const valms& vin, valms& vout );
void mtconverttype2( const valms& vin, valms& vout );

bool mtareequal( const valms& v1, const valms& v2 );
bool graphsequal( graphtype* g1, graphtype* g2 );

using params = std::vector<valms>;

using namedparams = std::vector<std::pair<std::string,valms>>;

class setitr;

class itrpos;

class setitr
{
public:
    std::vector<valms> totality {};

    itrpos* getitrpos();

    int pos = -1;
    virtual int getsize() {return 0;}
    measuretype t = mtdiscrete;

    virtual void reset() {pos = -1;}
    virtual bool ended() {return pos+1 >= getsize();}
    virtual valms getnext()
    {
        ++pos;
        valms v;
        v.t = mtbool;
        v.v.bv = false;
        return v;
    }
    virtual bool iselt( valms v ) {
        reset();
        while (!ended()) {
            valms u = getnext();
            if (mtareequal(v,u))
                return true;
        }
        return false;
    }
    setitr() {}
    virtual ~setitr();
};

class itrpos
{
public:
    setitr* parent;
    int pos = -1;
    bool ended()
    {
        // std::cout << "itrpos.ended " << pos+1 << ", " << parent->getsize() << "\n";
        return (pos+1 >= parent->getsize());
    }
    void reset() {pos = -1;}
    int getsize() {return parent->getsize();}
    virtual valms getnext()
    {
        if (ended())
            std::cout << "setitr.getnext() called with ended == true\n";
        ++pos;
        while (pos >= parent->totality.size() && !parent->ended())
        {
            // if (parent->pos+1 > parent->totality.size())
            // {
                // std::cout << "parent pos of " << parent->pos << " exceeds parent totality size of " << parent->totality.size() << "\n";
                // parent->reset();
            // }
            while (parent->pos+1 < parent->totality.size())
            {
                std::cout << "pos " << pos << ", parent->pos == " << parent->pos << ", parent->totality.size() == " << parent->totality.size() << std::endl;
                if (parent->ended())
                    break;
                parent->getnext();
            }
            if (!parent->ended())
                parent->getnext();
        }
        if (pos < parent->totality.size())
        {
            //std::cout << "pos == " << pos << std::endl;
            // std::cout << "parent totality pos t == " << parent->totality[pos].t << std::endl;
            // std::cout << "iv " << parent->totality[pos].v.iv << std::endl;

            return parent->totality[pos];
        } else {
            std::cout << "pos == " << pos << " exceeds parent totality size of " << parent->totality.size() << ", ended() == " << this->ended() << "\n";
            exit(1);
        }
    }
    itrpos(setitr* parentin) : parent{parentin} {}
};

inline itrpos* setitr::getitrpos()
{
    auto itr = new itrpos(this);
    //itrs.push_back(itr);
    return itr;
}

inline setitr::~setitr()
{
    // for (itrpos* itr : itrs)
        // delete itr;
    // itrs.clear();
}

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
    case measuretype::mtset:
        { // code below assumes both sets sorted
            std::cout << "early code being used...\n"; // as coded, probably not capable of nested sets...
            auto i1 = a1.seti->getitrpos();
            auto i2 = a2.seti->getitrpos();
            bool match = i1->getsize() == i2->getsize();
            while (match && !i1->ended())
            {
                valms v = i1->getnext();
                match = match && !i2->ended() && v == i2->getnext();
            }
            delete i1;
            delete i2;
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
    virtual void compute() {computed = true;}
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
        return pos+1 >= getsize();
    }
    valms getnext() override
    {
        if (!computed)
        {
            compute();
            computed = true;
            pos = -1;
        }
        if (!ended())
        {
            ++pos;
            return totality[pos];
        } else
        {
            std::cout << "setitrmodeone getnext() called with ended() == true\n";
            valms v;
            v.t = mtdiscrete;
            v.v.iv = 0;
            return v;
        }
    }

    setitrmodeone() {}

    setitrmodeone( std::vector<valms> totalityin )
    {
        totality = totalityin;
        computed = true;
    }

};


class setitrtupleappend : public setitrmodeone {
public:
    setitr* setA;
    setitr* setB;
    public:
    void compute() override
    {
        auto Aitr = setA->getitrpos();
        auto Bitr = setB->getitrpos();
        pos = -1;
        std::vector<valms> temp {};
        // totality.clear();
        while (!Aitr->ended())
        {
            temp.push_back(Aitr->getnext());
            // totality.push_back(Aitr->getnext());
        }
        while (!Bitr->ended())
        {
            temp.push_back(Bitr->getnext());
        }
        totality.resize(temp.size());
        for (int i = 0; i < temp.size(); i++)
            totality[i] = temp[i];
        computed = true;
        reset();
        delete Aitr;
        delete Bitr;
    }
    setitrtupleappend(setitr* a, setitr* b) : setA{a}, setB{b}
    {
        reset();
    }
};

class setitrunion : public setitrmodeone
{
    setitr* setA;
    setitr* setB;
    public:
    void compute() override
    {
        auto Aitr = setA->getitrpos();
        auto Bitr = setB->getitrpos();
        pos = -1;
        std::vector<valms> temp {};
        // totality.clear();
        while (!Aitr->ended())
        {
            temp.push_back(Aitr->getnext());
            // totality.push_back(Aitr->getnext());
        }
        while (!Bitr->ended())
        {
            bool found = false;
            valms v = Bitr->getnext();

            for (int i = 0; !found && i < temp.size(); i++)
                found = mtareequal(temp[i], v);
            /*
                if (v.t == temp[i].t)
                    if (v.t == mtbool || v.t == mtdiscrete || v.t == mtcontinuous)
                        found = found || v == temp[i];
                    else
                    {
                        if (v.t == mtset)
                        {
                            auto tmpitr1 = v.seti->getitrpos();
                            auto tmpitr2 = temp[i].seti->getitrpos();
                            found = found || (setsubseteq( tmpitr1, tmpitr2) && setsubseteq( tmpitr2, tmpitr1 ));
                            delete tmpitr1;
                            delete tmpitr2;
                        } else
                            if (v.t == mttuple)
                            {
                                auto tmpitr1 = v.seti->getitrpos();
                                auto tmpitr2 = temp[i].seti->getitrpos();
                                found = found || tupleeq( tmpitr1, tmpitr2);
                                delete tmpitr1;
                                delete tmpitr2;

                            }
                            else
                                std::cout << "Unsupported type " << v.t << " in setitrunion\n";
                    }
            */
            if (!found)
                temp.push_back(v);
        }
        totality.resize(temp.size());
        for (int i = 0; i < temp.size(); i++)
            totality[i] = temp[i];
        computed = true;
        reset();
        delete Aitr;
        delete Bitr;
    }

    setitrunion(setitr* a, setitr* b) : setA{a}, setB{b}
    {
        reset();
    };

};

class setitrdupeunion : public setitrmodeone
{
    public:
    setitr* setA;
    setitr* setB;

    void compute() override
    {
        auto Aitr = setA->getitrpos();
        auto Bitr = setB->getitrpos();
        pos = -1;
        std::vector<valms> temp {};
        // totality.clear();
        while (!Aitr->ended())
        {
            temp.push_back(Aitr->getnext());
            // totality.push_back(Aitr->getnext());
        }
        while (!Bitr->ended())
        {
            temp.push_back(Bitr->getnext());
        }

        totality.resize(temp.size());
        for (int i = 0; i < temp.size(); i++)
            totality[i] = temp[i];
        computed = true;
        reset();
        delete Aitr;
        delete Bitr;
    }

    setitrdupeunion(setitr* a, setitr* b) : setA{a}, setB{b}
    {
        reset();
    };

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
        pos = -1;
        std::vector<valms> temp {};
        std::vector<valms> temp2 {};
        // totality.clear();
        while (!Aitr->ended())
        {
            temp.push_back(Aitr->getnext());
        }
        while (!Bitr->ended())
        {
            bool found = false;
            valms v = Bitr->getnext();
            for (int i = 0; !found && i < temp.size(); i++)
                found = found || mtareequal(temp[i], v);
            /*
                if (v.t == temp[i].t)
                    if (v.t == mtbool || v.t == mtdiscrete || v.t == mtcontinuous)
                        found = found || v == temp[i];
                    else
                        if (v.t == mtset)
                        {
                            auto tmpitr1 = v.seti->getitrpos();
                            auto tmpitr2 = temp[i].seti->getitrpos();
                            found = found || (setsubseteq( tmpitr1, tmpitr2) && setsubseteq( tmpitr2, tmpitr1 ));
                            delete tmpitr1;
                            delete tmpitr2;
                        } else
                            if (v.t == mttuple)
                            {
                                auto tmpitr1 = v.seti->getitrpos();
                                auto tmpitr2 = temp[i].seti->getitrpos();
                                found = found || tupleeq( tmpitr1, tmpitr2);
                                delete tmpitr1;
                                delete tmpitr2;

                            }
                            else
                                std::cout << "Unsupported type " << v.t << " in setitrintersection\n";
                */
            if (found)
                temp2.push_back(v);
        }
        totality.resize(temp2.size());
        for (int i = 0; i < temp2.size(); i++)
            totality[i] = temp2[i];
        computed = true;
        reset();
        delete Aitr;
        delete Bitr;
    }

    setitrintersection(setitr* a, setitr* b) : setA{a}, setB{b}
    {
        reset();
    };

};

class setitrsetminus : public setitrmodeone
{
    public:
    setitr* setA;
    setitr* setB;
    void compute() override
    {
        auto Aitr = setA->getitrpos();
        auto Bitr = setB->getitrpos();
        pos = -1;
        totality.clear();
        std::vector<valms> temp {};
        while (!Aitr->ended())
        {
            bool found = false;
            valms v = Aitr->getnext();
            Bitr->reset();
            while (!found && !Bitr->ended()) {
                valms omitv = Bitr->getnext();
                found = found || mtareequal(v,omitv);
                /*
                if (v.t == omitv.t)
                    switch (v.t) {
                        case mtbool:
                        case mtdiscrete:
                        case mtcontinuous:
                        found = found || v == omitv;
                        break;
                        case mtset: {
                        case mttuple:
                            found = mtareequal(v, omitv);
                            break;}
                        default:
                            std::cout << "Unsupported type " << v.t << " in setitrunion\n";
                    }*/
            }
            if (!found)
                temp.push_back(v);
        }
        totality.resize(temp.size());
        for (int i = 0; i < temp.size(); i++)
            totality[i] = temp[i];
        computed = true;
        reset();
        delete Aitr;
        delete Bitr;
    }

    setitrsetminus(setitr* a, setitr* b) : setA{a}, setB{b}
    {
        reset();
    };

};


class setitrsetxor : public setitrmodeone
{
public:
    setitr* setA;
    setitr* setB;

    void compute() override
    {
        auto Aitr = setA->getitrpos();
        auto Bitr = setB->getitrpos();
        pos = -1;
        std::vector<valms> temp {};
        std::vector<valms> tempA {};
        std::vector<valms> tempB {};
        while (!Aitr->ended())
            tempA.push_back(Aitr->getnext());
        while (!Bitr->ended())
            tempB.push_back(Bitr->getnext());
        for (int i = 0; i < tempB.size(); i++)
        {
            valms v = tempB[i];
            bool found = false;
            int j = 0;
            while (!found && j < tempA.size()) {
                valms omitv = tempA[j++];
                found = found || mtareequal(v, omitv);
                /*
                if (v.t == omitv.t)
                    switch (v.t) {
                        case mtbool:
                        case mtdiscrete:
                        case mtcontinuous:
                        found = found || v == omitv;
                        break;
                        case mtset: {
                            auto tmpitr1 = v.seti->getitrpos();
                            auto tmpitr2 = omitv.seti->getitrpos();
                            found = found || (setsubseteq( tmpitr1, tmpitr2) && setsubseteq( tmpitr2, tmpitr1 ));
                            delete tmpitr1;
                            delete tmpitr2;
                            break;}
                        case mttuple: {
                            auto tmpitr1 = v.seti->getitrpos();
                            auto tmpitr2 = omitv.seti->getitrpos();
                            found = found || tupleeq( tmpitr1, tmpitr2);
                            delete tmpitr1;
                            delete tmpitr2;
                            break;}
                        default:
                            std::cout << "Unsupported type " << v.t << " in setitrunion\n";
                    }
                */
            }
            if (!found)
                temp.push_back(v);
            else
                tempA.erase(tempA.begin() + j - 1);
        }
        for (auto v2 : tempA)
            temp.push_back(v2);
        totality.clear();
        for (auto v : temp)
            totality.push_back(v);
        computed = true;
        reset();
        delete Aitr;
        delete Bitr;
    }

    setitrsetxor(setitr* a, setitr* b) : setA{a}, setB{b}
    {
        reset();
    }

};

inline void fastsetunion( const int maxint1, const int maxint2, const int maxintout, bool* elts1, bool* elts2, bool* out);


class setitrint : public setitrmodeone // has subset functionality built in
{
    public:
    int maxint;
    bool* elts;
    void compute() override
    {
        std::vector<valms> temp {};
        temp.resize(maxint+1);
        int j = 0;
        for (int i = 0; i <= maxint; ++i)
        {
            if (elts[i])
            {
                temp[j].t = measuretype::mtdiscrete;
                temp[j++].v.iv = i;
            }
        }
        totality.resize(j);
        for (int i = 0; i < j; ++i)
            totality[i] = temp[i];
        computed = true;
        reset();
    }
    void setmaxint( const int maxintin )
    {
        delete elts;
        maxint = maxintin;
        if (maxint >= 0)
        {
            elts = (bool*)malloc((maxint+1)*sizeof(bool));
            memset(elts, true, (maxint+1)*sizeof(bool));
        } else
            elts = nullptr;
        computed = false;
        // totality.clear();
        reset();
    }
    virtual bool iselt( valms v ) {
        if (!computed)
            compute();
        return elts[v.v.iv];
    }
    setitrint(const int maxintin) : elts{nullptr}
    {
        setmaxint(maxintin);
        t = mtdiscrete;
        reset();
    }
    setitrint(const int maxintin, bool* eltsin)
    {
        maxint = maxintin;
        elts = eltsin;
        t = mtdiscrete;
        computed = false;
        reset();
    }
    ~setitrint() {
         delete elts;
    }

};


template<typename T>
class setitrtuple : public setitrmodeone
{
public:
    T* elts = nullptr;
    int length;
    int maxelt = -1;

    virtual valms assignvalms( T elt )
    {
        std::cout << "Abstract ancestor assignvalms called\n";
        valms v; v.t = mtbool; v.v.bv = false; return v;
    }

    void compute() override
    {
        totality.resize(length);
        for (int i = 0; i < length; ++i) {
            totality[i] = assignvalms(elts[i]);
            maxelt = (i == 0 || elts[i] > maxelt) ? elts[i] : maxelt;
        }
        computed = true;
        reset();
    }
    void setlength( const int lengthin )
    {
        delete elts;
        length = lengthin;
        if (length > 0)
        {
            elts = (T*)malloc(length*sizeof(T));
            // memset(elts, true, (maxint+1)*sizeof(T));
        } else
            elts = nullptr;
        computed = false;
        // totality.clear();
        reset();
    }
    setitrtuple(const int lengthin) : elts(nullptr)
    {
        setlength(lengthin);
        reset();
    }
    setitrtuple(const int lengthin, T* eltsin) : elts(eltsin), length(lengthin)
    {
        reset();
    }
    setitrtuple(std::vector<T>& vecin)
    {
        setlength(vecin.size());
        for (int i = 0; i < length; ++i)
            elts[i] = vecin[i];
        reset();
    }
    ~setitrtuple() {
         delete elts;
    }
};

template<>
inline valms setitrtuple<int>::assignvalms( int elt ) {
    valms v;
    v.t = mtdiscrete;
    v.v.iv = elt;
    return v;
}
template<>
inline valms setitrtuple<double>::assignvalms( double elt ) {
    valms v;
    v.t = mtcontinuous;
    v.v.dv = elt;
    return v;
}
template<>
inline valms setitrtuple<bool>::assignvalms( bool elt ) {
    valms v;
    v.t = mtbool;
    v.v.bv = elt;
    return v;
}

class setitrintpair : public setitrmodeone
{
protected:
    int inta;
    int intb;
    public:
    void compute() override
    {
        totality.resize(2);
        totality[0].t = mtdiscrete;
        totality[1].t = mtdiscrete;
        totality[0].v.iv = inta;
        totality[1].v.iv = intb;
    }
    public:
    setitrintpair(int intain, int intbin) : inta{intain}, intb{intbin} {}
};
class setitrint2d : public setitrmodeone {
public:
    setitrint* itrint;
    int dim1;
    int dim2;
    void compute() override {
        std::vector<valms> temp;
        temp.resize(dim1*dim2);
        int k = 0;
        for (int i = 0; i < dim1; ++i)
            for (int j = 0; j < dim2; ++j) {
                if (itrint->elts[i*dim1 + j]) {
                    valms v;
                    v.t = mtset;
                    v.seti = new setitrintpair(i,j);
                    temp[k++] = v;
                }
            }

        temp.resize(k);
        for (auto v : totality)
            delete v.seti;
        totality.resize(temp.size());
        int i = 0;
        for (auto v : temp)
            totality[i++] = v;

        computed = true;
        reset();
    }
    setitrint2d( int dim1in, int dim2in) : dim1{dim1in}, dim2{dim2in}, itrint{new setitrint(dim1in*dim2in-1)} {}
    setitrint2d( int dim1in, int dim2in, bool* elts) : dim1{dim1in}, dim2{dim2in}, itrint{new setitrint(dim1in*dim2in - 1, elts)} {}
    setitrint2d( int dim1in, int dim2in, setitrint* itrintin ) : dim1{dim1in}, dim2{dim2in}, itrint{itrintin} {}
    ~setitrint2d()
    {
        delete itrint;
    }
};
inline void demirrorelts( const int dim1, bool* elts) {
    for (int i = 0; i < dim1; ++i)
        for (int j = 0; j < i; ++j)
            elts[i*dim1 + j] = false;
}
class setitrint2dsymmetric : public setitrint2d {
public:
    void compute() override {
        // setitrint2d::compute();
        // the below is almost identical to the inherited... just a bit faster on the inner for loop

        std::vector<valms> temp;
        temp.resize(dim1*dim2);
        int k = 0;
        for (int i = 0; i+1 < dim1; ++i)
            for (int j = i+1; j < dim2; ++j) {
                if (itrint->elts[i*dim1 + j]) {
                    valms v;
                    v.t = mtset;
                    v.seti = new setitrintpair(i,j);
                    temp[k++] = v;
                }
            }

        temp.resize(k);
        for (auto v : totality)
            delete v.seti;
        totality.resize(temp.size());
        int i = 0;
        for (auto v : temp)
            totality[i++] = v;

        computed = true;
        reset();
    }
    setitrint2dsymmetric(int dim1in) : setitrint2d(dim1in,dim1in) {}
    setitrint2dsymmetric(int dim1in, bool* eltsin) : setitrint2d(dim1in,dim1in,new bool[dim1in*dim1in])
    {
        memcpy(this->itrint->elts,eltsin,dim1*dim1*sizeof(bool));
        demirrorelts(dim1,this->itrint->elts);
    }
    setitrint2dsymmetric(int dim1in, setitrint* itrintin ) : setitrint2d(dim1in,dim1in, itrintin) {}
    ~setitrint2dsymmetric() {
        for (auto v : totality)
            delete v.seti;
    }

};

class setitrsubset : public setitr {
public:
    setitrint* itrint;
    itrpos* superset {};
    void setsuperset( itrpos* supersetposin )
    {
        superset = supersetposin;
        itrint->setmaxint(superset->getsize() - 1);
        reset();
    }

    int getsize() override
    {
        return itrint->getsize();
    }

    valms getnext() override
    {
        if (pos+1 < totality.size())
            return totality[++pos];

        if (!itrint->computed)
            itrint->compute();
        valms res;
        int superpos = -1;
        if (!itrint->ended())
            superpos = itrint->totality[++pos].v.iv;
        if (superpos >= 0)
        {
            if (superset->pos+1 > superpos)
                res = superset->parent->totality[superpos];
            else
            {
                while (superset->pos+1 < superpos+1 && !superset->ended())
                    res = superset->getnext();
                if (superpos > superset->pos)
                {
                    std::cout << "Error in setitrsubset, parent size exceeded\n";
                    exit(1);
                }
            }
        } else
        {
            std::cout << "Error itrbool totality item is negative, in pass to setitrsubset" << std::endl;
        }
        // std::cout << "res.t == " << res.t << std::endl;
        totality.resize(pos + 1);
        if (pos >= 0)
            totality[pos] = res;
        return res;
        // may add error catching code here
    }
    void reset() override
    {
        superset->reset();
        itrint->reset();
        pos = -1;
    }
    bool ended() override
    {
        if (itrint->computed)
            return pos + 1 >= itrint->totality.size();
        else
            return pos + 1 >= itrint->getsize();
    }
    setitrsubset(itrpos* supersetin) : superset{supersetin}, itrint{new setitrint(supersetin->getsize() - 1)}
    {
        t = superset->parent->t;
        pos = -1;
    };
    setitrsubset(itrpos* supersetin, setitrint* itrintin) : superset{supersetin}, itrint{itrintin} {
        t = superset->parent->t;
        pos = -1;
    }
    setitrsubset() : superset{}, itrint{new setitrint(-1)}
    {
        t = mtdiscrete;
        pos = -1;
    };
    ~setitrsubset() {
        delete itrint;
        delete superset;
    }
};

inline void fastsetunion( const int maxint1, const int maxint2, const int maxintout, bool* elts1, bool* elts2, bool* out) {
    if (maxint1 > 0)
        memcpy(out,elts1,(maxint1+1)*sizeof(bool));
    // for (int i = 0; i <= maxint1; ++i)
    //    out[i] = elts1[i];
     for (int i = 0; i <= maxint2; ++i)
         out[i] = out[i] || elts2[i];
}
inline void fastsetintersection( const int maxint1, const int maxint2, const int maxintout, bool* elts1, bool* elts2, bool* out) {
    for (int i = 0; i <= maxintout; ++i)
        out[i] = elts1[i] && elts2[i];
}
inline void fastsetminus( const int maxint1, const int maxint2, const int maxintout, bool* elts1, bool* elts2, bool* out) {
    int i;
    for (i = 0; i <= maxintout; ++i)
        out[i] = elts1[i] && !elts2[i];
    if (i <= maxint1)
        memcpy(out+i,elts1+i,(maxint1+1 - i)*sizeof(bool));
    // for ( ; i <= maxint1; ++i)
    //    out[i] = elts1[i];
}
inline void fastsetxor( const int maxint1, const int maxint2, const int maxintout, bool* elts1, bool* elts2, bool* out) {
    if (maxint1 <= maxint2) {
        int i;
        for (i = 0; i <= maxint1; ++i)
            out[i] = elts1[i] != elts2[i];
        if (i <= maxint2)
            memcpy(out+i,elts2 + i,(maxint2+1 - i)*sizeof(bool));
        // for (; i <= maxint2; ++i)
        //    out[i] =  elts2[i];
    } else {
        int i;
        for (i = 0; i <= maxint2; ++i)
            out[i] = elts1[i] != elts2[i];
        if (i <= maxint1)
            memcpy(out+i,elts1 + i,(maxint1+1 - i)*sizeof(bool));
        // for (; i <= maxint1; ++i)
        //     out[i] =  elts1[i];
    }
}
inline bool fastsetsubset( const int maxint1, const int maxint2, bool* elts1, bool* elts2) {
    if (maxint1 <= maxint2) {
        int i;
        for (i = 0; i <= maxint1; ++i)
            if (elts1[i] && !elts2[i])
                return false;
    } else {
        int i;
        for (i = 0; i <= maxint2; ++i)
            if (elts1[i] && !elts2[i])
                return false;
        for (; i <= maxint1; ++i)
            if (elts1[i])
                return false;
    }
    return true;
}
inline bool fastsetpropersubset( const int maxint1, const int maxint2, bool* elts1, bool* elts2) {
    bool diff = false;
    if (maxint1 <= maxint2) {
        int i;
        for (i = 0; i <= maxint1; ++i)
            if (!elts2[i]) {
                if (elts1[i])
                    return false;
            } else
                if (!elts1[i])
                    diff = true;
        for (; i <= maxint2 && !diff; ++i)
            diff = elts2[i];
    } else {
        int i;
        for (i = 0; i <= maxint2; ++i)
            if (!elts2[i]) {
                if (elts1[i])
                    return false;
            } else
                if (!elts1[i])
                    diff = true;
        for (; i <= maxint1; ++i)
            if (elts1[i])
                return false;
    }
    return diff;
}
inline bool fastsetequals( const int maxint1, const int maxint2, bool* elts1, bool* elts2) {
    if (maxint1 <= maxint2) {
        int i;
        for (i = 0; i <= maxint1; ++i)
            if (elts1[i] != elts2[i])
                return false;
        for (; i <= maxint2; ++i)
            if (elts2[i])
                return false;
    } else {
        int i;
        for (i = 0; i <= maxint2; ++i)
            if (elts1[i] != elts2[i])
                return false;
        for (; i <= maxint1; ++i)
            if (elts1[i])
                return false;
    }
    return true;
}
inline bool fastsetmeet( const int maxint1, const int maxint2, bool* elts1, bool* elts2) {
    if (maxint1 <= maxint2) {
        int i;
        for (i = 0; i <= maxint1; ++i)
            if (elts1[i] && elts2[i])
                return true;
    } else {
        int i;
        for (i = 0; i <= maxint2; ++i)
            if (elts1[i] && elts2[i])
                return true;
    }
    return false;
}
inline bool fastboolsetops( setitrint* setA, setitrint* setB, const formulaoperator fo ) {
    int maxintA = setA->maxint;
    int maxintB = setB->maxint;
    switch (fo) {
        case formulaoperator::folte:
            return fastsetsubset( maxintA, maxintB, setA->elts, setB->elts );
        case formulaoperator::folt:
            return fastsetpropersubset( maxintA, maxintB, setA->elts, setB->elts );
        case formulaoperator::fogte:
            return fastsetsubset( maxintB, maxintA, setB->elts, setA->elts );
        case formulaoperator::fogt:
            return fastsetpropersubset( maxintB, maxintA, setB->elts, setA->elts );
        case formulaoperator::foe:
            return fastsetequals( maxintA, maxintB, setA->elts, setB->elts );
        case formulaoperator::fone:
            return !fastsetequals( maxintA, maxintB, setA->elts, setB->elts );
        case formulaoperator::fomeet:
            return fastsetmeet( maxintA, maxintB, setA->elts, setB->elts );
        case formulaoperator::fodisjoint:
            return !fastsetmeet( maxintA, maxintB, setA->elts, setB->elts );
    }
    return false;
}
inline setitrint* fastsetops( setitrint* setA, setitrint* setB, const formulaoperator fo ) {
    auto maxintA = setA->maxint;
    auto maxintB = setB->maxint;
    int maxintout;
    switch (fo) {
        case formulaoperator::fodupeunion:
        case formulaoperator::founion:
        case formulaoperator::fosetxor:
            maxintout = maxintA >= maxintB ? maxintA : maxintB;
            break;
        case formulaoperator::fointersection:
            maxintout = maxintA <= maxintB ? maxintA : maxintB;
            break;
        case formulaoperator::fosetminus:
            maxintout = maxintB <= maxintA ? maxintB : maxintA;
            break;
        default:
            std::cout << "Cannot call fastsetops with non-set operator\n";
    }
    auto out = new setitrint( maxintout );
    switch (fo) {
        case formulaoperator::fodupeunion:
        case formulaoperator::founion:
            fastsetunion( maxintA, maxintB, maxintout, setA->elts, setB->elts, out->elts );
            break;
        case formulaoperator::fointersection:
            fastsetintersection( maxintA, maxintB, maxintout, setA->elts, setB->elts, out->elts );
            break;
        case formulaoperator::fosetminus:
            fastsetminus( maxintA, maxintB, maxintout, setA->elts, setB->elts, out->elts );
            break;
        case formulaoperator::fosetxor:
            fastsetxor( maxintA, maxintB, maxintout, setA->elts, setB->elts, out->elts );
            break;
        default:
            std::cout << "Cannot call fastsetops with non-set operator\n";
    }
    out->computed = false;
    out->reset();
    return out;
}
/* for now the set-valued set operations are the same for tuples as for set */
template<typename T>
inline void fasttupleunion( const int lengthA, const int lengthB, const int length, T* eltsA, T* eltsB, T* out) {
    memcpy(out, eltsA, lengthA * sizeof(T));
    memcpy(out + lengthA, eltsB, lengthB * sizeof(T));
}
template<typename T>
inline void fasttuplesetminus( const int lengthA, const int lengthB, int& length, T* eltsA, T* eltsB, T* out) {
    bool deprecated[lengthA];
    memset(deprecated, false, lengthA * sizeof(T));
    for (int i = 0; i < lengthA; ++i)
        for (int j = 0; !deprecated[i] && j < lengthB; ++j) {
            deprecated[i] = deprecated[i] || (eltsA[i] == eltsB[j]);
        }
    int pos = 0;
    for (int i = 0; i < lengthA; ++i)
        if (!deprecated[i]) {
           out[pos++] = eltsA[i];
           length = pos;
        }
}
template<typename T>
inline void fasttupleintersection( const int lengthA, const int lengthB, int& length, T* eltsA, T* eltsB, T* out) {
// the convention here is that the "tuple" on the right be treated as a set, so this is like subtracting its complement
    bool deprecated[lengthA];
    memset(deprecated, true, lengthA * sizeof(T));
    for (int i = 0; i < lengthA; ++i)
        for (int j = 0; deprecated[i] && j < lengthB; ++j) {
            deprecated[i] = deprecated[i] && (eltsA[i] != eltsB[j]);
        }
    int pos = 0;
    for (int i = 0; i < lengthA; ++i)
        if (!deprecated[i]) {
           out[pos++] = eltsA[i];
           length = pos;
        }
}
template<typename T>
inline bool fasttuplemeet( const int lengthA, const int lengthB, const int n, T* eltsA, T* eltsB) {
    const int length = lengthA < lengthB ? lengthA : lengthB;
    int j = 0;
    for (auto i = 0; i < length; i++) {
        if (eltsA[i] == eltsB[i]) {
            ++j;
            if (j >= n)
                return true;
        }
    }
    return false;
}
template<typename T>
inline bool fasttupledisjoint( const int lengthA, const int lengthB, const int n, T* eltsA, T* eltsB) {
    return !fasttuplemeet( lengthA, lengthB, n, eltsA, eltsB );
}
template<typename T>
inline bool fasttupleequal( const int lengthA, const int lengthB, T* eltsA, T* eltsB) {
    if (lengthA != lengthB)
        return false;
    for (auto i = 0; i < lengthA; ++i) {
        if (eltsA[i] != eltsB[i])
            return false;
    }
    return true;
}
template<typename T>
inline bool fasttupleinitialsegment( const int lengthA, const int lengthB, T* eltsA, T* eltsB ) {
    bool res = lengthA <= lengthB;
    for (int i = 0; res && i < lengthA; i++) {
        res = res && eltsA[i] == eltsB[i];
    }
    return res;
}
inline bool slowtupleinitialsegment( itrpos* tupleA, itrpos* tupleB ) {
    auto itrA = tupleA;
    auto itrB = tupleB;
    while (!itrA->ended()) {
        if (itrB->ended())
            return false;
        valms v = itrA->getnext();
        valms w = itrB->getnext();
        if (!mtareequal( v, w ))
            return false;
    }
    return true;
}
inline bool slowtuplemeet( itrpos* tupleA, itrpos* tupleB, const int n ) {
    int cnt = 0;
    tupleA->reset();
    tupleB->reset();
    while (!tupleA->ended()) {
        auto a = tupleA->getnext();
        while (!tupleB->ended()) {
            auto b = tupleB->getnext();
            if (mtareequal( a, b )) {
                cnt++;
                if (cnt >= n)
                   return true;
            }
        }
    }
    return false;
}
inline bool slowtupledisjoint( itrpos* tupleA, itrpos* tupleB, const int n ) {
    return !slowtuplemeet( tupleA, tupleB, n );
}
template<typename T>
inline bool fastbooltupleops( setitrtuple<T>* tupleA, setitrtuple<T>* tupleB, const formulaoperator fo ) {
    auto lengthA = tupleA->length;
    auto lengthB = tupleB->length;
    switch (fo) {
        case formulaoperator::folte:
            return fasttupleinitialsegment<T>( lengthA, lengthB, tupleA->elts, tupleB->elts );
        case formulaoperator::folt:
            return lengthA < lengthB && fasttupleinitialsegment<T>( lengthA, lengthB, tupleA->elts, tupleB->elts );
        case formulaoperator::fogte:
            return fasttupleinitialsegment<T>( lengthB, lengthA, tupleB->elts, tupleA->elts );
        case formulaoperator::fogt:
            return lengthB < lengthA && fasttupleinitialsegment<T>( lengthB, lengthA, tupleB->elts, tupleA->elts );
        case formulaoperator::foe:
            return fasttupleequal<T>( lengthA, lengthB, tupleA->elts, tupleB->elts );
        case formulaoperator::fone:
            return !fasttupleequal<T>( lengthA, lengthB, tupleA->elts, tupleB->elts );
        case formulaoperator::fomeet:
            return fasttuplemeet<T>( lengthA, lengthB, 1, tupleA->elts, tupleB->elts );
        case formulaoperator::fodisjoint:
            return fasttupledisjoint<T>( lengthA, lengthB, 1, tupleA->elts, tupleB->elts );
        default:
            std::cout << "Non-tuple operation applied to tuple\n";
    }
    return false;
}
template<typename T>
inline setitrtuple<T>* fasttupleops( setitrtuple<T>* tupleA, setitrtuple<T>* tupleB, const formulaoperator fo ) {
    auto lengthA = tupleA->length;
    auto lengthB = tupleB->length;
    int length;
    switch (fo) {
        case formulaoperator::fodupeunion:
        case formulaoperator::founion:
            length = lengthA + lengthB;
            break;
        case formulaoperator::fosetminus:
            length = lengthA;
            break;
        case formulaoperator::fointersection:
            length = lengthA < lengthB ? lengthA : lengthB;
            break;
        default:
            std::cout << "Cannot call fasttupleops with non-tuple operator\n";
    }
    auto out = new setitrtuple<T>(length);
    switch (fo) {
        case formulaoperator::fodupeunion:
        case formulaoperator::founion:
            fasttupleunion<T>( lengthA, lengthB, length, tupleA->elts, tupleB->elts, out->elts );
            break;
        case formulaoperator::fosetminus:
            fasttuplesetminus<T>( lengthA, lengthB, length, tupleA->elts, tupleB->elts, out->elts );
            out->length = length;
            break;
        case formulaoperator::fointersection:
            fasttupleintersection<T>( lengthA, lengthB, length, tupleA->elts, tupleB->elts, out->elts );
            out->length = length;
            break;
    }
    out->computed = false;
    out->reset();
    return out;
}
class setitrabstractops : public setitrmodeone {
public:
    virtual int setopunioncount( const int cutoff, const formulaoperator fo ) {std::cout << "Error: abstract empty method called\n";}
    virtual int setopintersectioncount( const int cutoff, const formulaoperator fo ) {std::cout << "Error: abstract empty method called\n";}
    virtual setitr* setops( const formulaoperator fo ) {auto out = new setitrint(-1); return out;};
    virtual bool boolsetops( const formulaoperator fo ) {return false;};
};
class setitrfastops : public setitrabstractops {
public:
    setitrint* castA;
    setitrint* castB;

    setitr* setops( const formulaoperator fo )  override {
        auto out = fastsetops( castA, castB, fo );
        return out;
    }
    bool boolsetops( const formulaoperator fo ) override {
        return fastboolsetops( castA, castB, fo );
    }
    setitrfastops( setitrint* castAin, setitrint* castBin ) : castA{castAin}, castB{castBin} {}
};
class setitrfastpluralops : public setitrabstractops {
public:
    std::vector<setitrint*> casts;

    setitr* setops( const formulaoperator fo )  override {
        int maxint = -1;
        switch (fo) {
        case formulaoperator::foqdupeunion:
        case formulaoperator::foqunion: {
            for (auto c : casts)
                maxint = maxint > c->maxint ? maxint : c->maxint;
            break; }
        case formulaoperator::foqintersection: {
            for (auto c : casts)
                maxint = maxint < c->maxint ? maxint : c->maxint;
            break; }
        default: std::cout << "No support for fast plural set ops other than union and intersection\n";
            maxint = -1;
            exit(1);
        }
        auto out = new setitrint( maxint );
        switch (fo) {
        case formulaoperator::foqdupeunion:
        case formulaoperator::foqunion: {
            memset(out->elts,false,sizeof(bool)*(maxint + 1));
            for (auto i = 0; i <= maxint; ++i)
                for (auto j = 0; j < casts.size(); ++j)
                    if (casts[j]->maxint >= i)
                        if (casts[j]->elts[i]) {
                            out->elts[i] = true;
                            break;
                        }
            break;
        }
        case formulaoperator::foqintersection: {
            memset(out->elts,true,sizeof(bool)*(maxint + 1));
            for (auto i = 0; i <= maxint; ++i)
                for (auto j = 0; j < casts.size(); ++j)
                    if (!casts[j]->elts[i]) {
                        out->elts[i] = false;
                        break;
                    }
            break;
        }

        }
        if (casts.size() < 1) {
            // std::cout << "Less than one set passed to plural ops\n";
        }
        return out;
    }
    int setopunioncount( const int cutoff, const formulaoperator fo ) override {
        if (fo != formulaoperator::founion) {
            std::cout << "No support for plural set bool ops other than for 'meet' and 'disjoint'; try pairwise (iv)\n";
            return 0;
        }
        int maxsz;
        if (casts.size() > 0)
            maxsz = casts[0]->maxint +1;
        for (int i = 1; i < casts.size(); ++i)
            maxsz = maxsz > casts[i]->maxint+1 ? maxsz : casts[i]->maxint + 1;

        bool* elts = new bool[maxsz];
        memset( elts, false, sizeof(bool) * maxsz );
        for (int j = 0; j < casts.size(); ++j) {
            for (int i = 0; i <= casts[j]->maxint; ++i) {
                elts[i] = elts[i] || (casts[j]->elts[i]);
            }
        }
        int cnt = 0;
        for (int i = 0; i < maxsz; ++i) {
            if (elts[i]) {
                ++cnt;
                if (cnt >= cutoff && cutoff != -1)
                    break;
            }
        }
        delete elts;
        return cnt;
    }
    int setopintersectioncount( const int cutoff, const formulaoperator fo ) override {
        if (fo != formulaoperator::fointersection) {
            std::cout << "No support for plural set bool ops other than for 'meet' and 'disjoint'; try pairwise (iii)\n";
            return 0;
        }
        int minsz;
        if (casts.size() > 0)
            minsz = casts[0]->maxint +1;
        for (int i = 1; i < casts.size(); i++)
            minsz = minsz < casts[i]->maxint+1 ? minsz : casts[i]->maxint + 1;

        bool* elts = new bool[minsz];
        int cnt = 0;
        memset( elts, true, sizeof(bool) * minsz );
        for (int i = 0; i < minsz; ++i) {
            for (int j = 0; j < casts.size(); ++j) {
                elts[i] = elts[i] && casts[j]->elts[i];
            }
            if (elts[i]) {
                ++cnt;
                if (cnt >= cutoff && cutoff != -1)
                    break;
            }
        }
        delete elts;
        return cnt;
    }
    bool boolsetops( const formulaoperator fo ) override {
        if (casts.size() < 2) {
            std::cout << "Less than two sets passed to plural bool ops\n";
            return false;
        }
        if (fo != formulaoperator::fomeet && fo != formulaoperator::fodisjoint) {
            std::cout << "No support for plural set bool ops other than for 'meet' and 'disjoint'; try pairwise (ii)\n";
            return false;
        }
        if (fo == formulaoperator::fomeet)
            return setopintersectioncount( 1, formulaoperator::fointersection ) > 0 ? true : false;
        else
            return setopintersectioncount(1, formulaoperator::fointersection ) < 1 ? true : false;
    }
    setitrfastpluralops( std::vector<setitrint*> castsin ) : casts{castsin} {}
};
class setitrfastpluralssops : public setitrabstractops {
public:
    std::vector<setitrsubset*> castsss;
    setitrfastpluralops* fastpluralops {};

    bool boolsetops( const formulaoperator fo )  override {
        return fastpluralops->boolsetops( fo );
    }
    int setopunioncount( const int max, const formulaoperator fo ) override {
        return fastpluralops->setopunioncount( max, fo );
    }
    int setopintersectioncount( const int min, const formulaoperator fo ) override {
        return fastpluralops->setopintersectioncount( min, fo );
    }
    setitr* setops( const formulaoperator fo ) override {
        return fastpluralops->setops( fo );
    }

    setitrfastpluralssops( std::vector<setitrsubset*>& castsss ) : castsss{castsss} {
        std::vector<setitrint*> casts {};
        for (auto s : castsss)
            casts.push_back(s->itrint);
        fastpluralops = new setitrfastpluralops(casts);
    }
    ~setitrfastpluralssops() {
        delete fastpluralops;
    }
};
class setitrfastssops : public setitrabstractops {
public:
    setitrsubset* castAss;
    setitrsubset* castBss;

    setitr* setops( const formulaoperator fo )  override {
        auto itrint = fastsetops( castAss->itrint, castBss->itrint, fo );
        auto out = new setitrsubset( castAss->superset, itrint );
        return out;
    }
    bool boolsetops( const formulaoperator fo ) override {
        return fastboolsetops( castAss->itrint, castBss->itrint, fo );
    }
    setitrfastssops( setitrsubset* castAssin, setitrsubset* castBssin ) : castAss{castAssin}, castBss{castBssin} {}
};
class setitrfastplural2dops : public setitrabstractops {
public:
    std::vector<setitrint2d*> casts2d;
    setitrfastpluralops* fastpluralops {};

    bool boolsetops( const formulaoperator fo )  override {
        return fastpluralops->boolsetops( fo );
    }

    int setopintersectioncount( const int min, const formulaoperator fo ) override {
        return fastpluralops->setopintersectioncount( min, fo );
    }
    setitr* setops( const formulaoperator fo ) override {
        return fastpluralops->setops( fo );
    }

    setitrfastplural2dops( std::vector<setitrint2d*>& casts2d ) : casts2d{casts2d} {
        std::vector<setitrint*> casts {};
        for (auto s : casts2d)
            casts.push_back(s->itrint);
        fastpluralops = new setitrfastpluralops(casts);
    }
    ~setitrfastplural2dops() {
        delete fastpluralops;
    }
};
class setitrfast2dops : public setitrabstractops {
public:
    setitrint2d* castA2d;
    setitrint2d* castB2d;

    setitr* setops( const formulaoperator fo )  override {
        auto itrint = fastsetops( castA2d->itrint, castB2d->itrint, fo );
        int m = castA2d->dim1;
        if (m*m != itrint->maxint + 1)
            m = castB2d->dim1; // fussy but need to recover the newly sized square
        auto out = new setitrint2dsymmetric( m, itrint );
        return out;
    }
    bool boolsetops( const formulaoperator fo ) override {
        return fastboolsetops( castA2d->itrint, castB2d->itrint, fo );
    }
    setitrfast2dops( setitrint2d* castA2din, setitrint2d* castB2din ) : castA2d{castA2din}, castB2d{castB2din} {}
};

class setitrslowops : public setitrabstractops {
public:
    setitr* setA;
    setitr* setB;
    setitr* setops( const formulaoperator fo ) override {
        switch (fo) {
            case formulaoperator::founion:
                return new setitrunion( setA, setB );
            case formulaoperator::fodupeunion:
                return new setitrdupeunion( setA, setB );
            case formulaoperator::fointersection:
                return new setitrintersection( setA, setB );
            case formulaoperator::fosetminus:
                return new setitrsetminus( setA, setB );
            case formulaoperator::fosetxor:
                return new setitrsetxor( setA, setB );
            default:
                std::cout << "setops (slow) called with non-set operator\n";
                return new setitrunion( setA, setB );
        }
    }
    bool boolsetops( const formulaoperator fo ) override {
        auto tmpitrA = setA->getitrpos();
        auto tmpitrB = setB->getitrpos();
        bool out;
        switch (fo) {
            case formulaoperator::folte:
                out = setsubseteq( tmpitrA, tmpitrB );
                break;
            case formulaoperator::folt:
                out = setsubseteq( tmpitrA, tmpitrB ) && !setsubseteq( tmpitrB, tmpitrA );
                break;
            case formulaoperator::fogte:
                out = setsubseteq( tmpitrB, tmpitrA );
                break;
            case formulaoperator::fogt:
                out = setsubseteq( tmpitrB, tmpitrA ) && !setsubseteq( tmpitrA, tmpitrB );
                break;
            case formulaoperator::foe:
                out = setsubseteq( tmpitrB, tmpitrA ) && setsubseteq( tmpitrA, tmpitrB );
                break;
            case formulaoperator::fone:
                out = !setsubseteq( tmpitrB, tmpitrA ) || !setsubseteq( tmpitrA, tmpitrB );
                break;
            case formulaoperator::fomeet: {
                auto tmp = new setitrintersection( setA, setB );
                out = tmp->getsize() > 0;
                delete tmp;
                break;}
            case formulaoperator::fodisjoint: {
                auto tmp = new setitrintersection( setA, setB );
                out = tmp->getsize() == 0;
                delete tmp;
                break;}
            default:
                std::cout << "boolsetops (slow) called with non-set or non boolean operator\n";
                break;
        }
        delete tmpitrA;
        delete tmpitrB;
        return out;
    }
    setitrslowops( setitr* setAin, setitr* setBin ) : setA{setAin}, setB{setBin} {}
};
class setitrslowpluralops : public setitrabstractops {
public:
    std::vector<setitr*> sets;
    setitr* setops( const formulaoperator fo ) override {

        setitr* res = nullptr;
        if (sets.empty())
            res = new setitrint(-1);
        else
            res = sets.back();
        switch (fo) {
            case formulaoperator::foqunion:
                for (int i = sets.size() - 2; i >= 0; --i)
                    res = new setitrunion( res, sets[i] );
                break;
            case formulaoperator::foqdupeunion:
                for (int i = sets.size() - 2; i >= 0; --i)
                    res = new setitrdupeunion( res, sets[i] );
                break;
            case formulaoperator::foqintersection:
                for (int i = sets.size() - 2; i >= 0; --i)
                    res = new setitrintersection( res, sets[i] );
                break;
            default:
                std::cout << "setpluralops (slow) called with unsupported operator\n";
                exit(1);
                break;
        }
        return res;
    }

    int setopunioncount( const int cutoff, const formulaoperator fo ) override {
        if (fo != formulaoperator::founion) {
            std::cout << "No support for plural set bool ops other than for 'meet' and 'disjoint'; try pairwise (i)\n";
            return 0;
        }

        std::vector<setitr*> out {};
        if (sets.size() > 1)
            out.push_back( new setitrunion(sets[0], sets[1]) );
        else {
            std::cout << "Empty sets in slowpluralops\n";
            return 0;
        }
        int cnt = out[0]->getsize();
        for (int j = 2; j < sets.size() && cnt < cutoff; ++j) {
            out.push_back(new setitrunion(out[out.size()-1],sets[j]));
            cnt = out[out.size()-1]->getsize();
            if (cnt >= cutoff && cutoff != -1)
                break;
        }
        for (auto s : out)
            delete s;
        return cnt;
    }

    int setopintersectioncount( const int cutoff, const formulaoperator fo ) override {
        if (fo != formulaoperator::fointersection) {
            std::cout << "No support for plural set bool ops other than for 'cup' and 'cap'; try pairwise\n";
            return 0;
        }

        std::vector<setitr*> out {};
        if (sets.size() > 1)
            out.push_back( new setitrintersection(sets[0], sets[1]) );
        int cnt = out[0]->getsize();
        for (int j = 2; j < sets.size() && cnt >= cutoff; ++j) {
            out.push_back(new setitrintersection(out[out.size()-1],sets[j]));
            cnt = out[out.size()-1]->getsize();
            if (cnt < cutoff && cutoff != -1)
                break;
        }
        for (auto s : out)
            delete s;
        return cnt;
    }
    bool boolsetops( const formulaoperator fo ) override {
        if (sets.size() < 2) {
            std::cout << "Less than two sets passed to plural bool ops\n";
            return false;
        }
        if (fo != formulaoperator::fomeet && fo != formulaoperator::fodisjoint) {
            std::cout << "No support for plural set bool ops other than for 'meet' and 'disjoint'; try pairwise\n";
            return false;
        }
        if (fo == formulaoperator::fomeet)
            return setopintersectioncount( 1, formulaoperator::fointersection ) > 0 ? true : false;
        else
            return setopintersectioncount(1, formulaoperator::fointersection ) < 1 ? true : false;
    }
    setitrslowpluralops( std::vector<setitr*> setsin ) : sets{setsin} {}
};
template<typename T>
class setitrtuplefastops : public setitrabstractops {
public:
    setitrtuple<T>* castA;
    setitrtuple<T>* castB;

    setitr* setops( const formulaoperator fo )  override {
        auto out = fasttupleops<T>( castA, castB, fo );
        return out;
    }
    bool boolsetops( const formulaoperator fo ) override {
        return fastbooltupleops<T>( castA, castB, fo );
    }
    setitrtuplefastops( setitrtuple<T>* castAin, setitrtuple<T>* castBin ) : castA{castAin}, castB{castBin} {}
};
/*
class setitrtuplefastssops : public setitrabstractops {
public:
    setitrsubset* castAss;
    setitrsubset* castBss;

    setitr* setops( const formulaoperator fo )  override {
        auto itrint = fasttupleops( castAss->itrint, castBss->itrint, fo );
        auto out = new setitrsubset( castAss->superset, itrint );
        return out;
    }
    bool boolsetops( const formulaoperator fo ) override {
        return fastbooltupleops( castAss->itrint, castBss->itrint, fo );
    }
    setitrtuplefastssops( setitrsubset* castAssin, setitrsubset* castBssin ) : castAss{castAssin}, castBss{castBssin} {}
};*/
class setitrtupleslowops : public setitrabstractops {
public:
    setitr* setA;
    setitr* setB;
    setitr* setops( const formulaoperator fo )  override {
        switch (fo) {
            case formulaoperator::founion:
            case formulaoperator::fodupeunion:
                return new setitrtupleappend( setA, setB );
            case formulaoperator::fosetminus:
                return new setitrsetminus( setA, setB );
            case formulaoperator::fointersection:
                return new setitrintersection( setA, setB );
            default:
                std::cout << "tupleops (slow) called with non-set operator\n";
                return new setitrunion( setA, setB );
        }
    }
    bool boolsetops( const formulaoperator fo ) override {
        auto tmpitrA = setA->getitrpos();
        auto tmpitrB = setB->getitrpos();
        bool out;
        switch (fo) {
            case formulaoperator::folte:
                out = slowtupleinitialsegment( tmpitrA, tmpitrB );
                break;
            case formulaoperator::folt:
                out = slowtupleinitialsegment( tmpitrA, tmpitrB ) && tmpitrA->getsize() < tmpitrB->getsize();
                break;
            case formulaoperator::fogte:
                out = slowtupleinitialsegment( tmpitrB, tmpitrA );
                break;
            case formulaoperator::fogt:
                out = slowtupleinitialsegment( tmpitrB, tmpitrA ) && tmpitrB->getsize() < tmpitrA->getsize();
                break;
            case formulaoperator::foe:
                out = slowtupleinitialsegment( tmpitrA, tmpitrB ) && tmpitrB->getsize() == tmpitrA->getsize();
                break;
            case formulaoperator::fone:
                out = !slowtupleinitialsegment( tmpitrA, tmpitrB ) || tmpitrB->getsize() != tmpitrA->getsize();
                break;
            case formulaoperator::fodisjoint:
                out = !slowtuplemeet( tmpitrA, tmpitrB, 1 );
                break;
            case formulaoperator::fomeet:
                out = slowtuplemeet( tmpitrA, tmpitrB, 1 );
                break;

            default:
                std::cout << "tuple boolsetops (slow) called with non-set or non boolean operator\n";
                break;
        }
        delete tmpitrA;
        delete tmpitrB;
        return out;
    }
    setitrtupleslowops( setitr* setAin, setitr* setBin ) : setA{setAin}, setB{setBin} {}
};

inline setitrabstractops* getsetitrops( setitr* setA, setitr* setB ) {
    if (setitrint2dsymmetric* castA2d = dynamic_cast<setitrint2dsymmetric*>(setA))
        if (setitrint2dsymmetric* castB2d = dynamic_cast<setitrint2dsymmetric*>(setB))
            if (castA2d->dim1 == castB2d->dim1 && castA2d->dim2 == castB2d->dim2)
                return new setitrfast2dops( castA2d, castB2d );
    if (setitrint* castA = dynamic_cast<setitrint*>(setA))
        if (setitrint* castB = dynamic_cast<setitrint*>(setB))
            return new setitrfastops( castA, castB );
    if (setitrsubset* castAss = dynamic_cast<setitrsubset*>(setA))
        if (setitrsubset* castBss = dynamic_cast<setitrsubset*>(setB))
            if (castAss->superset->parent == castBss->superset->parent)
                return new setitrfastssops( castAss, castBss );
    return new setitrslowops( setA, setB );
}
inline setitrabstractops* getsetitrpluralops( std::vector<setitr*> sets ) {
    bool all = true;
    int i = 0;
    std::vector<setitrint2d*> casts2d {};
    while (all && i < sets.size()) {
        if (setitrint2dsymmetric* cast2d = dynamic_cast<setitrint2dsymmetric*>(sets[i]))
            casts2d.push_back(cast2d);
        else
            all = false;
        ++i;
    }
    if (all) {
        bool samedims = true;
        for (int i = 0; i+1 < casts2d.size() && samedims; ++i)
            samedims = samedims && casts2d[i]->dim1 == casts2d[i+1]->dim1 && casts2d[i]->dim2 == casts2d[i+1]->dim2;
        if (samedims)
            return new setitrfastplural2dops( casts2d );
    }
    all = true;
    i = 0;
    std::vector<setitrint*> casts {};
    while (all && i < sets.size()) {
        if (setitrint* cast = dynamic_cast<setitrint*>(sets[i]))
            casts.push_back(cast);
        else
            all = false;
        ++i;
    }
    if (all)
        return new setitrfastpluralops( casts );
    all = true;
    i = 0;
    std::vector<setitrsubset*> castsss {};
    while (all && i < sets.size()) {
        if (setitrsubset* castss = dynamic_cast<setitrsubset*>(sets[i]))
            castsss.push_back(castss);
        else
            all = false;
        ++i;
    }
    if (all) {
        bool sameparent = true;
        for (int i = 0; i+1 < castsss.size() && sameparent; ++i)
            sameparent = sameparent && (castsss[i]->superset->parent == castsss[i+1]->superset->parent);
        if (sameparent)
            return new setitrfastpluralssops( castsss );
    }
    return new setitrslowpluralops( sets );
}
inline setitrabstractops* gettupleops( setitr* setA, setitr* setB ) {

    if (setitrtuple<int>* castA = dynamic_cast<setitrtuple<int>*>(setA))
        if (setitrtuple<int>* castB = dynamic_cast<setitrtuple<int>*>(setB))
            return new setitrtuplefastops<int>( castA, castB );
    if (setitrtuple<bool>* castA = dynamic_cast<setitrtuple<bool>*>(setA))
        if (setitrtuple<bool>* castB = dynamic_cast<setitrtuple<bool>*>(setB))
            return new setitrtuplefastops<bool>( castA, castB );
    if (setitrtuple<double>* castA = dynamic_cast<setitrtuple<double>*>(setA))
        if (setitrtuple<double>* castB = dynamic_cast<setitrtuple<double>*>(setB))
            return new setitrtuplefastops<double>( castA, castB );

    /* if (setitrint* castA = dynamic_cast<setitrint*>(setA))
        if (setitrint* castB = dynamic_cast<setitrint*>(setB))
            return new setitrtuplefastops( castA, castB );
    if (setitrsubset* castAss = dynamic_cast<setitrsubset*>(setA))
        if (setitrsubset* castBss = dynamic_cast<setitrsubset*>(setB))
            if (castAss->superset->parent == castBss->superset->parent)
                return new setitrtuplefastssops( castAss, castBss ); */
    return new setitrtupleslowops( setA, setB );
}

class setitrset : public setitrmodeone
{
    public:
    setitrset( const std::vector<valms>& totalityin)
    {
        totality = totalityin;
        computed = true;
        pos = -1;
    }
    setitrset()
    {
        totality.clear();
        computed = true;
        pos = -1;
    }
};


class qclass {
public:
    std::string name;
    valms qs;
    formulaclass* superset;
    formulaclass* criterion;
    formulaclass* value;
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

/*
class setitrcp : public setitr
// cross-product
{
public:
    itrpos* setA;
    itrpos* setB;

    virtual int getsize() {return setA->getsize() * setB->getsize();}

    virtual void reset()
    {
        pos = -1;
        setA->reset();
        setB->reset();
    }
    virtual bool ended() {return setA->ended() && setB->ended();}
    virtual valms getnext()
    {
        valms r;
        r.t = mttuple;
        std::vector<valms> temp;
        temp.resize(2);
        if (setA->pos + 1 <= 0)
        {
            if (!setA->ended())
                temp[0] = setA->getnext();
            else
            {
                std::cout << "setitrcp error position past end of setA\n";
            }
            setB->reset();
        }
        if (!setB->ended())
        {
            temp.push_back(setB->getnext());
            r.seti = new setitrvalmstuple(temp);
        } else
        {
            if (!setA->ended())
            {
                temp[0] = setA->getnext();;
                setB->reset();
                if (!setB->ended())
                    temp[1] = setB->getnext();
                else
                {
                    std::cout << "setitrcp error position past end of setB\n";
                }
            }
        }
        if (++pos >= totality.size())
            totality.push_back(r);
        return totality[pos];
    }
    setitrcp(setitr* Ain, setitr* Bin) : setA{Ain->getitrpos()}, setB{Bin->getitrpos()} {}
    ~setitrcp()
    {
        delete setA;
        delete setB;
        for (auto t : totality)
            delete t.seti;
    }
};
*/

/*
class setitredges : public setitr
{
public:
    int posa = -1;
    int posb = -1;
    graphtype* g;
    int getsize() override
    {
        return edgecnt(g);
    }

    void reset() override
    {
        pos = -1;
        posa = -1;
        posb = -1;
    }

    bool ended() override {return posa >= g->dim-2 && posb >= g->dim-1; }
    valms getnext() override
    {
        if (++pos < totality.size())
            return totality[pos];
        valms r;
        r.t = mtset;
        int tmpposa = posa;
        int tmpposb = posb;
        while (tmpposa < g->dim-2 || tmpposb < g->dim-1)
        {
            if (tmpposa >= tmpposb-1)
            {
                while (tmpposa >= tmpposb-1)
                    ++tmpposb;
                tmpposa = 0;
            }
            else
                ++tmpposa;
            if (g->adjacencymatrix[tmpposa*g->dim + tmpposb])
            {
                posa = tmpposa;
                posb = tmpposb;
                r.seti = new setitrintpair(posa, posb);
                totality.resize(pos+1);
                totality[pos] = r;
                return r;
            }
        }
        r.seti = new setitrintpair(posa, posb);
        posa = tmpposa;
        posb = tmpposb;
        return r;
    }
    setitredges( graphtype* gin ) : g{gin} {}
};
*/


// Function to compute Stirling numbers of
// the second kind S(n, k) with memoization
inline int stirling(int n, int k) {

    // Base cases
    if (n == 0 && k == 0) return 1;
    if (k == 0 || n == 0) return 0;
    if (n == k) return 1;
    if (k == 1) return 1;


    // Recursive formula
    return k * stirling(n - 1, k) + stirling(n - 1, k - 1);
}

// Function to calculate the total number of
// ways to partition a set of `n` elements
inline int bellNumber(int n) {

    int result = 0;

    // Sum up Stirling numbers S(n, k) for all
    // k from 1 to n
    for (int k = 1; k <= n; ++k) {
        result += stirling(n, k);
    }
    return result;
}



inline int lookup_variable( const std::string& tok, const namedparams& context) {
    bool found = false;
    int i = context.size() - 1;
    for ( ; i >= 0 && !found; --i)
        found = context[i].first == tok;
    if (!found)
        return -1;
    return i+1;
}



struct formulavalue {
    valms v;
    litstruct lit;
    fnstruct fns;
    qclass* qc;
    variablestruct vs;
    setstruct ss;
    bool subgraph;
};

class formulaclass {
public:
    qclass* boundvariable {};
    formulavalue v;
    formulaclass* fcleft;
    formulaclass* fcright;
    formulaoperator fo;
    formulaclass(formulavalue vin, formulaclass* fcleftin, formulaclass* fcrightin, formulaoperator foin)
        : v{vin}, fcleft(fcleftin), fcright(fcrightin), fo(foin) {}
    ~formulaclass() {
        //delete fcleft;
        //delete fcright;
        //if (fo == formulaoperator::fofunction) {
        //    for (auto fnf : v.fns.ps)
        //        delete fnf;
        //}
        if (fo == formulaoperator::fovariable) {
            //if (v.v.t == mtset || v.v.t == mtpairset)
            //    delete v.v.seti;
        }
    }

};



formulaclass* fccombine( const formulavalue& item, formulaclass* fc1, formulaclass* fc2, formulaoperator fo );


inline std::map<formulaoperator,int> precedencemap {
                            {formulaoperator::foqexists,0},
                            {formulaoperator::foqforall,0},
                            {formulaoperator::foqsum,0},
                            {formulaoperator::foqproduct,0},
                            {formulaoperator::foqmin,0},
                            {formulaoperator::foqmax,0},
                            {formulaoperator::foqrange,0},
                            {formulaoperator::foqaverage,0},
                            {formulaoperator::foqtally,0},
                            {formulaoperator::foqcount,0},
                            {formulaoperator::foqset,0},
                            {formulaoperator::foqdupeset,0},
                            {formulaoperator::foqunion,0},
                            {formulaoperator::foqdupeunion,0},
                            {formulaoperator::foqintersection,0},
                            {formulaoperator::fonaming,0},
                            {formulaoperator::foexponent,1},
                            {formulaoperator::fotimes,2},
                            {formulaoperator::fodivide,2},
                            {formulaoperator::fomodulus,2},
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
                            {formulaoperator::founion,3},
                            {formulaoperator::fodupeunion,3},
                            {formulaoperator::fointersection,3},
                            {formulaoperator::foswitch,8},
                            {formulaoperator::focases, 7},
                            {formulaoperator::foin, 8},
                            {formulaoperator::foas, 8},
                            {formulaoperator::fosetminus, 3},
                            {formulaoperator::fosetxor, 3},
                            {formulaoperator::fomeet, 3},
                            {formulaoperator::fodisjoint, 3}};


bool is_operator( const std::string& tok );




formulaclass* parseformula(
    const std::string& sentence,
    const std::vector<int>& litnumps,
    const std::vector<measuretype>& littypes,
    const std::vector<std::string>& litnames,
    namedparams& ps,
    const std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>* fnptrs = &global_fnptrs  );


class evalformula
{
public:
    graphtype* g {};
    std::vector<valms> literals {};
    std::map<std::string,std::pair<double (*)(std::vector<double>&),int>>*fnptrs = &global_fnptrs;

    virtual valms evalpslit( const int idx, namedparams& context, neighborstype* subgraph, params& ps );
    virtual valms evalvariable( const variablestruct& v, const namedparams& context, const std::vector<int>& vidxin );
    virtual valms eval( const formulaclass& fc, const namedparams& context );
    evalformula();

    ~evalformula()
    {
    }

};

// outdated by map in math.cpp
// inline bool quantifierops( const formulaoperator fo );


inline bool searchfcforvariable( formulaclass* fc, std::vector<std::string> bound = {});

inline void mtconvertboolto( const bool vin, valms& vout )
{
    switch (vout.t)
    {
        case mtbool: vout.v.bv = vin;
            break;
        case mtdiscrete: vout.v.iv = vin ? 1 : 0;
            break;
        case mtcontinuous: vout.v.dv = vin ? 1 : 0;
            break;
        case mtset: vout.seti = vin ? new setitrint(1) : new setitrint(0);
            break;
        case mttuple: vout.seti = vin ? new setitrint(1) : new setitrint(0);
            break;
        case mtstring: vout.v.rv = new std::string(vin ? "true" : "false");
            break;
        case mtgraph: vout.v.nsv = new neighborstype(new graphtype(vin ? 1 : 0));
            break;
    }
}
inline void mtconvertdiscreteto( const int vin, valms& vout )
{
    switch (vout.t)
    {
    case mtbool: vout.v.bv = vin != 0 ? true : false;
        break;
    case mtdiscrete: vout.v.iv = vin;
        break;
    case mtcontinuous: vout.v.dv = vin != 0 ? 1 : 0;
        break;
    case mtset: vout.seti = new setitrint(vin);
        break;
    case mttuple: vout.seti = new setitrint(vin);
        break;
    case mtstring: vout.v.rv = new std::string(std::to_string(vin));
        break;
    case mtgraph: vout.v.nsv = new neighborstype(new graphtype(vin));
        break;
    }
}
inline void mtconvertcontinuousto( const double vin, valms& vout )
{
    switch (vout.t)
    {
    case mtbool: vout.v.bv = abs(vin) >= ABSCUTOFF ? true : false;
        break;
    case mtdiscrete: vout.v.iv = (int)vin;
        break;
    case mtcontinuous: vout.v.dv = vin;
        break;
    case mtset: vout.seti = new setitrint((int)vin);
        break;
    case mttuple: vout.seti = new setitrint((int)vin);
        break;
    case mtstring: vout.v.rv = new std::string(std::to_string(vin));
        break;
    case mtgraph: vout.v.nsv = new neighborstype(new graphtype((int)vin));
        break;
    }
}
inline void mtconvertsetto( setitr* vin, valms& vout )
{
    switch (vout.t)
    {
    case mtbool: vout.v.bv = vin->getsize() > 0 ? true : false;
        break;
    case mtdiscrete: vout.v.iv = vin->getsize();
        break;
    case mtcontinuous: vout.v.dv = vin->getsize();
        break;
    case mtset: vout.seti = vin;
        break;
    case mttuple: vout.seti = vin;
        break;
    case mtstring: vout.v.rv = new std::string("{ SET of size " + std::to_string(vin->getsize()) + "}");
        break;
    case mtgraph: vout.v.nsv = new neighborstype(new graphtype(vin->getsize()));
        break;
    }
}
inline void mtconverttupleto( setitr* vin, valms& vout )
{
    switch (vout.t)
    {
    case mtbool: vout.v.bv = vin->getsize() > 0 ? true : false;
        break;
    case mtdiscrete: vout.v.iv = vin->getsize();
        break;
    case mtcontinuous: vout.v.dv = vin->getsize();
        break;
    case mtset: vout.seti = vin;
        break;
    case mttuple: vout.seti = vin;
        break;
    case mtstring: vout.v.rv = new std::string("< TUPLE of size " + std::to_string(vin->getsize()) + ">");
        break;
    case mtgraph: vout.v.nsv = new neighborstype(new graphtype(vin->getsize()));
        break;
    }
}
inline void mtconvertstringto( std::string* vin, valms& vout )
{
    switch (vout.t)
    {
    case mtbool: vout.v.bv = stoi(*vin) != 0 ? true : false;
        break;
    case mtdiscrete: vout.v.iv = stoi(*vin);
        break;
    case mtcontinuous: vout.v.dv = stod(*vin);
        break;
    case mtset:
    case mttuple: {
        const std::regex r2 {"([[:alpha:]]\\w*)"};
        std::vector<valms> out2 {};
        for (std::sregex_iterator p2(vin->begin(),vin->end(),r2); p2!=std::sregex_iterator{}; ++p2)
        {
            valms v;
            v.t = mtstring;
            v.v.rv = new std::string;
            *v.v.rv = (*p2)[1];
            out2.push_back(v);
        }
        vout.seti = new setitrmodeone(out2);
        break;
    }
    case mtstring:
        {
            vout.v.rv = vin;
            break;
        }
    case mtgraph:
        {
            vout.v.nsv = new neighborstype(new graphtype(vin->size()));
            std::vector<std::string> temp {};
            temp.push_back(*vin);
            auto g = igraphstyle(temp);
            vout.v.nsv = new neighbors(g);
            break;
        }
    }
}
inline void mtconvertgraphto( neighborstype* vin, valms& vout )
{
    switch (vout.t)
    {
    case mtbool: vout.v.dv = vin->g->dim > 0 ? true : false;
        break;
    case mtdiscrete: vout.v.iv = vin->g->dim;
        break;
    case mtcontinuous: vout.v.dv = vin->g->dim;
        break;
    case mtset:
    case mttuple:
        {
            std::vector<valms> out2 {};
            int dim = vin->g->dim;
            if (vin->g->vertexlabels.size() == dim)
                for (int i = 0; i < dim; ++i)
                {
                    valms v;
                    v.t = mtstring;
                    v.v.rv = new std::string(vin->g->vertexlabels[i]);
                    out2.push_back(v);
                }
                else
                for (int i = 0; i < dim; ++i)
                {
                    valms v;
                    v.t = mtstring;
                    v.v.rv = new std::string(std::to_string(i));
                    out2.push_back(v);
                }
            vout.seti = new setitrmodeone(out2);
            break;
        }
    case mtstring: {
        std::stringstream ss {};
        osmachinereadablegraph(ss,vin->g);
        vout.v.rv = new std::string(ss.str());
        break;
    }
    case mtgraph: vout.v.nsv = vin;
        break;
    }
}
inline void mtconverttobool( const valms& vin, bool& vout )
{
    switch (vin.t)
    {
    case mtbool: vout = vin.v.bv; break;
    case mtdiscrete: vout = vin.v.iv != 0 ? true : false; break;
    case mtcontinuous: vout = abs(vin.v.dv) >= ABSCUTOFF ? true : false; break;
    case mtset:
    case mttuple: vout = vin.seti->getsize() > 0 ? true : false; break;
    case mtstring: vout = stoi(*vin.v.rv) != 0 ? true : false; break;
    case mtgraph: vout = vin.v.nsv->g->dim > 0 ? true : false; break;
    }
}
inline void mtconverttodiscrete( const valms& vin, int& vout )
{
    switch (vin.t)
    {
    case mtbool: vout = (int)vin.v.bv; break;
    case mtdiscrete: vout = vin.v.iv; break;
    case mtcontinuous: vout = (int)vin.v.dv; break;
    case mtset:
    case mttuple: vout = vin.seti->getsize(); break;
    case mtstring: vout = stoi(*vin.v.rv); break;
    case mtgraph: vout = vin.v.nsv->g->dim; break;
    }
}
inline void mtconverttocontinuous( const valms& vin, double& vout )
{
    switch (vin.t)
    {
    case mtbool: vout = vin.v.bv ? 1 : 0; break;
    case mtdiscrete: vout = vin.v.iv; break;
    case mtcontinuous: vout = vin.v.dv; break;
    case mtset:
    case mttuple: vout = vin.seti->getsize(); break;
    case mtstring: vout = stod(*vin.v.rv); break;
    case mtgraph: vout = vin.v.nsv->g->dim; break;
    }
}
inline void mtconverttoset( const valms& vin, setitr*& vout )
{
    switch (vin.t)
    {
    case mtbool: {vout = vin.v.bv ? new setitrint(1) : new setitrint(0); break;}
    case mtdiscrete: {vout = new setitrint(vin.v.iv); break; // verify works with any negative not just -1
            }
    case mtcontinuous: {vout = new setitrint((int)vin.v.dv); break;}
    case mtset:
    case mttuple: vout = vin.seti; break;
    case mtstring:
        {
            const std::regex r2 {"([[:alpha:]]\\w*)"};
            std::vector<valms> out2 {};
            for (std::sregex_iterator p2(vin.v.rv->begin(),vin.v.rv->end(),r2); p2!=std::sregex_iterator{}; ++p2)
            {
                valms v {};
                v.t = mtstring;
                v.v.rv = new std::string((*p2)[1]);
                out2.push_back(v);
            }
            vout = new setitrmodeone(out2);
            break;
        }
    case mtgraph:
        {
            vout = new setitrint(vin.v.nsv->g->dim); break;
        }
    }
}
inline void mtconverttotuple( const valms& vin, setitr*& vout )
{
    switch (vin.t)
    {
    case mtbool: {vout = vin.v.bv ? new setitrint(1) : new setitrint(0); break;}
    case mtdiscrete: {vout = new setitrint(vin.v.iv); break;} // verify works with any negative not just -1
    case mtcontinuous: {vout = new setitrint((int)vin.v.dv); break;}
    case mtset:
    case mttuple: {vout = vin.seti; break;}
    case mtstring:
        {
            const std::regex r2 {"([[:alpha:]]\\w*)"};
            std::vector<valms> out2 {};
            for (std::sregex_iterator p2(vin.v.rv->begin(),vin.v.rv->end(),r2); p2!=std::sregex_iterator{}; ++p2)
            {
                valms v {};
                v.t = mtstring;
                v.v.rv = new std::string((*p2)[1]);
                out2.push_back(v);
            }
            vout = new setitrmodeone(out2);
            break;
        }
    case mtgraph:
        {
            vout = new setitrint(vin.v.nsv->g->dim); break;
        }
    }
}
inline void mtconverttostring( const valms& vin, std::string*& vout )
{
    switch (vin.t)
    {
    case mtbool: *vout = vin.v.bv ? "true" : "false" ; break;
    case mtdiscrete: *vout = std::to_string(vin.v.iv); break;
    case mtcontinuous: *vout = std::to_string(vin.v.dv); break;
    case mtset:
    case mttuple: *vout = "{ SET of size " + std::to_string(vin.seti->getsize()) + "}";
    case mtstring: *vout = "< TUPLE of size " + std::to_string(vin.seti->getsize()) + ">";
    case mtgraph:
        std::stringstream ss {};
        osmachinereadablegraph(ss,vin.v.nsv->g);
        *vout = ss.str();
        break;
    }
}
inline void mtconverttograph( const valms& vin, neighborstype*& vout )
{
    switch (vin.t)
    {
    case mtbool: {vout = new neighborstype(new graphtype(vin.v.bv ? 1 : 0)); break;}
    case mtdiscrete: {vout = new neighborstype(new graphtype(vin.v.iv)); break;}
    case mtcontinuous: {vout = new neighborstype(new graphtype((int)vin.v.dv)); break;}
    case mtset:
    case mttuple:
        {
            int dim = vin.seti->getsize();
            vout = new neighborstype(new graphtype(dim));
            auto pos = vin.seti->getitrpos();
            int i = 0;
            vout->g->vertexlabels.resize(dim);
            while (!pos->ended()) {
                std::string* temp = &vout->g->vertexlabels[i++];
                mtconverttostring(pos->getnext(), temp );
            }
            break;
        }
    case mtstring:
        {
            vout = new neighborstype(new graphtype(vin.v.rv->size()));
            std::vector<std::string> temp {};
            temp.push_back(*vin.v.rv);
            auto g = igraphstyle(temp);
            vout = new neighbors(g);
            break;
        }
    case mtgraph: vout = vin.v.nsv;
        break;
    }
}
inline void mtconverttype1( const valms& vin, valms& vout )
{
    switch (vin.t)
    {
        case measuretype::mtbool: mtconvertboolto(vin.v.bv,vout); break;
        case measuretype::mtdiscrete: mtconvertdiscreteto(vin.v.iv,vout); break;
        case measuretype::mtcontinuous: mtconvertcontinuousto(vin.v.dv,vout); break;
        case measuretype::mtset: mtconvertsetto(vin.seti,vout); break;
        case measuretype::mttuple: mtconverttupleto(vin.seti,vout); break;
        case measuretype::mtstring: mtconvertstringto(vin.v.rv,vout); break;
        case measuretype::mtgraph: mtconvertgraphto(vin.v.nsv,vout); break;
    }

}
inline void mtconverttype2( const valms& vin, valms& vout )
{
    switch (vout.t)
    {
    case measuretype::mtbool: mtconverttobool(vin,vout.v.bv); break;
    case measuretype::mtdiscrete: mtconverttodiscrete(vin,vout.v.iv); break;
    case measuretype::mtcontinuous: mtconverttocontinuous(vin,vout.v.dv); break;
    case measuretype::mtset: mtconverttoset( vin, vout.seti); break;
    case measuretype::mttuple: mtconverttotuple(vin,vout.seti); break;
    case measuretype::mtstring: mtconverttostring(vin,vout.v.rv); break;
    case measuretype::mtgraph: mtconverttograph(vin,vout.v.nsv); break;
    }
}
inline bool graphsequal( graphtype* g1, graphtype* g2 ) {
// obviously a question of using heavier machinery, as in using graph isomorphisms here...
// also note the code assumes the adjacency matrices are properly formed
    int dim = g1->dim;
    if (dim != g2->dim) return false;
    for (int i = 0; i+1 < dim; ++i)
        for (int j = i+1; j < dim; ++j)
            if (g1->adjacencymatrix[i*dim + j] != g2->adjacencymatrix[i*dim + j])
                return false;
    return true;
}
inline bool mtareequal( const valms& v, const valms& w ) { // initially overloaded the == ops, but this seems more kosher
    if (v.t != w.t)
        return false;
    switch (v.t) {
        case mtbool:
        case mtdiscrete:
        case mtcontinuous:
            return v == w;
        case mtset: {
            auto abstractsetops = getsetitrops( v.seti, w.seti );
            bool res = abstractsetops->boolsetops( formulaoperator::foe );
            delete abstractsetops;
            return res; }
        case mttuple: {
            auto abstracttupleops = gettupleops( v.seti, w.seti );
            bool res = abstracttupleops->boolsetops( formulaoperator::foe );
            delete abstracttupleops;
            return res; }
        case mtstring:
            return *v.v.rv == *w.v.rv;
        case mtgraph:
            return graphsequal( v.v.nsv->g, w.v.nsv->g );
        default: std::cout << "Unsupported type " << v.t << " in mtareequal\n";
        return false;
    }
}

inline bool mtareequalgenerous( const valms& v, const valms& w ) { // initially overloaded the == ops, but this seems more kosher
    // if (v.t != w.t)
    //    return false;
    switch (v.t) {
    case mtbool:
    case mtdiscrete:
    case mtcontinuous:
        return v == w;
    case mtset: {
            setitr* temp;
            mtconverttoset(w,temp);
            auto abstractsetops = getsetitrops( v.seti, temp );
            bool res = abstractsetops->boolsetops( formulaoperator::foe );
            delete abstractsetops;
            return res; }
    case mttuple: {
            setitr* temp;
            mtconverttotuple(w,temp);
            auto abstracttupleops = gettupleops( v.seti, temp );
            bool res = abstracttupleops->boolsetops( formulaoperator::foe );
            delete abstracttupleops;
            return res; }
    case mtstring:
        if (w.t == mtstring)
            return *v.v.rv == *w.v.rv;
        break;
    case mtgraph:
        if (w.t == mtgraph)
            return graphsequal( v.v.nsv->g, w.v.nsv->g );
        break;
    }
    std::cout << "Unsupported type " << v.t << " and " << w.t << " in mtareequalgenerous\n";
    return false;
}




#endif //MATH_H
