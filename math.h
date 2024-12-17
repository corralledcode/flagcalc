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
    foswitch, focases, foin};

inline std::map<std::string,formulaoperator> operatorsmap
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
        {"IN", formulaoperator::foin}};


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
//    ~valms()
//    {
//        if (t == mtgraph)
//        {
//            delete v.nsv->g;
//            delete v.nsv;
//        } else
//        if (t == mtstring)
//            delete v.rv;
//    };
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
    // case measuretype::mtpair:
        // return a1.v.p->i == a2.v.p->i && a1.v.p->j == a2.v.p->j;
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

class setitrunion : public setitrmodeone
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
            bool found = false;
            valms v = Bitr->getnext();
            for (int i = 0; !found && i < temp.size(); i++)
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

class setitrbool : public setitrmodeone
{
public:
    int maxint;
    bool* elts = nullptr;

    void compute() override
    {
        std::vector<valms> temp {};
        // totality.clear();
        temp.resize(maxint+1);
        int j = 0;
        for (int i = 0; i < maxint+1; ++i)
        {
            if (elts[i])
            {
                valms v;
                v.t = measuretype::mtdiscrete;
                v.v.iv = i;
                v.seti = nullptr;
                temp[j++] = v;
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
        maxint = maxintin;
        if (maxint >= 0)
        {
            elts = (bool*)malloc((maxint+1)*sizeof(bool));
            memset(elts, true, (maxint+1)*sizeof(bool));
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
    };
    setitrbool()
    {
        setmaxint(-1);
        t = mtdiscrete;
    };

    ~setitrbool() {
        delete elts;
    }
};


class setitrsubset : public setitr {
public:
    setitrbool* itrbool;
    itrpos* superset {};
    void setsuperset( itrpos* supersetposin )
    {
        superset = supersetposin;
        itrbool->setmaxint(superset->getsize() - 1);
        reset();
    }

    int getsize() override
    {
        return itrbool->getsize();
    }

    valms getnext() override
    {
        if (pos+1 < totality.size())
            return totality[++pos];

        if (!itrbool->computed)
            itrbool->compute();
        valms res;
        int superpos = -1;
        if (!itrbool->ended())
            superpos = itrbool->totality[++pos].v.iv;
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
        itrbool->reset();
        pos = -1;
    }
    bool ended() override
    {
        if (itrbool->computed)
            return pos + 1 >= itrbool->totality.size();
        else
            return pos + 1 >= itrbool->getsize();
    }
    setitrsubset(itrpos* supersetin) : superset{supersetin}, itrbool{new setitrbool(supersetin->getsize() - 1)}
    {
        t = superset->parent->t;
        pos = -1;
    };
    setitrsubset() : superset{}, itrbool{new setitrbool(-1)}
    {
        t = mtdiscrete;
        pos = -1;
    };

};

class setitrint : public setitrmodeone
{
    public:
    int maxint = -1;
    bool* elts = nullptr;
    void compute() override
    {
        std::vector<valms> temp {};
        temp.clear();
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
        totality.clear();
        reset();
    }
    setitrint(const int maxintin)
    {
        setmaxint(maxintin);
        t = mtdiscrete;
        reset();
    }
    setitrint()
    {
        setmaxint(-1);
        t = mtdiscrete;
        reset();
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
        pos = -1;
    }
    setitrset()
    {
        totality.clear();
        computed = true;
        pos = -1;
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
    setitrintpair(int intain, int intbin) : inta{intain}, intb{intbin} {}
};

class setitrtuple : public setitrmodeone
{
protected:
    int size;
public:
    void compute() override
    {
        computed = true;
    }
    setitrtuple(std::vector<valms> totalityin)
    {
        totality = totalityin;
        computed = true;
    }
};

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
            r.seti = new setitrtuple(temp);
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

class setitrsizedsubset : public setitr
{
public:
    itrpos* setA;

    std::vector<int> posAprimes {};
    int size = 2;

    int getsize() override
    {
        // if (setA->parent == this)
        // {
            // std::cout << "Circular reference in setitrsizedsubset::getsize()\n";
            // return 0;
        // }
        int s = setA->getsize();
        //double res1 = tgamma(s+1);
        //double res2 = tgamma(s+1-size)*tgamma(size+1); // commented out due to floating point error and simply too large
        return nchoosek(s,size);
    }

    void reset() override
    {
        pos = -1;
        setA->reset();
        if (size > 0)
        {
            posAprimes.resize(size-1);
            for (int i = 0; i+1 < size; ++i)
                posAprimes[i] = i;
        } else
        {
            posAprimes.resize(0);
        }
    }
    bool ended() override
    {
        bool res;
        if (size == 0)
            res = pos+1 >= 1;
        else
        {
            res = setA->ended();
            int s = setA->getsize();
            for (int i = 0; i+1 < size; ++i)
                res = res && (posAprimes[i] == s - size + i);
        }
        return res;
    }
    valms getnext() override
    {
        if (++pos < totality.size())
            return totality[pos];
        valms r;
        r.t = mtset;
        auto subset = new setitrsubset(setA);
        r.seti = subset;
        bool inced = false;
        while (!setA->ended() && setA->pos+1 < size)
        {
            inced = true;
            setA->getnext();
        }
        for (int i = 0; !inced && i+1 < size; ++i)
        {
            if (i+2 == size)
            {
                if (posAprimes[i] + 1 == setA->pos)
                    continue;
            } else
                if (posAprimes[i] + 1 == posAprimes[i+1])
                    continue;
            ++posAprimes[i];
            for (int j = 0; j < i; ++j)
                posAprimes[j] = j;
            inced = true;
        }

        if (!inced)
        {
            setA->getnext();
            for (int i = 0; i < size-1; ++i)
            {
                posAprimes[i] = i;
            }
        }
        memset(subset->itrbool->elts, false, (subset->itrbool->maxint+1)*sizeof(bool));
        if (size > 0)
            subset->itrbool->elts[setA->pos] = true;
        for (int i = 0; i < size-1; ++i)
            subset->itrbool->elts[posAprimes[i]] = true;
        // r.setsize = size;
        totality.resize(pos+1);
        totality[pos] = r;
        // std::cout << "pos " << pos << ": ";
        // for (int i = 0; i < size-1; ++i)
            // std::cout << posAprimes[i] << ", ";
        // std::cout << setA->pos << std::endl;
        return r;
    }
    setitrsizedsubset(setitr* Ain, int sizein ) : setA{Ain ? Ain->getitrpos() : nullptr}, size{sizein}
    {
        t = mtset;
        if (Ain == this)
            std::cout << "Circular reference in setitrsizedsubset(); expect segfault\n";
        if (setA)
            reset();
    }
    ~setitrsizedsubset() override
    {
        delete setA;
    //   for (auto t : totality) // this is handled by ~setitr() above
    //       delete t.seti;
    //   setitr::~setitr();
    }
};


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


class setitrsetpartitions : public setitr
{
public:
    itrpos* setA;
    int setsize;

    bool endedvar;
    std::vector<int> sequence {};
    std::vector<setitrsubset*> subsets {};

    void codesubsets() {
        subsets.resize(setsize);

        int max = 0;
        for (int i = 0; i < setsize; ++i)
        {
            subsets[i] = new setitrsubset(setA);
            subsets[i]->itrbool = new setitrbool(setsize-1);
            for (int j = 0; j < setsize; ++j)
            {
                subsets[i]->itrbool->elts[j] = sequence[j] == i;
                max = max < sequence[j] ? sequence[j] : max;
            }
            subsets[i]->reset();
        }
        subsets.resize(max+1);

    }

    int getsize() override
    {
        setsize = setA->getsize();
        return bellNumber(setsize);
    }

    void reset() override
    {
        pos = -1;
        setsize = setA->getsize();
        setA->reset();
        sequence.resize(setsize);
        for (int i = 0; i < setsize; ++i)
            sequence[i] = 0;
        endedvar = setsize == 0;
        totality.clear();
    }
    bool ended() override
    {
        return endedvar;
    }
    valms getnext() override
    {
        if (pos+1 < totality.size())
            return totality[++pos];
        ++pos;
        if (!endedvar)
        {
            codesubsets();
            std::vector<valms> tot {};
            for (auto s : subsets)
            {
                valms v;
                v.t = mtset;
                v.seti = s;
                tot.push_back(v);
            }
            totality.resize(pos+1);
            valms u;
            u.t = mtset;
            u.seti = new setitrmodeone(tot);
            totality[pos] = u;

            int j = setsize;
            bool incrementable = false;
            endedvar = false;
            while (j > 1 && !incrementable)
            {
                --j;
                int i = j-1;
                while (!incrementable && i >= 0)
                    incrementable = sequence[j] <= sequence[i--];
            }
            if (incrementable)
            {
                sequence[j]++;
                for (int k = j+1; k < setsize; ++k)
                    sequence[k] = 0;

            } else
                endedvar = true;
            return totality[pos];
        }
        std::cout << "setitrsetpartitions: ended\n";
        valms v;
        v.t = mtset;
        v.seti = new setitrint(-1);
        return v;
    }

    setitrsetpartitions(setitr* Ain ) : setA{Ain ? Ain->getitrpos() : nullptr}
    {
        if (Ain == this)
            std::cout << "Circular reference in setitrsizedsubset(); expect segfault\n";
        if (setA)
            reset();
    }

    ~setitrsetpartitions() override
    {

        delete setA;
        for (auto s : subsets)
        {
            delete s->itrbool;
            delete s;
        }
    // for (auto t : totality) // this is handled by ~setitr() above
        // delete t.seti;
    //   setitr::~setitr();
    }
};




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
                            {formulaoperator::foin, 8}};


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

    virtual valms evalpslit( const int idx, namedparams& context, neighborstype* subgraph, std::vector<valms>& psin );
    virtual valms evalvariable( variablestruct& v, namedparams& context, std::vector<int>& vidxin );
    virtual valms eval( formulaclass& fc, namedparams& context );
    evalformula();

    ~evalformula()
    {
    }

};

inline bool quantifierops( const formulaoperator fo );


inline bool searchfcforvariable( formulaclass* fc, std::vector<std::string> bound = {})
{
    if (!fc)
        return false;
    if (quantifierops(fc->fo))
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





#endif //MATH_H
