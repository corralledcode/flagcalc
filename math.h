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
{foliteral,fofunction, foconstant, foqforall, foqexists,
    foplus, fominus, fotimes, fodivide, foexponent, fomodulus,
    folte, folt, foe,fone,fogte,fogt,founion, fodupeunion, fointersection, foelt,
    foand,foor,foxor,fonot,foimplies,foiff,foif,fotrue,fofalse,fovariable,
    foqsum, foqproduct, foqmin, foqmax, foqaverage, foqrange,
    foqtally, foqcount, foqset, foqdupeset, foqunion, foqdupeunion, foqintersection};

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
        {"BIGCAP", formulaoperator::foqintersection}};


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

enum measuretype { mtbool, mtdiscrete, mtcontinuous, mtset, mttuple };

class setitr;
class itrpos;

bool setsubseteq( itrpos* in1, itrpos* in2);
bool tupleeq( itrpos* in1, itrpos* in2);



/*
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
*/

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
    // std::vector<itrpos*> itrs {};
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
        return {};
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
//            computed = true;
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
            //computed = true;
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
    }
    setitrint()
    {
        setmaxint(-1);
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

class setitrpairs : public setitr
{
    public:
    itrpos* setA;

    int posAprime = -1;

    int getsize() override
    {
        int s = setA->getsize();
        return s*(s-1)/2;
    }

    void reset() override
    {
        pos = -1;
        setA->reset();
        posAprime = -1;
    }
    bool ended() override {return setA->ended() && posAprime == setA->getsize() - 2;}
    valms getnext() override
    {
        if (++pos < totality.size())
            return totality[pos];
        valms r;
        r.t = mtset;
        auto subset = new setitrsubset(setA);
        r.seti = subset;
        while (!setA->ended() && setA->pos < 1)
            setA->getnext();
        if (posAprime < setA->pos-1)
            ++posAprime;
        else
        {
            setA->getnext();
            posAprime = 0;
        }
        subset->superset = setA;
        memset(subset->itrbool->elts, false, (subset->itrbool->maxint+1)*sizeof(bool));
        subset->itrbool->elts[setA->pos] = true;
        subset->itrbool->elts[posAprime] = true;
        r.setsize = 2;
        totality.resize(pos+1);
        totality[pos] = r;
        // std::cout << posAprime << ", " << setA->pos << std::endl;
        return totality[pos];
    }
    setitrpairs(setitr* Ain) : setA{Ain->getitrpos()} {}
    ~setitrpairs() override
    {
        delete setA;
        for (auto t : totality)
            delete t.seti;
        //setitr::~setitr();
    }
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
            for (int i = 0; i < size-1; ++i)
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
        while (!setA->ended() && setA->pos < size-1)
        {
            inced = true;
            setA->getnext();
        }
        for (int i = 0; !inced && i < size-1; ++i)
        {
            if (i == size-2)
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
        r.setsize = size;
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
                            {formulaoperator::founion,7},
                            {formulaoperator::fodupeunion,7},
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
        //    delete q;
        }
    }

};



#endif //MATH_H
