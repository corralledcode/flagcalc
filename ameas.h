//
// Created by peterglenn on 7/15/24.
//

#ifndef AMEAS_H
#define AMEAS_H

#define KNMAXCLIQUESIZE 12

#include <cstring>
#include <stdexcept>
#include "math.h"
#include "graphs.h"

template<typename T>
class records;

template<typename T>
class precords;

class mrecords; // multiple types of precords

class crit;
class meas;
class tally;

template<typename T>
class ameas
{
protected:
    mrecords* rec;
public:
    const std::string shortname;
    const std::string name;
    virtual T takemeas(const int idx)
    {
        return {};
    }
    ameas( mrecords* recin , const std::string shortnamein, const std::string namein)
        : rec{recin}, shortname{shortnamein}, name{namein} {};

};

union amptrs {
    crit* cs;
    meas* ms;
    tally* ts;
};

struct ams
{
    measuretype t;
    amptrs a;
};

using params = std::vector<valms>;

struct itn
{
    int round;
    measuretype t;
    int iidx;
    params ps;
    bool hidden;
};


inline bool operator==(const ams& a1, const ams& a2)
{
    if (a1.t != a2.t)
        return false;

    switch (a1.t) {
        case measuretype::mtcontinuous:
            return a1.a.ms == a2.a.ms;
        case measuretype::mtdiscrete:
            return a1.a.ts == a2.a.ts;
        case measuretype::mtbool:
            return a1.a.cs == a2.a.cs;
    }
}




struct iters
{
    ams a;
    int round;
    bool hidden;
};



template<typename T>
class pameas : public ameas<T>
{
public:
    params ps {};
    int pssz = 0;
    virtual std::string getname()
    {
        return this->name;
    }
    T takemeas(const int idx) override
    {
        return {};
    }
    virtual T takemeas(const int idx, const params& ps )
    {
        return takemeas(idx);
    }

    pameas( mrecords* recin , std::string shortnamein, std::string namein)
        : ameas<T>(recin, shortnamein,namein)
    {}
};


class crit : public pameas<bool>
{
public:
    bool negated = false;
    std::string getname() override
    {
        return (negated ? "NOT " : "") + name;
    }
    crit( mrecords* recin , const std::string shortnamein, const std::string name)
        : pameas<bool>(recin,  shortnamein, name) {}
};

class meas : public pameas<double>
{
public:
    meas( mrecords* recin , const std::string shortnamein, const std::string name)
        : pameas<double>(recin,  shortnamein, name) {}
};

class tally : public pameas<int>
{
public:
    tally( mrecords* recin , const std::string shortnamein, const std::string name)
        : pameas<int>(recin,  shortnamein, name) {}
};

template<typename T>
class records
{
public:
    int sz = 0;
    std::vector<T*> res {};
    std::vector<bool*> computed{};
    std::vector<ameas<T>*>* msv {};

    virtual int findliteral( const std::string& sin )
    {
        for (auto i = 0; i < msv->size(); ++i )
            if ((*msv)[i]->shortname == sin)
                return i;
        return -1;
    }
    virtual int findms( const ameas<T>* msin)
    {
        for (auto i = 0; i < msv->size(); ++i)
            if ((*msv)[i] == msin)
                return i;
        return -1;
    }


    virtual void setsize(const int szin)
    {
        for (int i = 0; i < res.size(); ++i)
            delete res[i];
        for (int i = 0; i < computed.size(); ++i)
            delete computed[i];
        res.resize( msv->size());
        computed.resize(msv->size());
        sz = szin;
        for (int i = 0; i < msv->size(); ++i)
        {
            res[i] = (T*)malloc(sz *sizeof(T));
            computed[i] = (bool*)malloc(sz*sizeof(bool));
            for (int j = 0; j < sz; ++j)
                computed[i][j] = false;
        }
    }

    virtual T fetch( const int idx, const int iidx )
    {
        if (idx >= sz)
            throw std::out_of_range("Error in records: out of range");
        if (!computed[iidx][idx])
        {
            res[iidx][idx] = (*msv)[iidx]->takemeas(idx);
            computed[iidx][idx] = true;
        }
        return res[iidx][idx];
    }

    records() : msv{new std::vector<ameas<T>*>} {}

    ~records()
    {
        for (int i = 0; i < res.size(); ++i)
            delete res[i];
        for (int i = 0; i < computed.size(); ++i)
            delete computed[i];
        // res.clear();
        // computed.clear();
        // for (auto m : *msv)
            // delete m;
        // delete msv;
        // msv = nullptr;
    }

};

template<typename T>
class precords : public records<T>
{
public:
    struct Slookup
    {
        params* ps;
        int* i;
    };
    struct Sres
    {
        int* i;
        bool* b;
        T* r;
    };

    int maxplookup = 0;
    int blocksize = 25;
    Slookup* plookup = nullptr;
    Sres* pres = nullptr;


    // std::vector<std::map<params,int>> plookup {};
    // std::vector<std::map<int,T*>> pres {};
    // std::vector<std::map<int,bool*>> pcomputed {};
    std::vector<pameas<T>*>* pmsv {};

    int findliteral( const std::string& sin ) override
    {
        for (auto i = 0; i < pmsv->size(); ++i )
            if ((*pmsv)[i]->shortname == sin)
                return i;
        return -1;
    }
    virtual int findms( const pameas<T>* pamin)
    {
        for (auto i = 0; i < pmsv->size(); ++i)
            if ((*pmsv)[i] == pamin)
                return i;
        return -1;
    }

    virtual void setsize(const int szin)
    {

        plookup = {};
        pres = {};
        maxplookup = 0;

        this->sz = szin;

        if (plookup == nullptr)
           setblocksize(blocksize);
    }


    void setblocksize( const int newblocksize )
    {

        // add ability to copy existing data to the new blocks...

        plookup = (Slookup*)malloc(pmsv->size()*sizeof(Slookup));
        pres = (Sres*)malloc(pmsv->size()*sizeof(Sres));
        for (int i = 0; i < pmsv->size(); ++i)
        {
            plookup[i].i = (int*)malloc(blocksize*sizeof(int));
            plookup[i].ps = new params[blocksize]; //(params*)malloc(blocksize*sizeof(params));
            memset(plookup[i].i,0,blocksize*sizeof(int));
            pres[i].b = (bool*)malloc(this->sz*blocksize*sizeof(bool));
            pres[i].r = (T*)malloc(this->sz*blocksize*sizeof(T));
            pres[i].i = (int*)malloc(blocksize*sizeof(int));
            memset(pres[i].b,false,this->sz*blocksize*sizeof(bool));
            memset(pres[i].i, 0, blocksize*sizeof(int));
        }
    }

    void addparams( const int iidx, const int m)
    {
        maxplookup = m;
        int mult = 2;
        if (m > blocksize)
        {
            while (blocksize*mult++ < m)
                ;
            setblocksize(blocksize*mult);
        }
    }

    virtual T fetch(const int idx, const int iidx )
    {
        params ps;
        ps.clear();
        return this->fetch(idx,iidx,ps);
    }
    virtual T fetch( const int idx, const int iidx, const params& ps)
    {
        int i;
        bool found = false;
        for (i=1; !found && (i <= maxplookup); ++i)
            found = found || (plookup[iidx].ps[i] == ps);

        if (!found)
        {
            i = ++maxplookup;
            addparams(iidx,i);
            plookup[iidx].ps[i] = ps;
            plookup[iidx].i[i] = i;
            // for (auto p : ps)
            // {
                // valms pcpy;
                // pcpy.t = p.t;
                // pcpy.v = p.v;
                // plookup[iidx].ps[i].push_back(pcpy);
            // }
        }
        found = false;
        int j;
        for( j = 0; !found && (j <= i); ++j)
            found = found || (pres[iidx].i[j] == i);
        if (!found || (found && !pres[iidx].b[j*this->sz + idx])) {
            --j;
            pres[iidx].i[j] = i;
            pres[iidx].r[j*this->sz + idx] = (*this->pmsv)[iidx]->takemeas(idx,ps);
            pres[iidx].b[j*this->sz + idx] = true;
        }
        return pres[iidx].r[j*this->sz + idx];
    }

    precords() : records<T>(), pmsv{new std::vector<pameas<T>*>} {}


    ~precords()
    {
        // delete [] plookup;
        // for (int i = 0; i < pmsv->size(); ++i)
        // {
            // delete pres[i].b;
            // delete pres[i].r;
            // delete pres[i].i;
        // }
        // delete pres;

        plookup = {};
        pres = {};
        maxplookup = 0;
        // for (auto m : *pmsv)
            // delete m;
        // delete pmsv;
        // msv = nullptr;
    }

};

template<typename T>
class thrrecords : public precords<T>
{
public:

    T fetch( const int idx, const int iidx ) override
    {
        return precords<T>::fetch(idx,iidx);
    }


    T fetch( const int idx, const int iidx, const params& ps) override
    {
        return precords<T>::fetch(idx,iidx,ps);
    }

    virtual void threadfetch( const int startidx, const int stopidx, const int iidx, const params& ps)
    {
        for (int i = startidx; i < stopidx; ++i)
            this->fetch( i, iidx, ps);
    }

    virtual void threadfetchpartial( const int startidx, const int stopidx, const int iidx, const params& ps, std::vector<bool>* todo)
    {
        for (int i = startidx; i < stopidx; ++i)
            if ((*todo)[i])
                this->fetch( i, iidx, ps);
    }
};


class evalmformula : public evalformula
{
public:

    int idx;
    mrecords* rec;

    valms evalpslit( const int l, params& psin ) override;

    evalmformula( mrecords* recin ) : evalformula(), rec{recin} {}

};


class mrecords
{
public:
    int sz = 0;
    int msz = 0;
    std::vector<graphtype*>* gptrs;
    std::vector<neighborstype*>* nsptrs;
    thrrecords<bool> boolrecs;
    thrrecords<int> intrecs;
    thrrecords<double> doublerecs;
    std::map<int,std::pair<measuretype,int>> m;
    std::vector<evalmformula*> efv {};
    std::vector<valms*> literals {};

    void addliteralvalueb( const int iidx, const int idx, bool v )
    {
        if (iidx >= msz)
            setmsize(iidx + 1);
        literals[iidx][idx].t = mtbool;
        literals[iidx][idx].v.bv = v;
    }
    void addliteralvaluei( const int iidx, const int idx, int v )
    {
        if (iidx >= msz)
            setmsize(iidx+1);
        literals[iidx][idx].t = mtdiscrete;
        literals[iidx][idx].v.iv = v;
    }
    void addliteralvalued( const int iidx, const int idx, double v )
    {
        if (iidx >= msz)
            setmsize(iidx+1);
        literals[iidx][idx].t = mtcontinuous;
        literals[iidx][idx].v.dv = v;
    }



    int maxm()
    {
        int mx = -1;
        for (auto p : m)
            mx = p.first > mx ? p.first : mx;
        return mx;
    }
    void addm( const int i, measuretype t, int j )
    {
        m[i] = {t,j};
    }

    ams lookup(const int i)
    {
        ams res;
        res.t = m[i].first;
        switch (res.t)
        {
        case measuretype::mtbool:
            res.a.cs = (crit*)(*boolrecs.pmsv)[m[i].second];
            return res;
        case measuretype::mtcontinuous:
            res.a.ms = (meas*)(*doublerecs.pmsv)[m[i].second];
            return res;
        case measuretype::mtdiscrete:
            res.a.ts = (tally*)(*intrecs.pmsv)[m[i].second];
            return res;
        }
    }

    int intlookup(const int i)
    {
        return m[i].second;
    }

    void setsize(const int szin)
    {
        sz = szin;
        boolrecs.setsize(sz);
        intrecs.setsize(sz);
        doublerecs.setsize(sz);
        efv.resize(sz);
        for (int i = 0; i < sz; ++i)
        {
            efv[i] = new evalmformula(this);

        }
    }

    void setmsize( const int mszin )
    {
        int oldsz = literals.size();
        if (oldsz < mszin)
        {
            literals.resize(mszin);
            for (int i = oldsz; i < mszin; ++i )
                literals[i] = (valms*)malloc(sz * sizeof(valms));
        }
        msz = mszin;
    }

    ~mrecords()
    {
        for (int i = 0; i < sz; ++i)
            delete efv[i];
        for (int i = 0; i < msz; ++i)
            delete literals[i];
    }
};


inline valms evalmformula::evalpslit( const int l, params& psin )
{
    ams a = rec->lookup(l);

    params tmpps {};

    switch (a.t)
    {
    case mtbool:
        tmpps = a.a.cs->ps;
    case mtdiscrete:
        tmpps = a.a.ts->ps;
    case mtcontinuous:
        tmpps = a.a.ms->ps;
    }

    for (int i = 0; i < tmpps.size(); ++i)
            switch(tmpps[i].t)
            {
                case measuretype::mtbool:
                    switch (psin[i].t)
                    {
                        case measuretype::mtbool: break;
                        case measuretype::mtcontinuous: psin[i].v.bv = !(abs(psin[i].v.dv) < 0.0000001);
                            psin[i].t = mtbool;
                            break;
                        case mtdiscrete: psin[i].v.bv = (bool)psin[i].v.iv;
                            psin[i].t = mtbool;
                            break;
                    }
                   break;
                case measuretype::mtdiscrete:
                    switch (psin[i].t)
                    {
                        case measuretype::mtbool: psin[i].v.iv = (int)psin[i].v.bv;
                            psin[i].t = mtdiscrete;
                            break;
                        case measuretype::mtcontinuous: psin[i].v.iv = std::lround(psin[i].v.dv);
                            psin[i].t = mtdiscrete;
                            break;
                        case mtdiscrete: break;
                    }
                    break;
                case measuretype::mtcontinuous:
                    switch (psin[i].t)
                    {
                        case measuretype::mtbool: psin[i].v.dv = (double)psin[i].v.bv;
                            psin[i].t = mtcontinuous;
                            break;
                        case measuretype::mtcontinuous: break;
                        case mtdiscrete: psin[i].v.dv = (double)psin[i].v.iv;
                            psin[i].t = mtcontinuous;
                            break;
                    }
                    break;
            }

    valms r;
    r.t = a.t;
    switch (r.t)
    {
    case measuretype::mtbool: r.v.bv = a.a.cs->takemeas(idx,psin);
        return r;
    case measuretype::mtdiscrete: r.v.iv = a.a.ts->takemeas(idx,psin);
        return r;
    case measuretype::mtcontinuous: r.v.dv = a.a.ms->takemeas(idx,psin);
        return r;
    }

}


class sentofcrit : public crit
{
public:
    const formulaclass* fc;

    bool takemeas(const int idx) override
    {
        std::vector<valms> literals;
        literals.resize(rec->literals.size());
        for (int i = 0; i < rec->literals.size(); ++i)
            literals[i] = rec->literals[i][idx];
        rec->efv[idx]->literals = &literals;
        evalmformula* ef = rec->efv[idx];
        ef->idx = idx;
        valms r = ef->eval(*fc);
        switch (r.t)
        {
        case measuretype::mtbool: return negated != r.v.bv;
        case measuretype::mtdiscrete: return negated != (bool)r.v.iv;
        case measuretype::mtcontinuous: return negated != (bool)r.v.dv;
        }
    }

    bool takemeas(const int idx, const params& ps) override
    {
        return takemeas(idx);
    }


    sentofcrit( mrecords* recin , const std::vector<int>& litnumpsin, const std::vector<measuretype>& littypesin, const std::string& fstr )
        : crit( recin,  "sn", "Sentence " + fstr),
            fc{parseformula(fstr,litnumpsin,littypesin,&global_fnptrs)}
    {};

};

class formmeas : public meas
{
public:
    const formulaclass* fc;

    double takemeas(const int idx) override
    {
        std::vector<valms> literals;
        literals.resize(rec->literals.size());
        for (int i = 0; i < rec->literals.size(); ++i)
            literals[i] = rec->literals[i][idx];
        rec->efv[idx]->literals = &literals;
        evalmformula* ef = rec->efv[idx];
        ef->idx = idx;
        valms r = ef->eval(*fc);
        switch (r.t)
        {
        case measuretype::mtbool: return (double)r.v.bv;
        case measuretype::mtdiscrete: return (double)r.v.iv;
        case measuretype::mtcontinuous: return r.v.dv;
        }
    }

    double takemeas(const int idx, const params& ps) override
    {
        return takemeas(idx);
    }


    formmeas( mrecords* recin , const std::vector<int>& litnumpsin, const std::vector<measuretype>& littypesin, const std::string& fstr )
        : meas( recin,  "fm", "Formula " + fstr),
            fc{parseformula(fstr,litnumpsin,littypesin,&global_fnptrs)}
    {};

};






#endif //AMEAS_H
