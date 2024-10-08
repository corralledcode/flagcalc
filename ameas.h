//
// Created by peterglenn on 7/15/24.
//

#ifndef AMEAS_H
#define AMEAS_H

#define KNMAXCLIQUESIZE 12

#include <cstring>
#include <functional>
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
    struct Sres
    {
//        int* i; // params i
        bool* b; // computed
        T* r; // data
    };

    int* maxplookup;
    int blocksize = 100;
    params** plookup;
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

        this->sz = szin;
        int s = pmsv->size();
        plookup = (params**)malloc(s*szin*sizeof(params*));
        pres = (Sres*)malloc(s*blocksize*sizeof(Sres));
        maxplookup = (int*)malloc(s*szin*sizeof(int));
        memset(maxplookup,0,s*szin*sizeof(int));
        for (int i = 0; i < s; ++i)
        {
            for (int j = 0; j < szin; ++j) {
                plookup[i*szin+j] = new params[blocksize]; //(params*)malloc(blocksize*sizeof(params));

            }
            if ((*pmsv)[i]->pssz > 0)
            {
                for (int k = 0; k < blocksize; ++k)
                {
                    pres[i*blocksize+k].b = (bool*)malloc(szin*sizeof(bool));
                    pres[i*blocksize+k].r = (T*)malloc(szin*sizeof(T));
                    memset(pres[i*blocksize+k].b,false,szin*sizeof(bool));
                }
            } else
            {
                pres[i].b = (bool*)malloc(szin*sizeof(bool));
                pres[i].r = (T*)malloc(szin*sizeof(T));
                memset(pres[i].b,false,szin*sizeof(bool));
            }
        }
    }


    void setblocksize( const int iidx, const int newblocksize )
    {

        // add ability to copy existing data to the new blocks...

    }

    virtual T fetch(const int idx, const int iidx )
    {
        params ps;
        ps.clear();
        return this->fetch(idx,iidx,ps);
    }
    virtual T fetch( const int idx, const int iidx, const params& ps)
    {

        int i = 0;
        bool found = false;
        const int s = pmsv->size();
        const int szin = this->sz;
        if (ps.size() > 0)
            for (i=1; !found && (i <= maxplookup[iidx * szin + idx]); ++i)
                found = found || (plookup[iidx*szin + idx][i] == ps);
        else
        {
            i = 0;
            found = true;
        }

        if (i > 0)
        {
            if (!found)
            {
                i = (maxplookup[iidx * szin + idx])++;
                if (i >= blocksize)
                {
                    std::cout << "Increase block size\n";
                    return 0;
                }
                plookup[iidx*szin + idx]->resize(i+1);
                plookup[iidx*szin + idx][i] = ps;
                pres[iidx*blocksize + i].r[idx] = (*this->pmsv)[iidx]->takemeas(idx,ps);
                pres[iidx*blocksize + i].b[idx] = true;
            }
            if (found)
            {
                if (!pres[iidx*blocksize + i].b[idx])
                {
                    pres[iidx*blocksize + i].r[idx] = (*this->pmsv)[iidx]->takemeas(idx,ps);
                    pres[iidx*blocksize + i].b[idx] = true;
                }

            }
            return pres[iidx*blocksize + i].r[idx];

        } else
        {
            if (!pres[iidx].b[idx])
            {
                pres[iidx].r[idx] = (*this->pmsv)[iidx]->takemeas(idx,ps);
                pres[iidx].b[idx] = true;
            }
            return pres[iidx].r[idx];

        }
    }

    precords() : records<T>(), pmsv{new std::vector<pameas<T>*>} {}


    ~precords()
    {

        // delete [] plookup;
        // for (int s = 0; s < pmsv->size(); ++s)
        // {
            // for (int j = 0; j < blocksize; ++j) {
                // delete pres[s*blocksize + j].b;
                // delete pres[s*blocksize + j].r;
            // }
            // for (int m = 0; m < this->sz; ++m)
            // {
                // delete plookup[s*this->sz + m];
            // }
        // }
        // delete pres;
        // delete plookup;



        plookup = {};
        pres = {};
        //maxplookup = 0;
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

    // T fetch( const int idx, const int iidx ) override
    // {
        // return precords<T>::fetch(idx,iidx);
    // }


    // T fetch( const int idx, const int iidx, const params& ps) override
    // {
        // return precords<T>::fetch(idx,iidx,ps);
    // }

    virtual void threadfetch( const int startidx, const int stopidx, const int iidx, const params& ps)
    {
        for (int i = startidx; i < stopidx; ++i)
        {
            this->fetch( i, iidx, ps);
        }
    }

    virtual void threadfetchpartial( const int startidx, const int stopidx, const int iidx, const params& ps, std::vector<bool>* todo)
    {
        for (int i = startidx; i < stopidx; ++i)
            if ((*todo)[i])
                this->fetch( i, iidx, ps);
    }
};

inline void populatevariables(std::vector<qclass*>* variables ) {
    auto Vv = new qclass;
    auto Ev = new qclass;
    auto NEv = new qclass;

    // Vv->qs.v.iset = (bool*)malloc(g->dim * sizeof(bool));
    // memset(Vv->qs.v.iset,true,g->dim * sizeof(bool));
    // Vv->qs.setsize = g->dim;
    Vv->name = "V";
    Vv->qs.t = mtset;
    Vv->qs.v.iset = nullptr;
    // Ev->qs.v.iset = (bool*)malloc(g->dim*g->dim * sizeof(bool));
    // for (int i = 0; i < g->dim*g->dim; ++i)
        // Ev->qs.v.iset[i] = g->adjacencymatrix[i];
    // Ev->qs.setsize = g->dim*g->dim;
    Ev->name = "E";
    Ev->qs.t = mtset;
    Ev->qs.v.iset = nullptr;
    // NEv->qs.v.iset = (bool*)malloc(g->dim * g->dim * sizeof(bool));
    // for (int i = 0; i < g->dim*g->dim; ++i)
        // NEv->qs.v.iset[i] = !g->adjacencymatrix[i];
    // for (int i = 0; i < g->dim; ++i)
        // NEv->qs.v.iset[i*g->dim + i] = false;
    NEv->name = "NE";
    // NEv->qs.setsize = g->dim*g->dim;
    NEv->qs.t = mtset;
    NEv->qs.v.iset = nullptr;
    variables->push_back(Vv);
    variables->push_back(Ev);
    variables->push_back(NEv);
}


class evalmformula : public evalformula
{
public:

    int idx;
    //std::vector<qclass*> variables {};
    mrecords* rec;

    valms evalpslit( const int l, params& psin ) override;
    valms evalvariable(std::string& vname) override;

    evalmformula( mrecords* recin );

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
    std::vector<qclass*> variables {};


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
        // for (int i = 0; i < sz; ++i)
            // delete variablesv[i];
    }
};

inline evalmformula::evalmformula( mrecords* recin ) : evalformula(), rec{recin}
{

    // populatevariables(&variables);
}


inline valms evalmformula::evalpslit( const int l, params& psin )
{
    ams a = rec->lookup(l);

    params tmpps {};

    switch (a.t)
    {
    case mtbool:
        tmpps = a.a.cs->ps;
        break;
    case mtdiscrete:
        tmpps = a.a.ts->ps;
        break;
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
                /*
                case measuretype::mtpair:
                    switch (psin[i].t)
                    {
                    case measuretype::mtbool: psin[i].v.ip.i = 1;
                        psin[i].v.ip.j = 1;
                        break;
                    case measuretype::mtcontinuous: psin[i].v.ip.i = (int)psin[i].v.dv;
                        psin[i].v.ip.j = (int)psin[i].v.dv;
                        break;
                    case mtdiscrete: psin[i].v.ip.i = psin[i].v.iv;
                        psin[i].v.ip.j = psin[i].v.iv;
                        break;
                    case mtpair:
                        break;
                    }*/
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

inline valms evalmformula::evalvariable(std::string& vname)
{
    valms res;
    graphtype* g = (*rec->gptrs)[idx];

    int i = (lookup_variable(vname,variables));
    if (i >= 0)
        return evalformula::evalvariable(vname);
    if (vname == "V") {
        res.setsize = g->dim;
        res.t = mtset;
        res.v.iset = (bool*)malloc(g->dim*sizeof(bool));
        memset(res.v.iset,true,g->dim*sizeof(bool));
        //rec->variables[i]->qs = res;
        return res;
    }
    if (vname == "E") {
        res.setsize = g->dim; //g->dim*(g->dim-1)/2;
        res.t = mtpairset;
        res.v.iset = (bool*)malloc((g->dim*(g->dim-1)/2)*sizeof(bool));
        int gapidx = g->dim-1;
        int idx = gapidx;
        for (int i = 0; i < g->dim; ++i)
        {
            for (int j = i+1; j < g->dim; ++j)
                res.v.iset[idx + i - j] = g->adjacencymatrix[i*g->dim + j];
            --gapidx;
            idx += gapidx;
        }
        //rec->variables[i]->qs = res;
        return res;
    }
    if (vname == "NE")
    {
        res.setsize = g->dim; //g->dim*(g->dim-1)/2;
        res.t = mtpairset;
        res.v.iset = (bool*)malloc((g->dim*(g->dim-1)/2)*sizeof(bool));
        int gapidx = g->dim-1;
        int idx = gapidx;
        for (int i = 0; i < g->dim; ++i)
        {
            for (int j = i+1; j < g->dim; ++j)
                res.v.iset[idx - j] = !g->adjacencymatrix[i*g->dim + j];
            --gapidx;
            idx += gapidx;
        }
        //rec->variables[i]->qs = res;
        return res;
    }
}


class sentofcrit : public crit
{
public:
    formulaclass* fc;
    std::vector<qclass*> variables;

    bool takemeas(const int idx) override
    {
        std::vector<valms> literals;
        literals.resize(rec->literals.size());
        for (int i = 0; i < rec->literals.size(); ++i)
            literals[i] = rec->literals[i][idx];
        rec->efv[idx]->literals = &literals;
        evalmformula* ef = rec->efv[idx];
        ef->variables.resize(this->variables.size());
        for (int i = 0; i < variables.size(); ++i)
        {
            ef->variables[i] = new qclass;
            ef->variables[i]->name = this->variables[i]->name;
            ef->variables[i]->qs = this->variables[i]->qs;
            ef->variables[i]->superset = this->variables[i]->superset;
        }
        ef->idx = idx;
        // auto pv = std::function<void()>(std::bind(populatevariables,(*rec->gptrs)[idx],&ef->variables));
        // ef->populatevariablesbound = &pv;
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


    sentofcrit( mrecords* recin , const std::vector<int>& litnumpsin,
                const std::vector<measuretype>& littypesin, const std::string& fstr )
        : crit( recin,  "sn", "Sentence " + fstr) {
        fc = parseformula(fstr,litnumpsin,littypesin,variables,&global_fnptrs);
    };

    ~sentofcrit() {
        delete fc;
        // for (int i = 0; i < rec->sz; ++i)
        // {
            // for (int j = 0; j < variables.size(); ++j)
                // delete rec->variablesv[i][j];
            // delete variables[i];
        // }
    }

};



class formmeas : public meas
{
public:
    formulaclass* fc;
    std::vector<qclass*> variables {};

    double takemeas(const int idx) override
    {
        std::vector<valms> literals;
        literals.resize(rec->literals.size());
        for (int i = 0; i < rec->literals.size(); ++i)
            literals[i] = rec->literals[i][idx];
        rec->efv[idx]->literals = &literals;
        evalmformula* ef = rec->efv[idx];
        ef->variables.resize(this->variables.size());
        for (int i = 0; i < variables.size(); ++i)
        {
            ef->variables[i] = new qclass;
            ef->variables[i]->name = this->variables[i]->name;
            ef->variables[i]->qs = this->variables[i]->qs;
            ef->variables[i]->superset = this->variables[i]->superset;
        }
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
        : meas( recin,  "fm", "Formula " + fstr) {
        fc = parseformula(fstr,litnumpsin,littypesin,variables,&global_fnptrs);
    }

    ~formmeas() {
        delete fc;
    }


};






#endif //AMEAS_H
