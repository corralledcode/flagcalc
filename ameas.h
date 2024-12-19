//
// Created by peterglenn on 7/15/24.
//

#ifndef AMEAS_H
#define AMEAS_H

#define KNMAXCLIQUESIZE 12


#include <cstring>
#include <functional>
#include <stdexcept>

#include "graphio.h"
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
class set;
class strmeas;
class gmeas;

template<typename T>
class ameas
{
protected:
    mrecords* rec {};
public:
    const std::string shortname {};
    const std::string name {};
    virtual T takemeas(const int idx)
    {
        T res;
        return res;
    }
    ameas( mrecords* recin , const std::string shortnamein, const std::string namein)
        : rec{recin}, shortname{shortnamein}, name{namein} {};

};

union amptrs {
    crit* cs;
    meas* ms;
    tally* ts;
    set* ss;
    set* os; // ordered set aka tuple
    strmeas* rs;
    gmeas* gs;
};

struct ams
{
    measuretype t;
    amptrs a;
};


struct itn
{
    int round;
    measuretype t;
    int iidx;
    namedparams nps;
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
        case measuretype::mtset:
            return false;
            // return a1.a.ss == a2.a.ss;
        case measuretype::mtstring:
            return a1.a.rs == a2.a.rs;
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
    namedparams nps {};
    int npssz = 0;
    std::vector<int> npreferences {};

    void bindnamedparams()
    {
        npreferences.resize(this->nps.size());
        int i = 0;
        for (auto np : nps)
        {
            npreferences[i] = i;
            ++i;
        }
        npssz = nps.size();
    }
    virtual std::string getname()
    {
        return this->name;
    }
    T takemeas(const int idx) override
    {
        return {};
    }
    virtual T takemeas( neighborstype* ns, const params& ps )
    {
        T out;
        return out;
    }
    virtual T takemeas(const int idx, const params& ps )
    {
        if (ps.size() != npssz)
        {
            std::cout << "Incorrect number of parameters (" << ps.size() << ") passed to " << this->shortname << ", expecting " << this->nps.size() << "." << std::endl;
            exit(1);
        }
        for (int i = 0; i < ps.size(); ++i)
        {
            nps[npreferences[i]].second = ps[i];
        }
//        T out;
//        return out;
//        neighborstype* ns = (*this->rec->nsptrs)[idx];

        return this->takemeas(idx);
    }

    pameas( mrecords* recin , std::string shortnamein, std::string namein)
        : ameas<T>(recin, shortnamein,namein)
    {}
};


class meas : public pameas<double>
{
public:
    meas( mrecords* recin , const std::string shortnamein, const std::string name)
        : pameas<double>(recin,  shortnamein, name) {}
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

class tally : public pameas<int>
{
public:
    tally( mrecords* recin , const std::string shortnamein, const std::string name)
        : pameas<int>(recin,  shortnamein, name) {}
};

class set : public pameas<setitr*>
{
public:
    set( mrecords* recin , const std::string shortnamein, const std::string namein)
        : pameas<setitr*>( recin, shortnamein, namein ) {}
    set() : pameas<setitr*>( nullptr, "_abst", "_abstract (error)" ) {};
};

class strmeas : public pameas<std::string*>
{
public:
    strmeas( mrecords* recin , const std::string shortnamein, const std::string namein)
        : pameas<std::string*>( recin, shortnamein, namein ) {}
    strmeas() : pameas<std::string*>( nullptr, "_abst", "_abstract (error)" ) {};
};

class gmeas : public pameas<neighborstype*>
{
public:
    gmeas( mrecords* recin , const std::string shortnamein, const std::string name)
        : pameas<neighborstype*>(recin,  shortnamein, name) {}
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
        // std::cout << "Unknown literal " << sin << std::endl;
        // exit(-1);
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
        res.clear();
        computed.clear();
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
        std::cout << "Unknown literal " << sin << std::endl;
        exit(-1);
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
            if ((*pmsv)[i]->npssz > 0)
            {
                for (int k = 0; k < blocksize; ++k)
                {
                    pres[i*blocksize+k].b = (bool*)malloc(szin*sizeof(bool));
                    pres[i*blocksize+k].r = (T*)malloc(szin*sizeof(T));
                    memset(pres[i*blocksize+k].b,false,szin*sizeof(bool));
                }
            } else
            {
                pres[i*blocksize].b = (bool*)malloc(szin*sizeof(bool));
                pres[i*blocksize].r = (T*)malloc(szin*sizeof(T));
                memset(pres[i*blocksize].b,false,szin*sizeof(bool));
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
                    T res;
                    return res;
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
            if (!pres[iidx*blocksize].b[idx])
            {
                pres[iidx*blocksize].r[idx] = (*this->pmsv)[iidx]->takemeas(idx,ps);
                pres[iidx*blocksize].b[idx] = true;
            }
            return pres[iidx*blocksize].r[idx];

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



class evalmformula : public evalformula
{
public:

    int idx = -1;
    neighborstype* ns {};
    mrecords* rec;

    valms evalpslit( const int l, namedparams& nps, neighborstype* subgraph, params& psin ) override;
    valms eval( formulaclass& fc, namedparams& context ) override;
    valms evalinternal( formulaclass& fc, namedparams& context );
    evalmformula( mrecords* recin, const int idxin );
    evalmformula( mrecords* recin, neighborstype* nsin );
    ~evalmformula() {}
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
    thrrecords<setitr*> setrecs;
    thrrecords<setitr*> tuplerecs;
    thrrecords<std::string*> stringrecs;
    thrrecords<neighborstype*> graphrecs;
    std::map<int,std::pair<measuretype,int>> m;
    std::vector<evalmformula*> efv {};
    std::vector<valms*> literals {};
    // std::vector<evalmformula*> npsefv {};

    // void addef( evalmformula* efin )
    // {
        // npsefv.push_back( efin );
    // }

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
    void addliteralvalues( const int iidx, const int idx, setitr* v )
    {
        if (iidx >= msz)
            setmsize(iidx + 1);
        literals[iidx][idx].t = mtset;
        literals[iidx][idx].seti = v;
    }
    void addliteralvaluet( const int iidx, const int idx, setitr* v )
    {
        if (iidx >= msz)
            setmsize(iidx + 1);
        literals[iidx][idx].t = mttuple;
        literals[iidx][idx].seti = v;
    }
    void addliteralvaluer( const int iidx, const int idx, std::string* v )
    {
        if (iidx >= msz)
            setmsize(iidx + 1);
        literals[iidx][idx].t = mtstring;
        literals[iidx][idx].v.rv = v;
    }
    void addliteralvalueg( const int iidx, const int idx, neighborstype* v )
    {
        if (iidx >= msz)
            setmsize(iidx + 1);
        literals[iidx][idx].t = mtgraph;
        literals[iidx][idx].v.nsv = v;
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
        case measuretype::mtset:
            res.a.ss = (set*)(*setrecs.pmsv)[m[i].second];
            return res;
        case measuretype::mttuple:
            res.a.os = (set*)(*tuplerecs.pmsv)[m[i].second];
            return res;
        case measuretype::mtstring:
            res.a.rs = (strmeas*)(*stringrecs.pmsv)[m[i].second];
            return res;
        case measuretype::mtgraph:
            res.a.gs = (gmeas*)(*graphrecs.pmsv)[m[i].second];
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
        setrecs.setsize(sz);
        tuplerecs.setsize(sz);
        stringrecs.setsize(sz);
        graphrecs.setsize(sz);
        efv.resize(sz);
        for (int i = 0; i < sz; ++i)
        {
            efv[i] = new evalmformula(this, i);
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

inline evalmformula::evalmformula( mrecords* recin, const int idxin ) : evalformula(), rec{recin}, idx{idxin} {}
inline evalmformula::evalmformula( mrecords* recin, neighborstype* nsin ) : evalformula(), rec{recin}, ns{nsin} {}

inline valms evalmformula::evalpslit( const int l, namedparams& context, neighborstype* subgraph, params& psin )
{
    ams a = rec->lookup(l);

    namedparams tmpps {};

    switch (a.t)
    {
    case mtbool:
        tmpps = a.a.cs->nps;
        break;
    case mtdiscrete:
        tmpps = a.a.ts->nps;
        break;
    case mtcontinuous:
        tmpps = a.a.ms->nps;
        break;
    case mtset:
        tmpps = a.a.ss->nps;
        break;
    case mttuple:
        tmpps = a.a.os->nps;
        break;
    case mtgraph:
        tmpps = a.a.gs->nps;
        break;
    case mtstring:
        tmpps = a.a.rs->nps;
        break;
    }

    for (int i = 0; i < tmpps.size(); ++i)
    {
        valms tempps = psin[i];
        tempps.t = tmpps[i].second.t;
        mtconverttype1(psin[i],tempps);
        psin[i] = tempps;
    }

/*
        switch(tmpps[i].second.t)
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
            default:
            psin[i].t = tmpps[i].second.t;
            psin[i].v.bv = false;
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
    default:
        psin[i].t = tmpps[i].second.t;
            psin[i].v.iv = 0;
            break;
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
    default:
        psin[i].t = tmpps[i].second.t;
            psin[i].v.dv = 0;
            break;
        }
            break;
    case measuretype::mtset:
        if (psin[i].t != mtset && psin[i].t != mttuple)
        {
            std::cout << "Set or tuple type required as parameter number " << i << "\n";
            exit(1);
        }
            break;
    case measuretype::mttuple:
        if (psin[i].t != mttuple && psin[i].t != mtset)
        {
            std::cout << "Tuple or set type required as parameter number " << i << "\n";
            exit(1);
        }
            break;
        }
*/
    valms r;
    r.t = a.t;
    if (subgraph)
    {
        switch (r.t)
        {
        case measuretype::mtbool: r.v.bv = a.a.cs->takemeas(subgraph,psin);
            return r;
        case measuretype::mtdiscrete: r.v.iv = a.a.ts->takemeas(subgraph,psin);
            return r;
        case measuretype::mtcontinuous: r.v.dv = a.a.ms->takemeas(subgraph,psin);
            return r;
        case measuretype::mtset: r.seti = a.a.ss->takemeas(subgraph,psin);
            return r;
        case measuretype::mttuple: r.seti = a.a.os->takemeas(subgraph,psin);
            return r;
        case measuretype::mtstring: r.v.rv = a.a.rs->takemeas(subgraph,psin);
            return r;
        case measuretype::mtgraph: r.v.nsv = a.a.gs->takemeas(subgraph,psin);
            return r;
        }

    }

    switch (r.t)
    {
    case measuretype::mtbool: r.v.bv = a.a.cs->takemeas(idx,psin);
        return r;
    case measuretype::mtdiscrete: r.v.iv = a.a.ts->takemeas(idx,psin);
        return r;
    case measuretype::mtcontinuous: r.v.dv = a.a.ms->takemeas(idx,psin);
        return r;
    case measuretype::mtset: r.seti = a.a.ss->takemeas(idx,psin);
        return r;
    case measuretype::mttuple: r.seti = a.a.os->takemeas(idx,psin);
        return r;
    case measuretype::mtstring: r.v.rv = a.a.rs->takemeas(idx,psin);
        return r;
    case measuretype::mtgraph: r.v.nsv = a.a.gs->takemeas(idx,psin);
        return r;
    }

}

inline bool istuplezero( itrpos* tuplein )
{
    bool iszero = true;
    while (iszero && !tuplein->ended())
    {
        auto v = tuplein->getnext();
        switch (v.t)
        {
        case measuretype::mtbool: iszero = iszero && !v.v.bv; break;
        case measuretype::mtdiscrete: iszero = iszero && v.v.iv == 0; break;
        case measuretype::mtcontinuous: iszero = iszero && abs(v.v.dv) <= ABSCUTOFF; break;
        case measuretype::mtset: iszero = iszero && v.seti->getsize() == 0; break;
        case measuretype::mttuple:
            {
                auto itr = v.seti->getitrpos();
                iszero = iszero && istuplezero(itr);
                delete itr;
                break;
            }
        case measuretype::mtstring:
            {
                return v.v.rv->size() > 0;
            }
        case measuretype::mtgraph:
            {
                return v.v.nsv->g->dim > 0;
            }
        }
    }
    return iszero;

}


class setitrformulae : public setitr
{
public:

    int size;
    evalmformula* ef;
    mrecords* rec;
    std::vector<formulaclass*> formulae;
    namedparams nps;

    virtual int getsize() {return formulae.size();}

    virtual bool ended() {return pos+1 >= getsize();}
    virtual valms getnext()
    {
        ++pos;
        if (pos >= getsize())
        {
            valms v;
            v.t = mtdiscrete;
            v.v.iv = 0;
            return v;
        }
        if (pos >= totality.size())
        {
            totality.resize(pos+1);
            totality[pos] = ef->eval(*formulae[pos],nps);
        }
        return totality[pos];
    }

    setitrformulae( mrecords* recin, const int idxin, std::vector<formulaclass*>& fcin, namedparams& npsin )
        : formulae{fcin}, rec{recin}, nps{npsin}
    {
        ef = new evalmformula(rec,idxin);

        totality.clear();
        reset();
        // to do... search the fcin for unbound variables, and compute now if any are found
        // int n = 0;
        // bool variablefound = false;
        // while (n < fcin.size() && !variablefound)
            // variablefound = searchfcforvariable(fcin[n++]);
        // if (variablefound)
            while (!ended())
                getnext();
    }

    ~setitrformulae()
    {
        delete ef;
    }
};




class sentofcrit : public crit
{
public:
    formulaclass* fc;

    bool takemeas( neighborstype* ns, const params& psin ) override
    {
        evalmformula* ef = new evalmformula(rec,ns);
        namedparams npslocal = nps;
        const valms r = ef->eval(*fc, npslocal);
        delete ef;
        bool out;
        mtconverttobool(r,out);
        return out;
    }

    bool takemeas(const int idx) override {
        evalmformula* ef = new evalmformula(rec,idx);
        namedparams npslocal = nps;
        const valms r = ef->eval(*fc, npslocal);
        delete ef;
        bool out;
        mtconverttobool(r,out);
        return out;

    }

    sentofcrit( mrecords* recin , const std::vector<int>& litnumpsin,
                const std::vector<measuretype>& littypesin, const std::vector<std::string>& litnamesin,
                const namedparams& npsin, const std::string& fstr, const std::string shortnamein = "" )
        : crit( recin,  shortnamein == "" ? "sn" : shortnamein, "Sentence " + fstr )
    {
        nps = npsin;
        bindnamedparams();
        fc = parseformula(fstr,litnumpsin,littypesin,litnamesin,nps, &global_fnptrs);
    };

    ~sentofcrit() {
        delete fc;
    }

};


class formmeas : public meas
{
public:
    formulaclass* fc;

    double takemeas( neighborstype* ns, const params& psin ) override
    {
        evalmformula* ef = new evalmformula(rec,ns);
        namedparams npslocal = nps;
        const valms r = ef->eval(*fc, npslocal);
        delete ef;
        bool out;
        mtconverttobool(r,out);
        return out;
    }

    double takemeas(const int idx) override
    {
        evalmformula* ef = new evalmformula(rec, idx);
        namedparams npslocal = nps;

        valms r = ef->eval(*fc, npslocal);
        delete ef;
        double out;
        mtconverttocontinuous(r,out);
        return out;
    }

    formmeas( mrecords* recin , const std::vector<int>& litnumpsin, const std::vector<measuretype>& littypesin,
        const std::vector<std::string>& litnamesin,
        const namedparams& npsin, const std::string& fstr, const std::string shortnamein = "" )
        : meas( recin,  shortnamein == "" ? "fm" : shortnamein, "Formula " + fstr )
    {
        nps = npsin;
        bindnamedparams();
        fc = parseformula(fstr,litnumpsin,littypesin,litnamesin,nps, &global_fnptrs);
    }
    ~formmeas() {
        delete fc;
    }
};


class formtally : public tally
{
public:
    formulaclass* fc;
    int takemeas(neighborstype* ns, const params& ps) override
    {
        evalmformula* ef = new evalmformula(rec,ns);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc, npslocal);
        delete ef;
        int out;
        mtconverttodiscrete(r,out);
        return out;
    }

    int takemeas(const int idx) override
    {
        evalmformula* ef = new evalmformula(rec,idx);
        namedparams npslocal = nps;

        valms r = ef->eval(*fc, npslocal);
        delete ef;
        int out;
        mtconverttodiscrete(r,out);
        return out;
    }

    formtally( mrecords* recin , const std::vector<int>& litnumpsin,
        const std::vector<measuretype>& littypesin, const std::vector<std::string>& litnamesin,
        const namedparams& npsin, const std::string& fstr, const std::string shortnamein = "" )
            : tally( recin,  shortnamein == "" ? "ft" : shortnamein, "Int-valued formula " + fstr ) {
        nps = npsin;
        bindnamedparams();
        fc = parseformula(fstr,litnumpsin,littypesin,litnamesin,nps,&global_fnptrs);
    };

    ~formtally() {
        delete fc;
    }

};


class formset : public set
{
public:
    formulaclass* fc;
    setitr* takemeas( neighborstype* ns, const params& psin ) override
    {
        evalmformula* ef = new evalmformula(rec,ns);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc, npslocal);
        delete ef;

        setitr* seti;
        mtconverttoset(r,seti);
        return seti;
    }

    setitr* takemeas(const int idx) override
    {
        evalmformula* ef = new evalmformula(rec,idx);
        namedparams npslocal = nps;

        valms r = ef->eval(*fc, npslocal);
        delete ef;

        setitr* seti;
        mtconverttoset(r,seti);
        return seti;
    }

    formset( mrecords* recin , const std::vector<int>& litnumpsin,
        const std::vector<measuretype>& littypesin, const std::vector<std::string>& litnamesin,
        const namedparams& npsin, const std::string& fstr, const std::string shortnamein = "")
            : set( recin,  shortnamein == "" ? "st" : shortnamein, "Set-valued formula " + fstr ) {
        nps = npsin;
        bindnamedparams();
        fc = parseformula(fstr,litnumpsin,littypesin,litnamesin,nps, &global_fnptrs);
    };

    ~formset() {
        delete fc;
    }

};


class formtuple : public set
{
public:
    formulaclass* fc;

    setitr* takemeas( neighborstype* ns, const params& psin ) override
    {
        evalmformula* ef = new evalmformula(rec,ns);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc,npslocal);
        delete ef;
        setitr* seti;
        mtconverttotuple(r,seti);
        return seti;
    }

    setitr* takemeas(const int idx) override
    {
        evalmformula* ef = new evalmformula(rec,idx);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc,npslocal);
        delete ef;
        setitr* seti;
        mtconverttotuple(r,seti);
        return seti;
    }

    formtuple( mrecords* recin , const std::vector<int>& litnumpsin,
                const std::vector<measuretype>& littypesin, const std::vector<std::string>& litnamesin,
                namedparams& npsin,
                const std::string& fstr, const std::string shortnamein = "" )
        : set( recin,  shortnamein == "" ? "fp" : shortnamein, "Tuple-valued formula " + fstr)
    {
        nps = npsin;
        bindnamedparams();
        fc = parseformula(fstr,litnumpsin,littypesin,litnamesin,nps, &global_fnptrs);
    };

    ~formtuple() {
        delete fc;
    }

};


class formstring : public strmeas
{
public:
    formulaclass* fc;

    std::string* takemeas(neighborstype* ns, const params& ps) override
    {
        evalmformula* ef = new evalmformula(rec,ns);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc,npslocal);
        delete ef;
        std::string* out = new std::string();
        mtconverttostring(r,out);
        return out;
    }
    std::string* takemeas(const int idx) override
    {
        evalmformula* ef = new evalmformula(rec,idx);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc,npslocal);
        delete ef;
        std::string* out = new std::string();
        mtconverttostring(r,out);
        return out;
    }

    formstring( mrecords* recin , const std::vector<int>& litnumpsin,
                const std::vector<measuretype>& littypesin, const std::vector<std::string>& litnamesin,
                namedparams& npsin,
                const std::string& fstr, const std::string shortnamein = "" )
        : strmeas( recin,  shortnamein == "" ? "fr" : shortnamein, "String-valued formula " + fstr)
    {
        nps = npsin;
        bindnamedparams();
        fc = parseformula(fstr,litnumpsin,littypesin,litnamesin,nps, &global_fnptrs);
    };

    ~formstring() {
        delete fc;
    }

};


class formgraph : public gmeas
{
public:
    formulaclass* fc;

    neighborstype* takemeas(neighborstype* ns, const params& ps) override
    {
        evalmformula* ef = new evalmformula(rec,ns);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc, npslocal);
        delete ef;
        neighborstype* out;
        mtconverttograph(r,out);
        return out;
    }

    neighborstype* takemeas(const int idx) override
    {
        evalmformula* ef = new evalmformula(rec,idx);
        namedparams npslocal = nps;
        valms r = ef->eval(*fc, npslocal);
        delete ef;
        neighborstype* out;
        mtconverttograph(r,out);
        return out;
    }

    formgraph( mrecords* recin , const std::vector<int>& litnumpsin,
                const std::vector<measuretype>& littypesin, const std::vector<std::string>& litnamesin,
                const namedparams& npsin, const std::string& fstr, const std::string shortnamein = "" )
        : gmeas( recin,  shortnamein == "" ? "fg" : shortnamein, "Graph-valued formula " + fstr )
    {
        nps = npsin;
        bindnamedparams();
        fc = parseformula(fstr,litnumpsin,littypesin,litnamesin,nps, &global_fnptrs);
    };
    ~formgraph() {
        delete fc;
    }
};




class setitrpaths : public setitrmodeone
{
public:
    vertextype v1,v2;
    graphtype* g;
    neighborstype* ns;

    void compute() override
    {
        if (computed)
            return;
        totality.clear();
        std::vector<std::vector<vertextype>> out {};
        pathsbetweentuples(g,ns,v1,v2,out);

        // for (int i = 0; i < out.size(); ++i)
        // {
            // std::cout << "{";
            // for (int j = 0; j < out[i].size(); ++j)
                // std::cout << out[i][j] << ", ";
            // std::cout << "}" << std::endl;
        // }

        totality.resize(out.size());
        int j = 0;
        for (auto path : out)
        {
            std::vector<valms> pathvalms;
            pathvalms.resize(path.size());
            for (int i = 0; i < path.size(); ++i)
            {
                pathvalms[i].t = mtdiscrete;
                pathvalms[i].v.iv = path[i];
            }
            totality[j].t = mttuple;
            totality[j++].seti = new setitrmodeone(pathvalms);
        }
        computed = true;
        pos = -1;
    }

    setitrpaths( graphtype* gin, neighborstype* nsin, vertextype v1in, vertextype v2in )
        : g{gin}, ns{nsin}, v1{v1in}, v2{v2in}
    {
    }

    ~setitrpaths()
    {
        for (auto p : totality)
            delete p.seti;
    }

};

class setitrcyclesv : public setitrmodeone
{
public:
    vertextype v;
    graphtype* g;
    neighborstype* ns;

    void compute() override
    {
        if (computed)
            return;
        totality.clear();
        std::vector<std::vector<vertextype>> out {};
        cyclesvset(g,ns,v,out);

        // for (int i = 0; i < out.size(); ++i)
        // {
        // std::cout << "{";
        // for (int j = 0; j < out[i].size(); ++j)
        // std::cout << out[i][j] << ", ";
        // std::cout << "}" << std::endl;
        // }

        totality.resize(out.size());
        int j = 0;
        for (auto cycle : out)
        {
            std::vector<valms> pathvalms;
            pathvalms.resize(cycle.size());
            for (int i = 0; i < cycle.size(); ++i)
            {
                pathvalms[i].t = mtdiscrete;
                pathvalms[i].v.iv = cycle[i];
            }
            totality[j].t = mttuple;
            totality[j++].seti = new setitrtuple(pathvalms);
        }
        computed = true;
        pos = -1;
    }

    setitrcyclesv( graphtype* gin, neighborstype* nsin, vertextype vin )
        : g{gin}, ns{nsin}, v{vin}
    {
    }

    ~setitrcyclesv()
    {
        for (auto p : totality)
            delete p.seti;
    }

};

class setitrcycles : public setitrmodeone
{
public:
    graphtype* g;
    neighborstype* ns;

    void compute() override
    {
        if (computed)
            return;
        totality.clear();
        std::vector<std::vector<vertextype>> out {};
        cyclesset(g,ns,out);

        // for (int i = 0; i < out.size(); ++i)
        // {
        // std::cout << "{";
        // for (int j = 0; j < out[i].size(); ++j)
        // std::cout << out[i][j] << ", ";
        // std::cout << "}" << std::endl;
        // }

        totality.resize(out.size());
        int j = 0;
        for (auto cycle : out)
        {
            std::vector<valms> pathvalms;
            pathvalms.resize(cycle.size());
            for (int i = 0; i < cycle.size(); ++i)
            {
                pathvalms[i].t = mtdiscrete;
                pathvalms[i].v.iv = cycle[i];
            }
            totality[j].t = mttuple;
            totality[j++].seti = new setitrtuple(pathvalms);
        }
        computed = true;
        pos = -1;
    }

    setitrcycles( graphtype* gin, neighborstype* nsin )
        : g{gin}, ns{nsin}
    {
    }

    ~setitrcycles()
    {
        for (auto p : totality)
            delete p.seti;
    }

};

class abstractmakesubset {

    public:
    virtual setitr* makesubset( int maxint, bool* elts ) {return nullptr;};
};

class fastmakesubset : public abstractmakesubset {
public:
    setitrint* superset;
    setitr* makesubset( int maxint, bool* elts ) {
        auto out = new setitrint( maxint, elts );
        return out;
    }
    fastmakesubset( setitrint* supersetin ) : superset {supersetin} {}
};
class slowmakesubset : public abstractmakesubset {
public:
    setitr* superset;
    setitr* makesubset( int maxint, bool* elts ) {
        auto itrint = new setitrint( maxint, elts );
        auto out = new setitrsubset( superset->getitrpos(), itrint );
        return out;
    }
    slowmakesubset( setitr* supersetin ) : superset{supersetin} {}
};
inline abstractmakesubset* getsubsetmaker( setitr* superset ) {
    if (setitrint* cast = dynamic_cast<setitrint*>(superset))
        return new fastmakesubset( cast );
    return new slowmakesubset( superset );
}

class setitrpowerset : public setitr
{
protected:
    int subsetsize;
    int supersetsize;
    std::vector<int> subsetsv;
    int possubsetsv;
    int numberofsubsets = 1;
    setitrset inprocesssupersetitr;
    std::vector<valms> subsets {};
    // setitrsubset subsetitr;
    itrpos* supersetpos {};
    itrpos* inprocesssupersetpos;

    abstractmakesubset* subsetmaker;

    std::vector<std::vector<int>> ssvs {};
    int posssv;

public:

    setitr* supersetitr;

    void reset() override
    {
        // supersetsize = supersetitr->getsize();
        subsetsize = 0;
        // possubsetsv = -1;
        numberofsubsets = 1;
        // subsetsv.clear();
        // subsets.clear();
        // if (!supersetpos)
            // supersetpos = supersetitr->getitrpos();
        // supersetpos->reset();
        // inprocesssupersetpos->reset();
        pos = -1;

        posssv = -1;
    }
    bool ended() override
    {

        return posssv+1 >= numberofsubsets && subsetsize >= supersetsize;
        // return (subsetsize >= supersetsize-1) && (possubsetsv >= numberofsubsets-1);
    }
    valms getnext() override  // override this and then invoke it alongside populating elts
    {
        if (pos+1 < totality.size())
            return totality[++pos];

        valms r;
        r.t = mtset;

        bool* elts = new bool[supersetsize];
        // auto subset = new setitrsubset(supersetpos);
        // r.seti = subset;
        if (++posssv >= numberofsubsets)
        {
            ++subsetsize;
            numberofsubsets = ssvs[subsetsize].size()/subsetsize;
            posssv = 0;
        }
        // if (posssv == 0)
            // std::cout << "posssv == 0, subsetsize == " << subsetsize << " numberofsubsets == " << numberofsubsets << "\n";
        //memset(subset->itrint->elts, false, supersetsize*sizeof(bool));
        memset(elts,false,supersetsize*sizeof(bool));
        for (int j = 0; j < subsetsize; ++j)
            // subset->itrint->elts[ssvs[subsetsize][posssv*subsetsize + j]] = true;
            elts[ssvs[subsetsize][posssv*subsetsize + j]] = true;
        // if (subset.size() > 0)
        // subset->elts[subset->maxint] = true;
        // subset->itrint->computed = false;
        // subset->reset();
        r.seti = subsetmaker->makesubset(supersetsize-1,elts);
        totality.resize(++pos+1);
        totality[pos] = r;

        // std::cout << "subset ";
        // while (!subset->ended())
            // std::cout << subset->getnext().v.iv << ", ";
        // std::cout << std::endl;

        return r;

    }

    int getsize() override
    {
        if (supersetpos)
            return pow(2,supersetsize);
        return 0;
    }

    setitrpowerset(setitr* setin)
        : supersetitr{setin}, inprocesssupersetitr(), subsetmaker{getsubsetmaker(setin)}
    {
        t = mtset;
        if (supersetitr)
            supersetsize = supersetitr->getsize();
        else
            supersetsize = 0;
        // inprocesssupersetitr.totality.resize(0);
        // inprocesssupersetpos = inprocesssupersetitr.getitrpos();
        // subsetitr.setsuperset(inprocesssupersetpos);
        supersetpos = supersetitr->getitrpos();

        ssvs.resize(supersetsize+1);
        for (int i = 0; i <= supersetsize; ++i)
            enumsizedsubsets(0,i,nullptr,0,supersetsize,&(ssvs[i]));

        reset();
        // subsetsize = 0;
        // numberofsubsets = 1;
        // posssv = -1;
    }
    setitrpowerset() : supersetitr{nullptr}, inprocesssupersetitr()
    {
        t = mtset;
        supersetsize = 0;
        // inprocesssupersetpos = inprocesssupersetitr.getitrpos();
        // subsetitr.setsuperset(inprocesssupersetpos);
    }
    ~setitrpowerset()
    {
        delete supersetpos;
        delete subsetmaker;
    }
};

class setitrsizedsubset : public setitr
{
public:
    itrpos* setA;
    int supersetsize;

    std::vector<int> posAprimes {};
    int size = 2;

    abstractmakesubset* subsetmaker;

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
        // auto subset = new setitrsubset(setA);
        // r.seti = subset;
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
        auto elts = new bool[supersetsize];
        memset(elts, false, supersetsize*sizeof(bool));
        if (size > 0)
            elts[setA->pos] = true;
        for (int i = 0; i + 1 < size; ++i)
            elts[posAprimes[i]] = true;
//        memset(subset->itrint->elts, false, (subset->itrint->maxint+1)*sizeof(bool));
//        if (size > 0)
//            subset->itrint->elts[setA->pos] = true;
//        for (int i = 0; i < size-1; ++i)
//            subset->itrint->elts[posAprimes[i]] = true;
        // r.setsize = size;
        r.seti = subsetmaker->makesubset(supersetsize-1, elts);
        totality.resize(pos+1);
        totality[pos] = r;
        // std::cout << "pos " << pos << ": ";
        // for (int i = 0; i < size-1; ++i)
            // std::cout << posAprimes[i] << ", ";
        // std::cout << setA->pos << std::endl;
        return r;
    }
    setitrsizedsubset(setitr* Ain, int sizein ) : setA{Ain ? Ain->getitrpos() : nullptr}, size{sizein},
        supersetsize{Ain ? Ain->getsize() : 0}, subsetmaker{getsubsetmaker(Ain)}
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
        delete subsetmaker;
    //   for (auto t : totality) // this is handled by ~setitr() above
    //       delete t.seti;
    //   setitr::~setitr();
    }
};




class setitrsetpartitions : public setitr
{
public:
    itrpos* setA;
    int supersetsize;

    bool endedvar;
    std::vector<int> sequence {};
    std::vector<setitr*> subsets {};

    abstractmakesubset* subsetmaker;

    void codesubsets() {
        subsets.resize(supersetsize);

        int max = 0;
        for (int i = 0; i < supersetsize; ++i)
        {
            // subsets[i] = new setitrsubset(setA);
            auto elts = new bool[supersetsize];
            for (int j = 0; j < supersetsize; ++j)
            {
                elts[j] = sequence[j] == i;
                // subsets[i]->itrint->elts[j] = sequence[j] == i;
                max = max < sequence[j] ? sequence[j] : max;
            }
            subsets[i] = subsetmaker->makesubset(supersetsize-1, elts);
            subsets[i]->reset();
        }
        subsets.resize(max+1);

    }

    int getsize() override
    {
        supersetsize = setA->getsize();
        return bellNumber(supersetsize);
    }

    void reset() override
    {
        pos = -1;
        supersetsize = setA->getsize();
        setA->reset();
        sequence.resize(supersetsize);
        for (int i = 0; i < supersetsize; ++i)
            sequence[i] = 0;
        endedvar = supersetsize == 0;
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

            int j = supersetsize;
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
                for (int k = j+1; k < supersetsize; ++k)
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

    setitrsetpartitions(setitr* Ain ) : setA{Ain ? Ain->getitrpos() : nullptr}, subsetmaker{Ain ? getsubsetmaker(Ain) : nullptr}
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
            delete s;
        }
        delete subsetmaker;
    // for (auto t : totality) // this is handled by ~setitr() above
        // delete t.seti;
    //   setitr::~setitr();
    }
};



class setitrInitialsegment : public setitr
{
public:
    itrpos* superset {};
    int cutoff;
    void setsuperset( itrpos* supersetposin )
    {
        superset = supersetposin;
        reset();
    }
    int getsize() override
    {
        return cutoff;
    }
    valms getnext() override
    {
        valms res;
        if (++pos < totality.size())
            return totality[pos];
        res = superset->getnext();
        totality.resize(pos + 1);
        if (pos >= 0)
            totality[pos] = res;
        return res;
    }
    void reset() override
    {
        superset->reset();
        pos = -1;
    }
    bool ended() override
    {
        return pos + 1 >= cutoff;
    }
    setitrInitialsegment(itrpos* supersetin, const int cutoffin) : superset{supersetin}, cutoff{cutoffin}
    {
        t = superset->parent->t;
        pos = -1;
    };
    setitrInitialsegment() : superset{}
    {
        t = mtdiscrete;
        pos = -1;
    };
};

class Pset : public set
{
public:

    setitr* takemeas(neighborstype* ns, const params& ps ) override
    {
        auto itr = new setitrpowerset(ps[0].seti);
        return itr;
    }

    setitr* takemeas(const int idx, const params& ps ) override
    {
        auto itr = new setitrpowerset(ps[0].seti);
        return itr;
    }

    Pset( mrecords* recin ) : set(recin,"Ps", "Powerset")
    {
        valms v {};
        v.t = mtset;
        nps.push_back(std::pair{"set",v});
        bindnamedparams();
    }
};

class Permset : public set
{
    setitrmodeone* res {};
public:
    setitr* takemeas(const int idx, const params& ps ) override
    {
        // if (res)
            // for (auto v : res->totality)
                // delete v.seti;
        if (ps.size() == 1)
        {
            auto itr = ps[0].seti->getitrpos();
            std::vector<valms> v {};
            while (!itr->ended())
                v.push_back(itr->getnext());
            delete itr;
            std::vector<std::vector<int>> ps = getpermutations(v.size());
            std::vector<valms> totalitylocal {};
            for (auto p : ps)
            {
                std::vector<valms> tot {};
                for (auto i : p)
                    tot.push_back(v[i]);
                valms v;
                v.t = mttuple;
                v.seti = new setitrmodeone(tot);
                totalitylocal.push_back(v);
            }

            res = new setitrmodeone(totalitylocal);
            return res;
        }
        std::cout << "Error in Permset::takemeas\n";
        exit(1);
        return nullptr;;
    }

    Permset( mrecords* recin ) : set(recin,"Perms", "Set of permutation tuples")
    {
        valms v {};
        v.t = mtset;
        nps.push_back(std::pair{"set",v});
        bindnamedparams();
    }
    ~Permset()
    {
        if (res)
            for (auto v : res->totality)
                delete v.seti;
    }
};

class Stuple : public set
{
public:
    setitr* takemeas(const int idx, const params& ps) override
    {
        return new setitrInitialsegment(ps[0].seti->getitrpos(),ps[1].v.iv);
    }
    Stuple( mrecords* recin ) : set(recin,"Sp", "Tuple initial segment")
    {
        valms v;
        v.t = mttuple;
        nps.push_back(std::pair{"tuple",v});
        v.t = mtdiscrete;
        nps.push_back(std::pair{"n",v});
        bindnamedparams();
    }

};



class Vset : public set
{
public:
    setitr* takemeas(neighborstype* ns, const params& ps ) override
    {
        auto g = ns->g;
        auto itr = new setitrint(g->dim-1);
        memset(itr->elts,true,(itr->maxint+1)*sizeof(bool));
        itr->computed = false;
        itr->reset();
        return itr;
    }

    setitr* takemeas(const int idx, const params& ps) override
    {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);

        auto g = (*rec->gptrs)[idx];
        auto itr = new setitrint(g->dim-1);
        memset(itr->elts,true,(itr->maxint+1)*sizeof(bool));
        itr->computed = false;
        itr->reset();
        return itr;
    }
    Vset( mrecords* recin ) : set(recin,"V", "Graph vertices set") {}
};

class Eset : public set
{
public:
    setitr* takemeas(neighborstype* ns, const params& ps ) override
    {
        auto itr = new setitredges(ns->g);
        return itr;
    }

    setitr* takemeas(const int idx, const params& ps ) override
    {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    Eset( mrecords* recin ) : set(recin,"E", "Graph edges set") {}

};

class nEset : public set
{
public:
    graphtype* ginv;

    setitr* takemeas(neighborstype* ns, const params& ps ) override
    {
        auto g = ns->g;
        ginv = new graphtype(g->dim);
        for (int i = 0; i+1 <= g->dim; ++i)
        {
            ginv->adjacencymatrix[i*g->dim + i] = false;
            for (int j = i+1; j <= g->dim; ++j)
            {
                bool val = !g->adjacencymatrix[i*g->dim + j];
                ginv->adjacencymatrix[i*g->dim + j] = val;
                ginv->adjacencymatrix[j*g->dim + i] = val;
            }
        }
        ginv->adjacencymatrix[(g->dim-1)*g->dim + g->dim-1] = false;
        auto itr = new setitredges(ginv);
        return itr;
    }

    setitr* takemeas(const int idx, const params& ps ) override
    {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }

    nEset( mrecords* recin ) : set(recin,"nE", "Graph non-edges set") {}

    ~nEset()
    {
        delete ginv;
    }

};

class Sizedsubset : public set
{
public:
    setitr* takemeas(const int idx, const params& ps ) override
    {
        if (ps.size() == 2)
        {
            auto size = ps[1].v.iv;
            auto setA = ps[0].seti;
            // f[idx] = new setitrsizedsubset(setA,size );
            // if (setA == f)
                // std::cout << "Circular reference in Sizedsubset\n";
            // return f[idx];
            auto f = new setitrsizedsubset(setA,size);
            return f;
        }
        std::cout << "Error in Sizedsubset::takemeas\n";
        return nullptr;
    }

    Sizedsubset( mrecords* recin ) : set(recin,"Sizedsubset", "Sized subset")
    {
        valms v;
        v.t = mtset;
        nps.push_back(std::pair{"set",v});
        v.t = mtdiscrete;
        nps.push_back(std::pair{"size",v});
        bindnamedparams();
    }
    ~Sizedsubset()
    {
//        delete f;
    }
};


class Setpartition : public set
{
public:
    setitr* takemeas(neighborstype* ns, const params& ps ) override
    {
        if (ps.size() == 1)
        {
            auto setA = ps[0].seti;
            auto f = new setitrsetpartitions(setA);
            return f;
        }
        std::cout << "Error in Sizedsubset::takemeas\n";
        return nullptr;
    }

    setitr* takemeas(const int idx, const params& ps ) override
    {
        if (ps.size() == 1)
        {
            auto setA = ps[0].seti;
            auto f = new setitrsetpartitions(setA);
            return f;
        }
        std::cout << "Error in Sizedsubset::takemeas\n";
        return nullptr;
    }

    Setpartition( mrecords* recin ) : set(recin,"Setpartition", "Set partitions")
    {
        valms v;
        v.t = mtset;
        nps.push_back(std::pair{"set",v});
        bindnamedparams();
    }
    ~Setpartition()
    {
        //        delete f;
    }
};

/* TO DO
class Setsizedpartition : public set
{
public:
    setitr* takemeas(const int idx, const params& ps ) override
    {
        if (ps.size() == 2)
        {
            auto setA = ps[0].seti;
            auto s = ps[1].v.iv;
            auto f = new setitrsetpartitions(setA);
            return f;
        }
        std::cout << "Error in Sizedsubset::takemeas\n";
        return nullptr;
    }

    Setsizedpartition( mrecords* recin ) : set(recin,"Setpartition", "Set partitions")
    {
        valms v;
        v.t = mtset;
        nps.push_back(std::pair{"set",v});
        bindnamedparams();
    }
    ~Setsizedpartition()
    {
        //        delete f;
    }
}; */


class NNset : public set
{
public:
    // std::vector<setitrint*> f {};
    setitr* takemeas(const int idx, const params& ps ) override
    {
//        delete f;
        if (ps.size() == 1)
        {
            // if (f.size() <= idx)
                // f.resize(idx+1);
            int size = ps[0].v.iv;
            // f[idx] = new setitrint(size-1);
            // return f[idx];
            auto f = new setitrint(size-1);
            return f;
        }
        std::cout << "Error in NNset::takemeas\n";
        return nullptr;
    }

    NNset( mrecords* recin ) : set(recin,"NN", "Natural numbers finite set")
    {
        valms v;
        v.t = mtdiscrete;
        nps.push_back(std::pair{"n",v});
        bindnamedparams();
    }
};

class Nullset : public set
{
    public:
    setitrint* itr;
    setitr* takemeas(const int idx) override
    {
        return itr;
    }

    setitr* takemeas(const int idx, const params& ps) override
    {
        return itr;
    }

    Nullset( mrecords* recin ) : set(recin,"Nulls", "Null (empty) set"), itr{new setitrint(-1)} {}
    ~Nullset() {
        delete itr;
    }
};

class Subgraphsset: public set
{
    public:
    Subgraphsset( mrecords* recin ) : set(recin, "Subgraphs", "set of subgraphs") {}
};

class InducedSubgraphsset: public set
{ // Diestel p. 4
public:
    InducedSubgraphsset( mrecords* recin ) : set(recin, "InducedSubgraphss", "set of induced subgraphs") {}
};

inline void extendverticestocomponent( graphtype* g, neighborstype* ns, bool* vertices )
{
    const int dim = g->dim;
    bool changed = true;
    bool* donevertices = new bool[dim];
    memset(donevertices,0,sizeof(bool)*dim);
    while (changed)
    {
        changed = false;
        for (int i=0; i < dim; ++i)
        {
            if (vertices[i] && !donevertices[i]) {
                for (int j = 0; j < ns->degrees[i]; ++j )
                {
                    const int v = ns->neighborslist[i*dim + j];
                    changed = changed || !vertices[v];
                    vertices[v] = true;
                }
                donevertices[i] = true;
            }
        }
    }
    delete donevertices;
}

class Componentsset : public set
// Diestel p 12
{
public:
    setitr* takemeas( neighborstype* ns, const params& ps ) override
    {
        auto g = ns->g;
        setitrint* verticesset = new setitrint(g->dim-1);
        const int dim = g->dim;
        bool* vertices = new bool[dim];
        memset(vertices,0,sizeof(bool)*dim);
        std::vector<valms> tot {};
        for (int i = 0; i < dim; )
        {
            auto itr = verticesset->getitrpos();
            setitrsubset* component = new setitrsubset(itr);
            memset(component->itrint->elts,0, sizeof(bool)*dim);
            component->itrint->elts[i] = true;
            extendverticestocomponent(g,ns,component->itrint->elts);
            for (int j = 0; j < dim; ++j)
                vertices[j] = vertices[j] || component->itrint->elts[j];
            valms v;
            v.t = mtset;
            v.seti = component;
            tot.push_back(v);
            ++i;
            while (vertices[i] && i < dim)
                ++i;
        }
        delete vertices;
        return new setitrmodeone(tot);
    }
    setitr* takemeas(const int idx, const params& ps ) override
    {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    Componentsset( mrecords* recin ) : set(recin, "Componentss", "set of graph components") {}
};

class Maxconnectedgmeas : public gmeas
{

};

class Edgesset : public set {
public:
    setitr* takemeas(neighborstype* ns, const params& ps ) override {
        int dim = ns->g->dim;
        auto gtemp = new graphtype(dim);
        copygraph(ns->g,gtemp);
        auto itr = ps[0].seti->getitrpos();
        bool* negvertices = new bool[dim];
        memset(negvertices,true,sizeof(bool)*dim);
        while (!itr->ended()) {
            auto v = itr->getnext().v.iv;
            negvertices[v] = false;
        }
        for (auto i = 0; i < dim; ++i) {
            if (negvertices[i]) {
                for (auto j = 0; j < dim; ++j) {
                    gtemp->adjacencymatrix[i*dim+j] = false;
                    gtemp->adjacencymatrix[j*dim+i] = false;
                }
            }
        }
        auto out = new setitredges(gtemp);
        return out;
    }
    setitr* takemeas(const int idx, const params& ps ) override {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }

    Edgesset( mrecords* recin ) : set(recin, "Edgess", "Edges associated with the given set of vertices") {
        valms v;
        nps.push_back(std::pair{"U",v});
        bindnamedparams();
    }
};


class GraphonVEgmeas : public gmeas
{
public:
    neighborstype* takemeas(const int idx, const params& ps ) override
    {
        auto Upos = ps[0].seti->getitrpos();
        auto Fpos = ps[1].seti->getitrpos();
        int dim = Upos->getsize();
        auto gout = new graphtype(dim);
        int i = 0;
        gout->vertexlabels.resize(dim);
        auto vlabel = new std::string;
        while (!Upos->ended())
        {
            mtconverttostring(Upos->getnext(),vlabel);
            gout->vertexlabels[i++] = *vlabel;
        }
        memset(gout->adjacencymatrix,false,dim*dim*sizeof(bool));
        delete vlabel;
        while (!Fpos->ended())
        {
            auto pairitr = Fpos->getnext().seti->getitrpos();
            if (pairitr->getsize() != 2)
                std::cout << "Error non edge in set passed to GraphonVEg\n";
            else
            {
                auto v1 = pairitr->getnext().v.iv;
                auto v2 = pairitr->getnext().v.iv;
                if (v1 < dim && v2 < dim)
                {
                    gout->adjacencymatrix[v1*dim + v2] = true;
                    gout->adjacencymatrix[v2*dim + v1] = true;
                }
            }
            delete pairitr;
        }
        neighborstype* out = new neighborstype(gout);
        return out;
    }
    GraphonVEgmeas( mrecords* recin ) : gmeas(recin,"GraphonVEg", "graph on given vertices and edges")
    {
        valms v1;
        valms v2;
        v1.t = mtset;
        v2.t = mtset;
        nps.push_back(std::pair{"U",v1});
        nps.push_back(std::pair{"F",v2});
        bindnamedparams();
    }
};

class SubgraphonUgmeas : public gmeas
{
public:
    neighborstype* takemeas(neighborstype* ns, const params& ps ) override
    {
        auto g = ns->g;
        int dimg = g->dim;
        auto Upos = ps[0].seti->getitrpos();
        int dim = Upos->getsize();
        auto gout = new graphtype(dim);
        if (dim > 1)
        {
            int* vmap = (int*)(malloc(dim*sizeof(int)));
            int i = 0;
            bool labelled = g->vertexlabels.size() == dimg;
            if (labelled)
                gout->vertexlabels.resize(dim);
            while (!Upos->ended())
            {
                auto v = Upos->getnext();
                if (v.t == mtstring && labelled)
                {
                    for (int k = 0; k < dimg; ++k)
                        if (g->vertexlabels[k] == *v.v.rv)
                        {
                            vmap[i] = k;
                            gout->vertexlabels[i] = *v.v.rv;
                            break;
                        }
                } else
                {
                    int vidx;
                    mtconverttodiscrete(v,vidx);
                    vmap[i] = vidx;
                    if (labelled)
                        gout->vertexlabels[i] = ns->g->vertexlabels[vidx];
                }
                ++i;
            }
            memset(gout->adjacencymatrix,false,dim*dim*sizeof(bool));
            for (int i = 0; i+1 < dim; ++i)
                for (int j = i+1; j < dim; ++j)
                {
                    bool adj = g->adjacencymatrix[vmap[i]*dimg + vmap[j]];
                    gout->adjacencymatrix[i*dim + j] = adj;
                    gout->adjacencymatrix[j*dim + i] = adj;
                }
            delete vmap;
        } else
            gout->adjacencymatrix[0] = false;;
        neighborstype* out = new neighborstype(gout);
        return out;
    }
    neighborstype* takemeas(const int idx, const params& ps ) override
    {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    SubgraphonUgmeas( mrecords* recin ) : gmeas(recin,"SubgraphonUg", "Induced subgraph on given vertices")
    {
        valms v1;
        v1.t = mtset;
        nps.push_back(std::pair{"U",v1});
        bindnamedparams();
    }
};


class Ggmeas : public gmeas
{
public:
    neighborstype* takemeas(neighborstype* ns, const params& ps ) override
    {
        std::vector<std::string> tempstrv {};
        tempstrv.push_back(*ps[0].v.rv);
        auto g = igraphstyle(tempstrv);
        auto out = new neighbors(g);
        return out;
    }
    neighborstype* takemeas(const int idx, const params& ps ) override
    {
        return takemeas(nullptr,ps);
    }
    Ggmeas( mrecords* recin ) : gmeas(recin,"Gg", "Graph on given string")
    {
        valms v1;
        v1.t = mtstring;
        nps.push_back(std::pair{"s",v1});
        bindnamedparams();
    }
};

class TupletoSet : public set
{
    setitrmodeone* itr {};
    public:
    setitr* takemeas(const int idx, const params& ps) override
    {
/*        if (ps.size() != 1)
        {
            std::cout << "Error in TupletoSet::takemeas: wrong number of parameters\n";
            exit(1);
            return nullptr;
        }
        if (ps[0].t != mttuple)
        {
            std::cout << "Error in TupletoSet::takemeas: wrong type\n";
            exit(1);
        }*/

        if (setitrtuple* s = dynamic_cast<setitrtuple*>(ps[0].seti))
        {
            if (!s->computed)
                s->compute();
            auto res = new setitrmodeone(s->totality);
            return res;
        } else
        {
            std::cout << "Error in TupletoSet::takemeas: dynamic cast error, wrong type passed\n";
            exit(1);
        }
    }

    TupletoSet( mrecords* recin ) : set(recin, "TupletoSet", "Converts a tuple to a set")
    {
        valms v {};
        v.t = mttuple;
        nps.push_back(std::pair{"tuple",v});
        bindnamedparams();
    }
};

class Pathsset : public set
{
public:

    setitr* takemeas(neighborstype* ns, const params& ps) override
    {
/*        if (ps.size() != 2)
        {
            std::cout << "Incorrect number of parameters to Pathsset\n";
            exit(1);
        }
        if (ps[0].t != mtdiscrete || ps[1].t != mtdiscrete)
        {
            std::cout << "Incorrect parameter types passed to Pathsset\n";
            exit(1);
        }
*/
        auto g = ns->g;
        auto res = new setitrpaths(g,ns,ps[0].v.iv,ps[1].v.iv);
        return res;
    }


    setitr* takemeas(const int idx, const params& ps) override
    {
/*        if (ps.size() != 2)
        {
            std::cout << "Incorrect number of parameters to Pathsset\n";
            exit(1);
        }
        if (ps[0].t != mtdiscrete || ps[1].t != mtdiscrete)
        {
            std::cout << "Incorrect parameter types passed to Pathsset\n";
            exit(1);
        }
*/
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }

    Pathsset( mrecords* recin ) : set(recin,"Pathss", "Paths between two vertices set (ordered)")
    {
        valms v {};
        v.t = mtdiscrete;
        nps.push_back(std::pair{"v1",v});
        nps.push_back(std::pair{"v2",v});
        bindnamedparams();
    }

};

class Cyclesvset : public set
{
public:

    setitr* takemeas(const int idx, const params& ps) override
    {
/*        if (ps.size() != 1)
        {
            std::cout << "Incorrect number of parameters to Cyclesvset\n";
            exit(1);
        }
        if (ps[0].t != mtdiscrete)
        {
            std::cout << "Incorrect parameter types passed to Cyclesvset\n";
            exit(1);
        }
*/
        auto g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        auto res = new setitrcyclesv(g,ns,ps[0].v.iv);
        return res;
    }

    Cyclesvset( mrecords* recin ) : set(recin,"Cyclesvs", "Cycles from a vertex")
    {
        valms v {};
        v.t = mtdiscrete;
        nps.push_back(std::pair{"v",v});
        bindnamedparams();
    }

};

class Cyclesset : public set
{
public:

    setitr* takemeas(const int idx, const params& ps) override
    {
/*        if (ps.size() != 0)
        {
            std::cout << "Incorrect number of parameters to Cycless\n";
            exit(1);
        }
*/
        auto g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        auto res = new setitrcycles(g,ns);
        return res;
    }

    Cyclesset( mrecords* recin ) : set(recin,"Cycless", "All cycles") {}

};




#endif //AMEAS_H
