//
// Created by peterglenn on 7/16/24.
//

#include "ameas.h"
#include "graphs.h"
#include <unordered_set>

#include "cudagraph.cuh"
#include "cudaengine.cuh"


class kncrit : public crit
{
public:
    const int n;

    bool takemeas( neighborstype* ns, const params& ps) {
        int mincnt;
        if (ps.empty() || ps[0].t != mtdiscrete)
            mincnt = 1;
        else
            mincnt = ps[0].v.iv;
        if (n <= 0)
            return true;
        graphtype* g = ns->g;
        int dim = g->dim;
        std::vector<int> subsets {};
        enumsizedsubsets(0,n,nullptr,0,dim,&subsets);
        bool found = false;
        int foundcnt = 0;
        int j = 0;
        const int l = subsets.size()/n;
        while (foundcnt < mincnt && j < l) {
            found = true;
            for (auto i = 0; found && (i < n); ++i) {
                for (auto k = i+1; found && (k < n); ++k)
                    found = found && g->adjacencymatrix[dim*subsets[j*n + i] + subsets[j*n + k]];
            }
            foundcnt += (found ? 1 : 0);
            ++j;
        }
        return negated != (foundcnt >= mincnt);
    }
    bool takemeas( const int idx, const params& ps) override {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    kncrit( mrecords* recin , const int nin) : crit(recin ,"kn"+std::to_string(nin),"embeds K_" + std::to_string(nin) + " criterion"), n{nin}
    {
        valms p1;
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"n",p1});
        bindnamedparams();
    }


};

class indncrit : public crit
{
public:
    const int n;
    bool takemeas( neighborstype* ns, const params& ps) override {
        int mincnt;
        if (ps.empty() || ps[0].t != mtdiscrete)
            mincnt = 1;
        else
            mincnt = ps[0].v.iv;
        if (n <= 0)
            return true;
        graphtype* g = ns->g;
        int dim = g->dim;
        std::vector<int> subsets {};
        enumsizedsubsets(0,n,nullptr,0,dim,&subsets);
        bool found = false;
        int foundcnt = 0;
        int j = 0;
        const int l = subsets.size()/n;
        while (foundcnt < mincnt && j < l) {
            found = true;
            for (auto i = 0; found && (i < n); ++i) {
                for (auto k = i+1; found && (k < n); ++k)
                    found = found && !g->adjacencymatrix[dim*subsets[j*n + i] + subsets[j*n + k]];
            }
            foundcnt += (found ? 1 : 0);
            ++j;
        }
        return negated != (foundcnt >= mincnt);
    }
    bool takemeas( const int idx, const params& ps) override {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    indncrit( mrecords* recin , const int nin) : crit(recin ,"indn"+std::to_string(nin),"independent set of size" + std::to_string(nin) + " criterion"), n{nin}
    {
        valms p1 {};
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"n",p1});
        bindnamedparams();
    }
};


class knpcrit : public crit {
protected:
    std::vector<kncrit*> kns {};
public:
    void populatekns() {
        kns.resize(KNMAXCLIQUESIZE);
        for (int i = 0; i < KNMAXCLIQUESIZE; ++i) {
            auto kn = new kncrit(rec,i);
            kns[i] = kn;
        }
    }
    bool takemeas( neighborstype* ns, const params& ps ) override {
        params newps {};
        if (ps.size() >= 1 && ps[0].t == measuretype::mtdiscrete) {
            int ksz = ps[0].v.iv;
            int mincnt = 1;
            if (ps.size() == 2 && ps[1].t == measuretype::mtdiscrete)
            {
                mincnt = ps[1].v.iv;
                newps.push_back(ps[1]);
            }
            if (0<=ksz && ksz <= KNMAXCLIQUESIZE)
            {
                return negated != kns[ksz]->takemeas(ns,newps);
            } else {
                std::cout<< "Increase KNMAXCLIQUESIZE compiler define (current value " + std::to_string(KNMAXCLIQUESIZE) + ")";
                return false;
            }
        }
        // recode to prepare kn's in advance, and perhaps a faster algorithm than using FP
    }
    bool takemeas( const int idx, const params& ps ) override {
        params newps {};
        if (ps.size() >= 1 && ps[0].t == measuretype::mtdiscrete) {
            int ksz = ps[0].v.iv;
            int mincnt = 1;
            if (ps.size() == 2 && ps[1].t == measuretype::mtdiscrete)
            {
                mincnt = ps[1].v.iv;
                newps.push_back(ps[1]);
            }
            if (0<=ksz && ksz <= KNMAXCLIQUESIZE)
            {
                return negated != kns[ksz]->takemeas(idx,newps);
            } else {
                std::cout<< "Increase KNMAXCLIQUESIZE compiler define (current value " + std::to_string(KNMAXCLIQUESIZE) + ")";
                return false;
            }
        }
        // recode to prepare kn's in advance, and perhaps a faster algorithm than using FP
    }
    knpcrit( mrecords* recin  ) : crit( recin ,"Knc","Parameterized K_n criterion (parameters are complete set size and min count)") {
        populatekns();
        valms p1 {};
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"n",p1});
        nps.push_back(std::pair{"mincnt",p1});
        bindnamedparams();
    }
    ~knpcrit() {
        for (auto kn : kns)
            delete kn;
    }
};

class indnpcrit : public crit {
protected:
    std::vector<indncrit*> indns {};
public:
    void populateindns() {
        indns.resize(INDNMAXINDSIZE);
        for (int i = 0; i < INDNMAXINDSIZE; ++i) {
            auto indn = new indncrit(rec,i);
            indns[i] = indn;
        }
    }
    bool takemeas( neighborstype* ns, const params& ps ) override {
        params newps {};
        if (ps.size() >= 1 && ps[0].t == measuretype::mtdiscrete) {
            int indsz = ps[0].v.iv;
            int mincnt = 1;
            if (ps.size() == 2 && ps[1].t == measuretype::mtdiscrete)
            {
                mincnt = ps[1].v.iv;
                newps.push_back(ps[1]);
            }
            if (0<=indsz && indsz <= INDNMAXINDSIZE)
            {
                return negated != indns[indsz]->takemeas(ns,newps);
            } else {
                std::cout<< "Increase INDNMAXINDSIZE compiler define (current value " + std::to_string(INDNMAXINDSIZE) + ")";
                return false;
            }
        }
        // recode to prepare kn's in advance, and perhaps a faster algorithm than using FP
    }
    bool takemeas( const int idx, const params& ps ) override {
        params newps {};
        if (ps.size() >= 1 && ps[0].t == measuretype::mtdiscrete) {
            int indsz = ps[0].v.iv;
            int mincnt = 1;
            if (ps.size() == 2 && ps[1].t == measuretype::mtdiscrete)
            {
                mincnt = ps[1].v.iv;
                newps.push_back(ps[1]);
            }
            if (0<=indsz && indsz <= INDNMAXINDSIZE)
            {
                return negated != indns[indsz]->takemeas(idx,newps);
            } else {
                std::cout<< "Increase INDNMAXINDSIZE compiler define (current value " + std::to_string(INDNMAXINDSIZE) + ")";
                return false;
            }
        }
        // recode to prepare kn's in advance, and perhaps a faster algorithm than using FP
    }

    indnpcrit( mrecords* recin  ) : crit( recin ,"indnc","Parameterized Ind_n criterion (parameters are independent set size and min count)") {
        populateindns();
        valms p1 {};
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"n",p1});
        nps.push_back(std::pair{"mincnt",p1});
        bindnamedparams();
    }
    ~indnpcrit() {
        for (auto indn : indns)
            delete indn;
    }
};

class Kntally : public tally
{
public:
    int takemeas( neighborstype* ns, const params& ps ) override
    {
        int ksz = 0;
        if (ps.size() >= 1 && ps[0].t == mtdiscrete)
        {
            ksz = ps[0].v.iv;
        }
        if (ksz <= 0)
            return true;
        graphtype* g = ns->g;
        int dim = g->dim;
        std::vector<int> subsets {};
        enumsizedsubsets(0,ksz,nullptr,0,dim,&subsets);
        bool found = false;
        int foundcnt = 0;
        int j = 0;
        const int l = subsets.size()/ksz;
        while (j < l) {
            found = true;
            for (auto i = 0; found && (i < ksz); ++i) {
                for (auto k = i+1; found && (k < ksz); ++k)
                    found = found && g->adjacencymatrix[dim*subsets[j*ksz + i] + subsets[j*ksz + k]];
            }
            foundcnt += (found ? 1 : 0);
            ++j;
        }
        return foundcnt;
    }
    int takemeas( const int idx, const params& ps ) override
    {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    Kntally( mrecords* recin ) : tally( recin, "Knt", "K_n embeddings tally" )
    {
        valms p1 {};
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"n",p1});
        bindnamedparams();
    }
};

class indntally : public tally
{
public:
    int takemeas( neighborstype* ns, const params& ps ) override
    {
        int ksz = 0;
        if (ps.size() >= 1 && ps[0].t == mtdiscrete)
        {
            ksz = ps[0].v.iv;
        }

        if (ksz <= 0)
            return true;

        graphtype* g = ns->g;
        int dim = g->dim;

        std::vector<int> subsets {};
        enumsizedsubsets(0,ksz,nullptr,0,dim,&subsets);

        bool found = false;
        int foundcnt = 0;
        int j = 0;
        const int l = subsets.size()/ksz;
        while (j < l) {
            found = true;
            for (auto i = 0; found && (i < ksz); ++i) {
                for (auto k = i+1; found && (k < ksz); ++k)
                    found = found && !g->adjacencymatrix[dim*subsets[j*ksz + i] + subsets[j*ksz + k]];
            }
            foundcnt += (found ? 1 : 0);
            ++j;
        }
        return foundcnt;
    }
    int takemeas( const int idx, const params& ps ) override
    {
        auto ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    indntally( mrecords* recin ) : tally( recin, "indnt", "ind_n embeddings tally" )
    {
        valms p1 {};
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"n",p1});
        bindnamedparams();
    }

};


class embedscrit : public crit {
public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    embedscrit( mrecords* recin , neighbors* flagnsin,FP* fpin)
        : crit(recin , "embedsc","embeds type criterion"),
        flagg{flagnsin->g},flagns{flagnsin},fp{fpin} {}
    bool takemeas( neighborstype* ns, const params& ps ) override {
        return negated != (embedsquick(flagns, fp, ns, 1));
    }
    bool takemeas( const int idx ) override {
        return negated != (embedsquick(flagns, fp, (*this->rec->nsptrs)[idx], 1));
    }
    ~embedscrit()
    {
        freefps(fp,flagg->dim);
        delete fp;
        delete flagg;
        delete flagns;
        flagg = nullptr;
        flagns = nullptr;
        fp = nullptr;
    }
};


class forestcrit : public crit {
public:
    forestcrit( mrecords* recin  ) : crit(recin ,"forestc","Forest criterion") {}
    bool takemeas( neighborstype* ns, const params& ps ) override
    {
        auto g = ns->g;
        int dim = g->dim;
        if (dim <= 0)
            return negated != true;
        int* visited = (int*)malloc(dim*sizeof(int));
        for (int i = 0; i < dim; ++i)
        {
            visited[i] = -1;
        }
        visited[0] = 0;
        bool allvisited = false;
        while (!allvisited)
        {
            bool changed = false;
            for ( int i = 0; i < dim; ++i)
            {
                if (visited[i] >= 0)
                {
                    for (int j = 0; j < ns->degrees[i]; ++j)
                    {
                        vertextype nextv = ns->neighborslist[i*dim+j];
                        // loop = if a neighbor of vertex i is found in visited
                        // and that neighbor is not the origin of vertex i
                        bool loop = false;
                        if (visited[nextv] >= 0)
                            if (visited[nextv] != i)
                                if (visited[i] != nextv)
                                    loop = true;

                        if (loop)
                        {
                            free (visited);
                            return negated != false;
                        }
                        if (visited[nextv] < 0)
                        {
                            visited[nextv] = i;
                            changed = true;
                        }
                    }
                }
            }
            if (!changed) {
                allvisited = true;
                int firstunvisited = 0;
                while( allvisited && (firstunvisited < dim))
                {
                    allvisited = allvisited && (visited[firstunvisited] != -1);
                    ++firstunvisited;
                }
                if (allvisited)
                {
                    free (visited);
                    return negated != true;
                }
                visited[firstunvisited-1] = firstunvisited-1;
                changed = true;
            }
        }
    }
    bool takemeas( const int idx ) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        params ps {};
        return takemeas( ns, ps );
    }
};

class treecrit : public crit
{
public:
    treecrit(mrecords* recin ) : crit(recin,"treec","Tree criterion") {}

    bool takemeas( neighborstype* ns, const params& ps ) override
    {
        auto g = ns->g;
        int dim = g->dim;
        if (dim <= 0)
            return true;
        int* visited = (int*)malloc(dim*sizeof(int));

        for (int i = 0; i < dim; ++i)
        {
            visited[i] = -1;
        }

        visited[0] = 0;
        bool allvisited = false;
        while (!allvisited)
        {
            bool changed = false;
            for ( int i = 0; i < dim; ++i)
            {
                if (visited[i] >= 0)
                {
                    for (int j = 0; j < ns->degrees[i]; ++j)
                    {
                        vertextype nextv = ns->neighborslist[i*dim+j];
                        // loop = if a neighbor of vertex i is found in visited
                        // and that neighbor is not the origin of vertex i
                        bool loop = false;
                        if (visited[nextv] >= 0)
                            if (visited[nextv] != i)
                                if (visited[i] != nextv)
                                    loop = true;

                        if (loop)
                        {
                            free (visited);
                            return negated != false;
                        }
                        if (visited[nextv] < 0)
                        {
                            visited[nextv] = i;
                            changed = true;
                        }
                    }
                }
            }
            if (!changed) {
                allvisited = true;
                int firstunvisited = 0;
                while( allvisited && (firstunvisited < dim))
                {
                    allvisited = allvisited && (visited[firstunvisited] != -1);
                    ++firstunvisited;
                }
                if (allvisited)
                {
                    free (visited);
                    return negated != true;
                } else {
                    free (visited);
                    return negated != false;
                }
            }
        }
    }
    bool takemeas( const int idx ) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        params ps {};
        return takemeas( ns, ps );
    }
};


class truecrit : public crit{
protected:
public:

    bool takemeas(const int idx) override {
        return negated != true;
    }
    truecrit(mrecords* recin ) : crit(recin ,"truec","Always true") {}
};


class trianglefreecrit : public crit {
public:
    bool takemeas( neighborstype* ns, const params& ps ) override {
        auto g = ns->g;
        int dim = g->dim;
        for (int n = 0; n < dim-2; ++n) {
            for (int i = n+1; i < dim-1; ++i) {
                if (g->adjacencymatrix[n*dim + i]) {
                    for (int k = i+1; k < dim; ++k) {
                        if (g->adjacencymatrix[n*dim + k]
                            && g->adjacencymatrix[i*dim + k])
                            return negated != false;
                    }
                }
            }
        }
        //std::cout << "Triangle free!\n";
        //osadjacencymatrix(std::cout, g);
        //std::cout << "\n";
        return negated != true;
    }
    bool takemeas( const int idx, const params& ps)
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas( ns, ps );
    }
    bool takemeas( const int idx ) override {
        neighborstype* ns = (*rec->nsptrs)[idx];
        params ps {};
        return takemeas( ns, ps );
    }
    trianglefreecrit(mrecords* recin ) : crit(recin ,"cr1","Triangle-free crit") {}
};

class boolmeas : public meas {
public:
    double takemeas( const int idx, const params& ps ) override {
        return (double)true;
    }
    boolmeas(mrecords* recin) : meas( recin, "truem", "Always True measure") {}
};

class dimmeas : public meas {
public:
    double takemeas(neighborstype* ns, const params& ps) override
    {
        return ns->dim;
    }
    double takemeas( const int idx, const params& ps ) {
        return (*rec->gptrs)[idx]->dim;
    }

    dimmeas(mrecords* recin) : meas( recin, "dimm", "Graph's dimension") {}
};

class edgecntmeas : public meas {
public:

    double takemeas( neighborstype* ns, const params& ps ) {
        return edgecnt(ns->g);
    }
    double takemeas( const int idx, const params& ps ) {
        return edgecnt((*rec->gptrs)[idx]);
    }
    edgecntmeas(mrecords* recin) : meas( recin, "edgecm", "Graph's edge count") {}
};

class avgdegreemeas : public meas {
public:
    double takemeas( neighborstype* ns, const params& ps ) {
        int sum = 0;
        if (ns->dim == 0)
            return -1;
        for (int i = 0; i < ns->dim; ++i)
            sum += ns->degrees[i];
        return (double)sum / (double)ns->dim;
    }
    double takemeas( const int idx, const params& ps ) {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas( ns, ps );
    }
    avgdegreemeas(mrecords* recin) : meas(recin, "dm", "Graph's average degree") {}
};

class mindegreemeas : public meas {
public:
    double takemeas( neighborstype* ns, const params& ps ) {
        int min = ns->maxdegree;
        for (auto n = 0; n < ns->dim; ++n) {
            min = ns->degrees[n] < min ? ns->degrees[n] : min;
        }
        return min;
    }
    double takemeas( const int idx, const params& ps ) {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas( ns, ps );
    }
    mindegreemeas(mrecords* recin) : meas(recin, "deltam", "Graph's minimum degree") {}
};

class maxdegreemeas : public meas {
public:
    double takemeas( neighborstype* ns, const params& ps ) {
        return ns->maxdegree;
    }
    double takemeas( const int idx, const params& ps ) {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return ns->maxdegree;
    }
    maxdegreemeas(mrecords* recin) : meas( recin, "Deltam", "Graph's maximum degree") {}
};

class legacygirthmeas : public meas {
    // "girth" is shortest cycle (Diestel p. 8)
protected:
    std::vector<graphtype*> cyclegraphs {};
    std::vector<neighbors*> cycleneighbors {};
    std::vector<FP*> cyclefps {};
public:
    legacygirthmeas(mrecords* recin) : meas(recin, "legacygirthm","(Legacy alg.) Graph's girth") {
        for (int n = 0; n < GRAPH_PRECOMPUTECYCLESCNT; ++n) {
            auto cycleg = cyclegraph(n);
            neighbors* cyclens = new neighbors(cycleg);
            cyclegraphs.push_back(cycleg);
            cycleneighbors.push_back(cyclens);
            FP* cyclefp = (FP*)malloc(n*sizeof(FP));
            for (int i = 0; i < n; ++i) {
                cyclefp[i].ns = nullptr;
                cyclefp[i].nscnt = 0;
                cyclefp[i].v = i;
                cyclefp[i].parent = nullptr;
                cyclefp[i].invert = cyclens->degrees[i] >= int((n+1)/2); // = 2
            }
            takefingerprint(cyclens,cyclefp,n);
            cyclefps.push_back(cyclefp);
        }
    }
    ~legacygirthmeas() {
        for (int i = 0; i < cyclefps.size(); ++i ) {
            freefps(cyclefps[i],i);
            free (cyclefps[i]);
        }
        for (int i = 0; i < cyclegraphs.size(); ++i) {
            delete cyclegraphs[i];
            delete cycleneighbors[i];
        }
    }

    double takemeas( const int idx, const params& ps ) override
    {
        int n = 3;
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        bool embedsbool = false;
        if (cyclegraphs.size() > g->dim) {
            while (!embedsbool && n <= g->dim) {
                embedsbool = embedsbool || embeds(cycleneighbors[n], cyclefps[n], ns, 1);
                ++n;
            }
            if (embedsbool)
                return n-1;
        }

        if (g->dim >= cyclegraphs.size()) {
            for (int j = cyclegraphs.size(); j <= g->dim; ++j) {
                auto cycleg = cyclegraph(j);
                neighbors* cyclens = new neighbors(cycleg);
                FP* cyclefp = (FP*)malloc(j*sizeof(FP));
                for (int i = 0; i < j; ++i) {
                    cyclefp[i].ns = nullptr;
                    cyclefp[i].nscnt = 0;
                    cyclefp[i].v = i;
                    cyclefp[i].parent = nullptr;
                    cyclefp[i].invert = cyclens->degrees[i] >= int(j+1/2); // = 2
                }
                takefingerprint(cyclens,cyclefp,j);

                cyclegraphs.push_back(cycleg);
                cycleneighbors.push_back(cyclens);
                cyclefps.push_back(cyclefp);
            }
            while (!embedsbool && n <= g->dim) {
                embedsbool = embedsbool || embeds(cycleneighbors[n], cyclefps[n], ns,1);
                ++n;
            }
            if (embedsbool)
                return n-1;
        }
        if (!embedsbool) {
            return std::numeric_limits<double>::infinity();
        }
    }

};

class girthmeas : public meas {
public:
    girthmeas(mrecords* recin) : meas(recin, "girthm","Graph's girth") {}
    double takemeas( neighborstype* ns, const params& ps) override {
        auto g = ns->g;
        const int dim = g->dim;

        int* visited = (int*)malloc(dim*sizeof(int));
        memset(visited,0,dim*sizeof(int));

        int* originated = (int*)malloc(dim*sizeof(int));
        memset(originated,0,dim*sizeof(int));

        int mincyclesize = dim + 1;

        if (dim == 0)
        {
            free( visited );
            free( originated );
            return 0;
        }

        int startvertex = 0;
        int cyclesize = 1;
        visited[startvertex] = cyclesize;
        originated[startvertex] = startvertex + 1;
        while ((cyclesize <= dim) && (startvertex < dim))
        {
            for (int i = 0; i < dim; ++i)
            {
                if (visited[i] == cyclesize)
                {
                    for (int j = 0; j < ns->degrees[i]; ++j)
                    {
                        int n = ns->neighborslist[i*dim+j];
                        if (originated[i] != (n+1)) {
                            if (visited[n] > 0)
                            {
                                visited[n] = visited[n] + cyclesize - 1;
                                originated[n] = i+1;
                                mincyclesize = visited[n] < mincyclesize ? visited[n] : mincyclesize;
                            } else
                            {
                                visited[n] = cyclesize+1;
                                originated[n] = i+1;
                            }
                        }
                    }
                }
            }
            cyclesize++;
            if (cyclesize >= mincyclesize)
            {
                ++startvertex;
                if (startvertex < dim)
                {
                    cyclesize = 1;
                    memset(visited,0,dim*sizeof(int));
                    memset(originated,0,dim*sizeof(int));
                    visited[startvertex] = cyclesize;
                    originated[startvertex] = startvertex+1;
                } else
                    break;
            }
        }
        if (mincyclesize <= dim)
        {
            free( visited );
            free( originated);
            return mincyclesize;
        }

        free( visited );
        free( originated );
        return std::numeric_limits<double>::infinity();

    }
    double takemeas( const int idx, const params& ps ) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};



inline int cyclesearch( const graphtype* g, const neighborstype* ns, int* path, bool* visited, int n, const int limit )
{
    // if (limit < 3)
    // return;
    const int dim = g->dim;
    int res = 0;
    if (n >= limit)
    {
        bool r = g->adjacencymatrix[path[0]*dim + path[n-1]];
        res = r ? 1 : 0;

        return res;
    }
    int lastv = path[n-1];
    int d = ns->degrees[lastv];
    for (int i = 0; i < d; ++i )
    {
        int v = ns->neighborslist[lastv*dim + i];
        if (!visited[v])
        {
            path[n] = v;
            visited[v] = true;
            int k = cyclesearch(g,ns,path,visited,n+1,limit);
            res += k;
            visited[v] = false;
        }
    }
    return res;
}

class cycletally : public tally
{
public:
    cycletally( mrecords* recin ) : tally (recin, "cyclet", "Number of cycles of length n")
    {
        valms p1;
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"n",p1});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override {
        auto g = ns->g;
        int p;
        if (ps.size() > 0)
            switch (ps[0].t)
            {
        case mtcontinuous: p = (int)ps[0].v.dv; break;
        case mtdiscrete: p = ps[0].v.iv; break;
            }
        else
            p = 0;

        int dim = g->dim;
        if (dim < p || p < 3)
            return 0;

        int* path = (int*)malloc(p*sizeof(int));
        bool* visited = (bool*)malloc(dim*sizeof(bool));
        memset(visited,0,dim*sizeof(bool));

        int res = 0;
        for (int startv = 0; startv < dim; ++startv)
        {
            path[0] = startv;
            memset(visited,0,dim*sizeof(bool));
            visited[startv] = true;
            res += cyclesearch(g,ns,path,visited,1,p);
        }

        delete path;
        delete visited;
        return res/(2*p);
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }

};


class maxcliquemeas : public meas {
public:
    maxcliquemeas(mrecords* recin) : meas(recin,"cliquem","Graph's largest clique") {}
    double takemeas( neighborstype* ns, const params& ps) override {
        std::vector<kncrit*> kns {};
        for (int n = 2; n <= ns->dim; ++n) {
            auto kn = new kncrit(rec,n);
            kns.push_back(kn);
            if (!(kn->takemeas(ns,ps))) {
                for (auto kn : kns)
                    delete kn;
                kns.clear();
                return n-1;
            }
        }
        for (auto kn : kns)
            delete kn;
        kns.clear();
        return ns->dim;

    }

    double takemeas( const int idx, const params& ps ) override {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class connectedmeas : public meas
{
public:
    connectedmeas(mrecords* recin) : meas(recin, "connm", "Connected components number") {}
    double takemeas( neighborstype* ns, const params& ps ) override {
        graphtype* g = ns->g;
        int breaksize = -1; // by default wait until all connected components are counted
        if (ps.size() > 0)
            breaksize = ps[0].v.iv;
        return connectedcount( g, ns, breaksize );
    }
    double takemeas( const int idx, const params& ps ) override {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class connectedcrit : public crit {
public:
    connectedcrit( mrecords* recin ) : crit( recin, "connc", "Graph connected components less than criterion")
    {
        // the parameter is "less than" cutoff
        valms p1;
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"k",p1});
        bindnamedparams();
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        if (ps.size() < 1)
            return false;
        double p;
        switch (ps[0].t)
        {
        case mtcontinuous: p = ps[0].v.dv; break;
        case mtdiscrete: p = (double)ps[0].v.iv; break;
        }
        auto cm = new connectedmeas( rec );
        bool res = cm->takemeas(ns,ps) < p;
        delete cm;
        return res;
    }
    bool takemeas(const int idx, const params& ps) override
    {
        if (ps.size() < 1)
            return false;
        double p;
        switch (ps[0].t)
        {
        case mtcontinuous: p = ps[0].v.dv; break;
        case mtdiscrete: p = (double)ps[0].v.iv; break;
        }
        auto cm = new connectedmeas( rec );
        bool res = cm->takemeas(idx,ps) < p;
        delete cm;
        return res;
    }
};

class connected1crit : public crit {
public:
    connected1crit( mrecords* recin ) : crit( recin, "conn1c", "Graph 1-connected ") {}
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        auto cm = new connectedmeas( rec );
        params pslocal {};
        valms p1;
        p1.t = measuretype::mtdiscrete;
        p1.v.iv = 2;
        pslocal.push_back(p1);
        bool res = cm->takemeas(ns,pslocal) < 2;
        delete cm;
        return res;
    }
    bool takemeas(const int idx, const params& ps) override
    {
        auto cm = new connectedmeas( rec );
        params pslocal {};
        valms p1;
        p1.t = measuretype::mtdiscrete;
        p1.v.iv = 2;
        pslocal.push_back(p1);
        bool res = cm->takemeas(idx,pslocal) < 2;
        delete cm;
        return res;
    }
};

class radiusmeas : public meas
{
public:
    radiusmeas(mrecords* recin) : meas(recin, "radiusm", "Graph radius") {}
    double takemeas( neighborstype* ns, const params& ps ) override {
        graphtype* g = ns->g;
        int breaksize = -1; // by default wait until finding the actual (minimal) radius
        if (ps.size() > 0)
            breaksize = ps[0].v.iv;
        int dim = g->dim;
        if (dim <= 0)
            return 0;
        int* distances = (int*)malloc(dim*dim*sizeof(int));
        int* dextremes = (int*)malloc(dim*sizeof(int));
        for (int i = 0; i < dim; ++i)
        {
            dextremes[i] = -1;
            for (int j = 0; j < dim; ++j)
                distances[i*dim + j] = -1;
        }
        int v = -1;
        int nextv = 0;
        bool changed = true;
        int distance = 0;
        v = nextv;
        distances[v*dim + v] = distance;
        while (v != -1) {
            while (changed) {
                changed = false;
                for (int w = 0; w < dim; ++w) {
                    if (distances[v*dim + w] == distance ) {
                        for (int x = 0; x < ns->degrees[w]; ++x) {
                            int newv = ns->neighborslist[w*dim + x];
                            if (distances[v*dim + newv] < 0) {
                                distances[v*dim + newv] = distance+1;
                                changed = true;
                            }
                        }
                    }
                }
                if (changed)
                    distance++;
            }
            dextremes[v] = distance;
            nextv = v;
            while (dextremes[nextv] >= 0) {
                nextv++;
                if (nextv >= dim)
                    nextv = 0;
                if (nextv == v)
                    break;
            }
            if (distances[v*dim + nextv] < 0)
            {
                delete distances;
                delete dextremes;
                return std::numeric_limits<double>::infinity(); // the graph isn't connected
            }
            if ((breaksize >= 0) && (distance <= breaksize)) {
                double res = distance;
                for (int i = 0; i < dim; ++i)
                    if (distances[v*dim + i] < 0)
                        res = std::numeric_limits<double>::infinity();
                delete distances;
                delete dextremes;
                return res;  // the parameterized break-out option
            }
            if (nextv == v) {
                v = -1;
            } else {
                distance = 0;
                v = nextv;
                //nextvs.clear();
                changed = true;
                distances[v*dim + v] = distance;
            }
        }
        int min = dim + 1;
        for (int i = 0; i < dim; ++i) {
            //std::cout << dextremes[i] << "\n";
            min = dextremes[i] < min ? dextremes[i] : min;
        }
        delete distances;
        delete dextremes;
        return min;
    }
    double takemeas( const int idx, const params& ps ) override {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas( ns, ps );
    }
};

class radiuscrit : public crit
{
public:
    radiuscrit( mrecords* recin ) : crit( recin, "radiusc", "Radius less than criterion" )
    {
        // the parameter is "less than" cutoff
        valms p1;
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"max",p1});
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        if (ps.size() < 1)
            return false;
        double p;
        switch (ps[0].t)
        {
        case mtcontinuous: p = ps[0].v.dv; break;
        case mtdiscrete: p = (double)ps[0].v.iv; break;
        }
        auto rc = new radiusmeas( rec );
        bool res = rc->takemeas(ns,ps) < p;
        delete rc;
        return res;
    }
    bool takemeas(const int idx, const params& ps) override
    {
        if (ps.size() < 1)
            return false;
        double p;
        switch (ps[0].t)
        {
        case mtcontinuous: p = ps[0].v.dv; break;
        case mtdiscrete: p = (double)ps[0].v.iv; break;
        }
        auto rc = new radiusmeas( rec );
        bool res = rc->takemeas(idx,ps) < p;
        delete rc;
        return res;
    }
};

inline int recursecircumferencemeas( int* path, int pathsize, bool* visited, const graphtype* g, const neighborstype* ns, const int breaksize ) {
    if (pathsize == 0) {
        int respl = 0;
        bool found = true;
        int n = 0;
        while (found) {
            found = false;
            while (!found && n < g->dim) {
                found = !visited[n++];
            }
            if (found) {
                int newpath[] = {n-1};
                visited[n-1] = true;
                int newpl = recursecircumferencemeas(newpath,1,visited,g,ns,breaksize);
                if (breaksize > 2 && newpl > breaksize)
                    return newpl;
                //visited[n-1] = false;
                //for (int n = 0; n < g->dim; ++n) {
                //    int newpath[] = {n};
                //    int newpl = recursecircumferencemeas(newpath, 1, g, ns,breaksize);
                respl = (respl < newpl ? newpl : respl);
            }
        }
        return respl;
    }
    int respl = 0;
    int* newpath = new int[pathsize+1];
    for (int j = 0; j < pathsize; ++j) {
        newpath[j] = path[j];
    }
    for (int i = 0; i < ns->degrees[path[pathsize-1]]; ++i ) {
        int newpl = 0;
        int newv = ns->neighborslist[path[pathsize-1]*g->dim + i];
        int j = 0;
        bool found = false;
        for (; !found && (j < pathsize); ++j) {
            found = (path[j] == newv);
        }
        if (!found) {
            newpath[pathsize] = newv;
            visited[newv] = true;
            newpl = recursecircumferencemeas(newpath,pathsize+1,visited, g, ns,breaksize);
            //visited[newv] = false;
        } else {
            newpl = pathsize-j + 1;
        }
        if (breaksize > 2 && newpl > breaksize)
        {
            delete newpath;
            return newpl;
        }
        respl = (respl < newpl ? newpl : respl);
    }
    respl = respl == 2 ? 0 : respl;
    delete newpath;
    return respl;
}

inline int legacyrecursecircumferencemeas( int* path, int pathsize, const graphtype* g, const neighborstype* ns, const int breaksize ) {

    if (pathsize == 0) {
        int respl = 0;
        for (int n = 0; n < g->dim; ++n) {
            int newpath[] = {n};
            int newpl = legacyrecursecircumferencemeas(newpath, 1, g, ns,breaksize);
            respl = (respl < newpl ? newpl : respl);
        }
        return respl;
    }

    int respl = 0;
    for (int i = 0; i < ns->degrees[path[pathsize-1]]; ++i ) {
        int newpl = 0;
        int* newpath = new int [pathsize+1];
        int newv = ns->neighborslist[path[pathsize-1]*g->dim + i];
        bool found = false;
        int j = 0;
        for (; !found && (j < pathsize); ++j) {
            newpath[j] = path[j];
            found = (path[j] == newv);
        }
        if (!found) {
            newpath[pathsize] = newv;
            newpl = legacyrecursecircumferencemeas(newpath,pathsize+1, g, ns,breaksize);
        } else
            newpl = pathsize-j + 1;
        respl = (respl < newpl ? newpl : respl);
        delete newpath;
    }
    respl = respl == 2 ? 0 : respl;
    if (breaksize >= 3 && respl >= breaksize)
        return respl;

    return respl;
}

class circumferencemeas: public meas {
public:
    circumferencemeas(mrecords* recin) : meas(recin, "circm","Graph circumference") {}
    double takemeas( const int idx, const params& ps ) {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];

        int breaksize = -1; // by default compute the largest circumference possible
        if (ps.size() > 0)
            breaksize = ps[0].v.iv;

        int dim = g->dim;
        if (dim <= 0)
            return 0;

        bool* visited = new bool [g->dim];
        for (int i = 0; i < g->dim; ++i)
            visited[i] = false;
        auto res = recursecircumferencemeas(nullptr,0,visited,g,ns,breaksize);
        delete visited;
        return res;
    }
};

class legacycircumferencemeas: public meas {
public:
    legacycircumferencemeas(mrecords* recin) : meas(recin, "lcircm","(Legacy alg.) Graph circumference") {}
    double takemeas( const int idx, const params& ps ) {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        int breaksize = -1; // by default compute the largest circumference possible
        if (ps.size() > 0)
            breaksize = ps[0].v.iv;
        int dim = g->dim;
        if (dim <= 0)
            return 0;
        return legacyrecursecircumferencemeas(nullptr,0,g,ns,breaksize);
    }
};

class circumferencecrit : public crit
{
public:
    circumferencecrit( mrecords* recin )
        : crit( recin, "circc", "Circumference greater than criterion" )
    {
        // the parameter is "greater than" cutoff
        valms p1;
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"min",p1});
        bindnamedparams();
    }
    bool takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        int breaksize = -1; // by default compute the largest circumference possible
        double p;
        if (ps.size() > 0)
            switch (ps[0].t)
         {
         case mtcontinuous: p = ps[0].v.dv; break;
         case mtdiscrete: p = (double)ps[0].v.iv; break;
            }
        else
            p = 0;
        breaksize = p;
        int dim = g->dim;
        if (dim <= 0)
            return 0;
        bool* visited = new bool[g->dim];
//        memset(visited, false, g->dim * sizeof(bool)); NO TIME RIGHT NOW TO TEST THIS SPEEDIER ALTERNATIVE
        for (int i = 0; i < g->dim; ++i)
            visited[i] = false;
        auto res = recursecircumferencemeas(nullptr,0,visited,g,ns,breaksize) > breaksize;
        delete visited;
        return res;
    }
};

class diametermeas : public meas
{
public:
    diametermeas(mrecords* recin) : meas(recin, "diamm", "Graph diameter") {}
    double takemeas( neighborstype* ns, const params& ps ) override {
        graphtype* g = ns->g;
        int breaksize = -1; // by default wait until finding the actual (maximal) diameter
        if (ps.size() > 0)
            breaksize = ps[0].v.iv;
        int dim = g->dim;
        if (dim <= 0)
            return 0;
        int* distances = (int*)malloc(dim*dim*sizeof(int));
        int* dextremes = (int*)malloc(dim*sizeof(int));
        for (int i = 0; i < dim; ++i)
        {
            dextremes[i] = -1;
            for (int j = 0; j < dim; ++j)
                distances[i*dim + j] = -1;
        }
        int v = -1;
        int nextv = 0;
        bool changed = true;
        int distance = 0;
        v = nextv;
        distances[v*dim + v] = distance;
        while (v != -1) {
            while (changed) {
                changed = false;
                for (int w = 0; w < dim; ++w) {
                    if (distances[v*dim + w] == distance ) {
                        for (int x = 0; x < ns->degrees[w]; ++x) {
                            int newv = ns->neighborslist[w*dim + x];
                            if (distances[v*dim + newv] < 0) {
                                distances[v*dim + newv] = distance+1;
                                changed = true;
                            }
                        }
                    }
                }
                if (changed)
                    distance++;
            }
            dextremes[v] = distance;
            nextv = v;
            while (dextremes[nextv] >= 0) {
                nextv++;
                if (nextv >= dim)
                    nextv = 0;
                if (nextv == v)
                    break;
            }
            if (distances[v*dim + nextv] < 0)
            {
                delete distances;
                delete dextremes;
                return std::numeric_limits<double>::infinity(); // the graph isn't connected, so diameter is infinite
            }
            if ((breaksize >= 0) && (distance >= breaksize)) {
                double res = distance;
                for (int i = 0; i < dim; ++i)
                    if (distances[v*dim + i] < 0)
                        res = std::numeric_limits<double>::infinity();
                delete distances;
                delete dextremes;
                return res;  // the parameterized break-out option
            }
            if (nextv == v) {
                v = -1;
            } else {
                distance = 0;
                v = nextv;
                //nextvs.clear();
                changed = true;
                distances[v*dim + v] = distance;
            }
        }
        int max = 0;
        for (int i = 0; i < dim; ++i) {
            //std::cout << dextremes[i] << "\n";
            max = dextremes[i] > max ? dextremes[i] : max;
        }
        delete distances;
        delete dextremes;
        return max;
    }
    double takemeas( const int idx, const params& ps ) override {

        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas( ns, ps );
    }
};

class diametercrit : public crit
{
public:
    diametercrit( mrecords* recin ) : crit( recin, "diamc", "Diameter greater than criterion")
    {
        // the parameter is "greater than" cutoff
        valms p1 {};
        p1.t = measuretype::mtdiscrete;
        nps.push_back(std::pair{"min",p1});
        bindnamedparams();
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        if (ps.size() < 1)
            return false;
        double p;
        if (ps.size() > 0)
            switch (ps[0].t)
            {
        case mtcontinuous: p = ps[0].v.dv; break;
        case mtdiscrete: p = (double)ps[0].v.iv; break;
            }
        auto dm = new diametermeas( rec );
        bool res = dm->takemeas(ns,ps) > p;
        delete dm;
        return res;
    }
    bool takemeas(const int idx, const params& ps) override
    {
        if (ps.size() < 1)
            return false;
        double p;
        if (ps.size() > 0)
            switch (ps[0].t)
            {
        case mtcontinuous: p = ps[0].v.dv; break;
        case mtdiscrete: p = (double)ps[0].v.iv; break;
            }
        auto dm = new diametermeas( rec );
        bool res = dm->takemeas(idx,ps) > p;
        delete dm;
        return res;
    }
};

template<typename T> crit* critfactory(mrecords* recin) {return new T(recin);}
template<typename T> meas* measfactory(mrecords* recin) {return new T(recin);}
template<typename T> tally* tallyfactory(mrecords* recin) {return new T(recin);}
template<typename T> set* setfactory(mrecords* recin) {return new T(recin);}
template<typename T> set* tuplefactory(mrecords* recin) {return new T(recin);}
template<typename T> strmeas* stringfactory(mrecords* recin) {return new T(recin);}
template<typename T> gmeas* graphfactory(mrecords* recin) {return new T(recin);}

class embedstally : public tally {
public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    embedstally( mrecords* recin , neighbors* flagnsin,FP* fpin)
        : tally(recin , "embedst","embeds type tally"),
        flagg{flagnsin->g},flagns{flagnsin},fp{fpin} {}
    int takemeas( neighborstype* ns, const params& ps ) override {
        return embedscount(flagns, fp, ns);
    }
    int takemeas( const int idx ) override {
        return embedscount(flagns, fp, (*this->rec->nsptrs)[idx]);
    }
    ~embedstally()
    {
        freefps(fp,flagg->dim);
        delete fp;
        delete flagg;
        delete flagns;
        flagg = nullptr;
        flagns = nullptr;
        fp = nullptr;
    }
};

class kconnectedcrit : public crit {
// Diestel, Grath Theory, p. 11
    public:
    kconnectedcrit( mrecords* recin ) : crit( recin, "kconnc", "Graph k-connected ")
    {
        valms p1 {};
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"k",p1});
        bindnamedparams();
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        int k = 0;
        // if (ps.size() > 0 && ps[0].t == mtdiscrete)
        k = ps[0].v.iv;
        graphtype* g = ns->g;
        return kconnectedfn( g, ns, k);
    }
    bool takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class kappatally : public tally {
// Diestel, Graph Theory, page 12
public:
    kappatally( mrecords* recin ) : tally( recin, "kappat", "kappa: max k-connectedness") {}
    int takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        int k = g->dim;
        bool res = false;
        while (!res && k >= 0)
            res = kconnectedfn( g, ns, k-- );
        return k+1;
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class ledgeconnectedcrit : public crit {
// Diestel, Grath Theory, p. 12
public:
    ledgeconnectedcrit( mrecords* recin ) : crit( recin, "ledgeconnc", "Graph l-edge-connected ")
    {
        valms p1;
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"l",p1});
        bindnamedparams();
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        int l = 0;
        if (ps.size() > 0 && ps[0].t == mtdiscrete)
            l = ps[0].v.iv;
        graphtype* g = ns->g;
        return ledgeconnectedfn( g, ns, l);
    }

    bool takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class lambdatally : public tally {
// Diestel, Graph Theory, page 12
public:
    lambdatally( mrecords* recin ) : tally( recin, "lambdat", "lambda: max l-edge-connectedness") {}
    int takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        int l = g->dim;
        bool res = false;
        while (!res && l >= 0)
            res = ledgeconnectedfn( g, ns, l-- );
        return l+1;
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class vdtally : public tally {
public:
    vdtally( mrecords* recin ) : tally( recin, "vdt", "vertex degree")
    {
        valms p1;
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"v",p1});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override
    {
        vertextype v = 0;
        if (ps.size() > 0)
            v = ps[0].v.iv;
        return ns->degrees[v];
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        vertextype v = 0;
        if (ps.size() > 0)
            v = ps[0].v.iv;
        return ns->degrees[v];
    }
};

class acrit : public crit {
public:
    acrit( mrecords* recin ) : crit( recin, "ac", "vertices adjacent")
    {
        valms p1 {};
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"v1",p1});
        nps.push_back(std::pair{"v2",p1});
        bindnamedparams();
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        vertextype v1;
        vertextype v2;
//        if (ps.size() == 2)
//        {
            v1 = ps[0].v.iv;
            v2 = ps[1].v.iv;
            return g->adjacencymatrix[v1*g->dim + v2];
//        } else
//        {
//            std::cout << "Wrong number of parameters to ac\n";
//            exit(-1);
//        }
    }
    bool takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        // neighborstype* ns = (*rec->nsptrs)[idx];
        vertextype v1;
        vertextype v2;
//        if (ps.size() == 2)
//        {
            v1 = ps[0].v.iv;
            v2 = ps[1].v.iv;
            return g->adjacencymatrix[v1*g->dim + v2];
//        } else
//        {
//            std::cout << "Wrong number of parameters to ac\n";
//            exit(-1);
//        }
    }
};

class eadjcrit : public crit {
public:
    eadjcrit( mrecords* recin ) : crit( recin, "eadjc", "edges adjacent")
    {
        valms p1 {};
        p1.t = mtset;
        nps.push_back(std::pair{"e1",p1});
        nps.push_back(std::pair{"e2",p1});
        bindnamedparams();
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
//        if (ps.size() == 2) {
//            if (ps[0].t != mtset || ps[1].t != mtset)
//            {
//                std::cout << "Non set types passed to eadjc\n";
//                return false;
//            }
            bool res;
            auto itra = ps[0].seti->getitrpos();
            auto itrb = ps[1].seti->getitrpos();
            valms a1 = itra->getnext();
            valms b1 = itrb->getnext();
            valms a2 = itra->getnext();
            valms b2 = itrb->getnext();
            res = (a1 == b1 && !(a2 == b2)) || (a2 == b2 && !(a1 == b1)) || a1 == b2 || a2 == b1;
            delete itra;
            delete itrb;
            return res;
//        }
//        std::cout << "Incorrect number of parameters passed to eadjc\n";
//        return false;
    }
    bool takemeas(const int idx, const params& ps) override
    {
        return takemeas(nullptr,ps);
    }

};

class sizetally : public tally {
public:
    sizetally( mrecords* recin ) : tally( recin, "st", "set size tally")
    {
        valms p1;
        p1.t = mtset;
        nps.push_back(std::pair{"set",p1});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override
    {
//        if (ps.size() == 1) {
            return ps[0].seti->getsize();
//        }
//        return 0;
    }
    int takemeas(const int idx, const params& ps) override
    {
//        if (ps.size() == 1) {
            return ps[0].seti->getsize();
//        }
//        return 0;
    }
};

class lengthtally : public tally {
public:
    lengthtally( mrecords* recin ) : tally( recin, "lt", "tuple length tally")
    {
        valms p1;
        p1.t = mttuple;
        nps.push_back(std::pair{"tuple",p1});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override
    {
//        if (ps.size() == 1) {
            return ps[0].seti->getsize();
//        }
//        std::cout << "Incorrect number of parameters passed to lengthtally\n";
//        return 0;
    }
    int takemeas(const int idx, const params& ps) override
    {
//        if (ps.size() == 1) {
            return ps[0].seti->getsize();
//        }
//        std::cout << "Incorrect number of parameters passed to lengthtally\n";
//        return 0;
    }
};

class pctally : public tally {
public:
    pctally( mrecords* recin ) : tally( recin, "pct", "vertices path connected tally")
    {
        valms p1;
        p1.t = mtdiscrete;
        valms p2;
        p2.t = mtdiscrete;
        nps.push_back(std::pair{"v1",p1});
        nps.push_back(std::pair{"v2",p2});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        //osadjacencymatrix(std::cout, g);
//        if (ps.size() == 2) {
            // std::cout << ps[0].v.iv << " iv " << ps[1].v.iv << "\n";
            return pathsbetweencount(g,ns,ps[0].v.iv, ps[1].v.iv);
//        }
//        return 0;
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class cyclesvtally : public tally {
public:
    cyclesvtally( mrecords* recin ) : tally( recin, "cyclesvt", "cycles around a vertex tally")
    {
        valms p1;
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"v",p1});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        //osadjacencymatrix(std::cout, g);
//        if (ps.size() == 1) {
            // std::cout << ps[0].v.iv << " iv " << ps[1].v.iv << "\n";
            return cyclesvcount(g,ns,ps[0].v.iv); // undirected (cf. Diestel p 6-10) is half of directed
//        } else
//            std::cout << "Incorrect number of parameters passed to cyclesvtally\n";
//        return 0;
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class cyclestally : public tally {
public:
    cyclestally( mrecords* recin ) : tally( recin, "cyclest", "All cycles tally") {}
    int takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        //osadjacencymatrix(std::cout, g);
//        if (ps.size() == 0) {
            // std::cout << ps[0].v.iv << " iv " << ps[1].v.iv << "\n";
            return cyclescount(g,ns); // undirected (cf. Diestel p 6-10) is half of directed
//        } else
//            std::cout << "Incorrect number of parameters passed to cyclestally\n";
//        return 0;
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class ecrit : public crit
{
public:
    ecrit( mrecords* recin ) : crit( recin, "ec", "Edge holds in graph")
    {
        valms p1;
        p1.t = mtset;
        nps.push_back(std::pair{"e",p1});
        bindnamedparams();
    }
    bool takemeas( neighborstype* ns, const params& ps) override
    {
//        if (ps.size() != 1)
//            return false;
        graphtype* g = ns->g;
        auto itr = ps[0].seti->getitrpos();
        bool res = g->adjacencymatrix[g->dim * itr->getnext().v.iv + itr->getnext().v.iv];
        delete itr;
        return res;
    }
    bool takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class idxtally : public tally
{
public:
    idxtally( mrecords* recin ) : tally( recin, "idxt", "Index into an integer-comprised set")
    {
        valms p1;
        p1.t = mtset;
        nps.push_back(std::pair{"set",p1});
        p1.t = mtdiscrete;
        nps.push_back(std::pair{"idx",p1});
        bindnamedparams();
    }
    int takemeas( const int idx, const params& ps) override
    {
//        if (ps.size() != 2)
//        {
//            std::cout << "Wrong number of parameters to idxt\n";
//        }
        auto itr = ps[0].seti->getitrpos();
        if (itr->ended())
        {
            std::cout << "idxs called on empty set\n";
            return -1;  // better error-catching needed here
        }
        int res = itr->getnext().v.iv;
        for (int i = 0; i < ps[1].v.iv && !itr->ended(); ++i)
        {
            res = itr->getnext().v.iv;
        }
        delete itr;
        return res;
    }
};

class idxset : public set
{
public:
    idxset( mrecords* recin ) : set( recin, "idxs", "Index into a set-comprised set")
    {
        valms p1;
        valms p2;
        p1.t = mtset;
        nps.push_back(std::pair{"set",p1});
        p2.t = mtdiscrete;
        nps.push_back(std::pair{"idx",p2});
        bindnamedparams();
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
//        if (ps.size() != 2)
//        {
//            std::cout << "Wrong number of parameters to idxs\n";
//        }
        auto itr = ps[0].seti->getitrpos();
        if (itr->ended())
        {
            std::cout << "idxs called on empty set\n";
            return {};  // better error-catching needed here
        }
        setitr* res = itr->getnext().seti;
        for (int i = 0; i < ps[1].v.iv && !itr->ended(); ++i)
        {
            res = itr->getnext().seti;
        }
        delete itr;
        return res;
    }

};

class bipcrit : public crit
{
public:
    bipcrit( mrecords* recin ) : crit( recin, "bipc", "Bipartite")
    {
        valms p1;
        p1.t = mtset;
        nps.push_back(std::pair{"setA",p1});
        nps.push_back(std::pair{"setB",p1});
        bindnamedparams();
    }
    bool takemeas( neighborstype* ns, const params& ps) override
    {
//        if (ps.size() != 2)
//            return false;
        graphtype* g = ns->g;
        bool* eltsleft = nullptr;
        bool* eltsright = nullptr;
        int maxintleft;
        int maxintright;
        bool all = true;
        if (setitrint* slparent = dynamic_cast<setitrint*>(ps[0].seti))
            if (setitrint* srparent = dynamic_cast<setitrint*>(ps[1].seti))
            {
                eltsleft = slparent->elts;
                eltsright = srparent->elts;
                maxintleft = slparent->maxint;
                maxintright = srparent->maxint;
            }
        if (!eltsleft || !eltsright)
            if (setitrsubset* slparent = dynamic_cast<setitrsubset*>(ps[0].seti))
                if (setitrsubset* srparent = dynamic_cast<setitrsubset*>(ps[1].seti))
                {
                    eltsleft = slparent->itrint->elts;
                    eltsright = srparent->itrint->elts;
                    maxintleft = slparent->itrint->maxint;
                    maxintright = srparent->itrint->maxint;
                }
        if (!eltsleft || !eltsright)
        {
            std::cout << "Unknown class type passed to bipc, expecting setitrint or setitrsubset\n";
            return false;
        }
                /*
                for (int i = 0; i <= sl->maxint; ++i)
                    if (sl->elts[i])
                    {
                        int j = 0;
                        while (all && j <= sr->maxint)
                        {
                            all = all && (!sr->elts[j] || g->adjacencymatrix[i*g->dim + j]);
                            ++j;
                        }
                    }
                for (int j = 0; all && (j <= sr->maxint); ++j)
                    if (sr->elts[j])
                    {
                        int i = 0;
                        while (all && i <= sl->maxint)
                        {
                            all = all && (!sl->elts[i] || g->adjacencymatrix[i*g->dim + j]);
                            ++i;
                        }
                    }
                    */  // the above code commented out per Diestel's definition (p 17)

        for (int i = 0; all && (i < maxintleft); ++i)
            for (int j = i+1; all && (j <= maxintleft); ++j)
                all = !eltsleft[i] || !eltsleft[j] || !(g->adjacencymatrix[i*g->dim + j]);
        for (int i = 0; all && (i < maxintright); ++i)
            for (int j = i+1; all && (j <= maxintright); ++j)
                all = !eltsright[i] || !eltsright[j] || !(g->adjacencymatrix[i*g->dim + j]);
        return all;

        std::cout << "Non-integer (non-vertex) set passed to bipcrit\n";
        return false;
    }
    bool takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Ntally : public tally {
// Diestel p. ??
public:
    Ntally( mrecords* recin ) : tally( recin, "Nt", "Neighbors of a vertex set")
    {
        valms p1;
        p1.t = mtset;
        nps.push_back(std::pair{"v",p1});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
//        if (ps.size() == 1) {
            auto s = ps[0].seti->getitrpos();
            bool* S = (bool*)malloc(g->dim * sizeof(bool));
            memset(S,false,g->dim*sizeof(bool));
            while (!s->ended())
            {
                auto v = s->getnext();
                S[v.v.iv] = true;
            }
            bool* N = (bool*)malloc(g->dim * sizeof(bool));
            memset(N,false,g->dim * sizeof(bool));
            int cnt = 0;
            for (int i = 0; i < g->dim; ++i)
                if (S[i])
                    for (int j = 0; j < ns->degrees[i]; ++j)
                    {
                        vertextype nbr = ns->neighborslist[i*g->dim + j];
                        cnt += (!N[nbr] && !S[nbr]) ? 1 : 0;
                        N[nbr] = true;
                    }
            delete N;
            delete S;
            delete s;
            return cnt;
 //       }
 //       std::cout << "Wrong number of parameters or parameter types passed to Ntally\n";
 //       return 0;
    }
    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns, ps);
    }
};

class Nset : public set {
    // Diestel p. ??
public:

    Nset( mrecords* recin ) : set( recin, "Ns", "Set of neighbors of a vertex set")
    {
        valms p1;
        p1.t = mtset;
        nps.push_back(std::pair{"set",p1});
        bindnamedparams();
    }
    setitr* takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
//        if (ps.size() == 1) {
            auto s = ps[0].seti->getitrpos();
            // if (setitrint* s = dynamic_cast<setitrint*>(ps[0].seti))
            // {
            bool* S = (bool*)malloc(g->dim * sizeof(bool));
            memset(S,false,g->dim*sizeof(bool));
            while (!s->ended())
            {
                auto v = s->getnext();
                S[v.v.iv] = true;
            }
            bool* N = (bool*)malloc(g->dim * sizeof(bool));
            memset(N,false,g->dim * sizeof(bool));
            int cnt = 0;
            std::vector<valms> nbrs {};
            for (int i = 0; i < g->dim; ++i)
                if (S[i])
                    for (int j = 0; j < ns->degrees[i]; ++j)
                    {
                        vertextype nbr = ns->neighborslist[i*g->dim + j];
                        if (!N[nbr] && !S[nbr])
                        {
                            valms v;
                            v.t = mtdiscrete;
                            v.v.iv = nbr;
                            nbrs.push_back(v);
                            cnt++;
                        }
                        N[nbr] = true;
                    }
            delete N;
            delete S;
            delete s;
            valms res;
            return new setitrmodeone(nbrs);
            // }
//        }
//        std::cout << "Wrong number of parameters or parameter types passed to Nset\n";
//        return 0;
    }
    setitr* takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns, ps);
    }
};


class Nsscrit : public crit {
    // Diestel p. ??
    // note it doesn't check the case of overlapping sets
public:
    Nsscrit( mrecords* recin ) : crit( recin, "Nssc", "Neighboring vertex sets")
    {
        valms p1;
        valms p2;
        p1.t = mtset;
        p2.t = mtset;
        nps.push_back(std::pair{"setA",p1});
        nps.push_back(std::pair{"setB",p2});
        bindnamedparams();
    }
    bool takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
//        if (ps.size() == 2 && (ps[0].t == mtset || ps[0].t == mttuple) && (ps[1].t == mtset || ps[1].t == mttuple) ) {
            auto s1 = ps[0].seti->getitrpos();
            auto s2 = ps[1].seti->getitrpos();
            bool res = false;
            while (!s1->ended() && !res)
            {
                vertextype v1 = s1->getnext().v.iv;
                while (!s2->ended() && !res)
                {
                    vertextype v2 = s2->getnext().v.iv;
                    res = res || g->adjacencymatrix[v1*g->dim + v2];
                }
                s2->reset();
            }
            delete s1;
            delete s2;
            return res;
//        }
//        std::cout << "Wrong number or types of parameters passed to Nssc\n";
//        return false;
    }
    bool takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns, ps);
    }
};

class Nsstally : public tally {
    // cf. Diestel p. 187
    // note it doesn't check the case of overlapping sets
public:
    Nsstally( mrecords* recin ) : tally( recin, "Nsst", "Number of X-Y edges (X,Y disjoint sets)")
    {
        valms p1;
        p1.t = mtset;
        valms p2;
        p2.t = mtset;
        nps.push_back(std::pair{"setA",p1});
        nps.push_back(std::pair{"setB",p2});
        bindnamedparams();
    }
    int takemeas(neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
//        if (ps.size() == 2 && (ps[0].t == mtset || ps[0].t == mttuple) && (ps[1].t == mtset || ps[1].t == mttuple) ) {
            auto s1 = ps[0].seti->getitrpos();
            auto s2 = ps[1].seti->getitrpos();
            int res = 0;

            bool* vertices = (bool*)malloc(g->dim * sizeof(bool));
            memset (vertices, false, g->dim * sizeof(bool));
            while (!s2->ended())
                vertices[s2->getnext().v.iv] = true;

            while (!s1->ended())
            {
                vertextype v1 = s1->getnext().v.iv;
                for (int i = 0; i < g->dim; ++i) {
                    if (vertices[i])
                        res += g->adjacencymatrix[v1*g->dim + i] ? 1 : 0;
                }
            }
            delete s1;
            delete s2;
            return res;
//        }
//        std::cout << "Wrong number or types of parameters passed to Nsst\n";
//        return false;
    }

    int takemeas(const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns, ps);
    }
};

// following code around chromatic numbers found online

// Function to check if it's safe to color a vertex with a
// given color
inline bool isSafe(int v, const std::vector<std::vector<int> >& graph,
            const std::vector<int>& color, int c)
{
    for (int neighbor : graph[v]) {
        if (color[neighbor] == c) {
            return false; // If any adjacent vertex has the
            // same color, it's not safe
        }
    }
    return true;
}

// Backtracking function to find a valid coloring
inline bool graphColoringUtil(int v,
                    const std::vector<std::vector<int> >& graph,
                    std::vector<int>& color, int m)
{
    if (v == graph.size()) {
        return true; // All vertices are colored, a solution
        // is found
    }

    for (int c = 1; c <= m; ++c) {
        if (isSafe(v, graph, color, c)) {
            color[v] = c;

            // Recur for the next vertices
            if (graphColoringUtil(v + 1, graph, color, m)) {
                return true;
            }

            // Backtrack
            color[v] = 0;
        }
    }

    return false; // No solution found for this coloring
}

// Main function to find chromatic number
inline int graphColoring(const std::vector<std::vector<int> >& graph, int m)
{
    int n = graph.size();
    std::vector<int> color(n, 0);

    if (!graphColoringUtil(0, graph, color, m)) {
        // std::cout << "No feasible solution exists";
        return 0;
    }

    // Print the solution
    // std::cout << "Vertex colors: ";
    // for (int c : color) {
        // std::cout << c << " ";
    // }
    // std::cout << std::endl;

    // Count unique colors to determine chromatic number
    std::unordered_set<int> uniqueColors(color.begin(),
                                    color.end());
    return uniqueColors.size();
}

inline std::vector<int> graphColoringtuple(const std::vector<std::vector<int> >& graph, int m)
{
    int n = graph.size();
    std::vector<int> color(n, 0);

    if (!graphColoringUtil(0, graph, color, m)) {
        // std::cout << "No feasible coloring solution exists";
        return {};
    }

    // Print the solution
    // std::cout << "Vertex colors: ";
    // for (int c : color) {
        // std::cout << c << " ";
    // }
    // std::cout << std::endl;

    // Count unique colors to determine chromatic number
    std::unordered_set<int> uniqueColors(color.begin(),
                                    color.end());
    return color;
}

inline std::vector<std::vector<int>> convertadjacencymatrix( neighborstype* ns )
{
    std::vector<std::vector<int>> out;
    out.resize(ns->dim);
    for (int n = 0; n < ns->dim; ++n)
    {
        std::vector<int> neighbors {};
        for (int i = 0; i < ns->degrees[n]; ++i)
        {
            neighbors.push_back( ns->neighborslist[n*ns->dim + i]);
        }
        out[n] = neighbors;
    }
    // for (auto a : out)
    // {
        // for (auto b : a)
            // std::cout << b << " ";
        // std::cout << "\n";
    // }
    // std::cout << std::endl;

    return out;
}

class Chitally : public tally
{
public:
    Chitally( mrecords* recin ) : tally( recin, "Chit", "Chromatic number of graph using back-tracking algorithm") {}
    int takemeas( neighborstype* ns, const params& ps) override
    {
//        if (ps.size() != 0)
//        {
//            std::cout << "Wrong number of parameters to Chit\n";
//        }
        graphtype* g = ns->g;
        std::vector<std::vector<int> > graph = convertadjacencymatrix(ns);
        int c = g->dim == 0 ? 0 : 1;
        while (!graphColoring(graph,c) && c < g->dim)
            ++c;
        // auto res = graphColoring(convertadjacencymatrix(ns), g->dim);
        return c;
    }
    int takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Chituple : public set
{
public:
    Chituple( mrecords* recin ) : set( recin, "Chip", "Coloring of a graph using back-tracking algorithm") {}
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
//        if (ps.size() != 0)
//        {
//            std::cout << "Wrong number of parameters to Chip\n";
//        }
        graphtype* g = ns->g;
        std::vector<std::vector<int> > graph = convertadjacencymatrix(ns);
        int c = 1;
        while (!graphColoring(graph,c) && c < g->dim)
            ++c;
        auto color = graphColoringtuple(convertadjacencymatrix(ns), c);
        std::vector<int> tot;
        tot.resize(color.size());
        int i = 0;
        for (auto c : color )
        {
//            valms v;
//            v.t = mtdiscrete;
//            v.v.iv = c;
            tot[i++] = c;
        }
        auto res = new setitrtuple<int>(tot);
        return res;
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

// also sourced on the internet geeksforgeeks.org

// Function to find the chromatic number using the greedy
// coloring algorithm
inline int greedyColoring(const std::vector<std::vector<int> >& graph)
{
    int n = graph.size();
    std::vector<int> colors(n, -1);

    for (int v = 0; v < n; ++v) {
        std::unordered_set<int> usedColors;

        // Check neighbors and mark their colors as used
        for (int neighbor : graph[v]) {
            if (colors[neighbor] != -1) {
                usedColors.insert(colors[neighbor]);
            }
        }

        // Find the smallest available color
        for (int color = 1;; ++color) {
            if (usedColors.find(color)
                == usedColors.end()) {
                colors[v] = color;
                break;
                }
        }
    }

    // Find the maximum color used (chromatic number)
    int chromaticNumber
        = *std::max_element(colors.begin(), colors.end());
    return chromaticNumber;
}

inline std::vector<int> greedyColoringtuple(const std::vector<std::vector<int> >& graph)
{
    int n = graph.size();
    std::vector<int> colors(n, -1);

    for (int v = 0; v < n; ++v) {
        std::unordered_set<int> usedColors;

        // Check neighbors and mark their colors as used
        for (int neighbor : graph[v]) {
            if (colors[neighbor] != -1) {
                usedColors.insert(colors[neighbor]);
            }
        }

        // Find the smallest available color
        for (int color = 1;; ++color) {
            if (usedColors.find(color)
                == usedColors.end()) {
                colors[v] = color;
                break;
                }
        }
    }

    // Find the maximum color used (chromatic number)
    int chromaticNumber
        = *max_element(colors.begin(), colors.end()) + 1;
    return colors;
}


class Chigreedytally : public tally
{
public:
    Chigreedytally( mrecords* recin ) : tally( recin, "Chigreedyt", "Chromatic number of graph using greedy algorithm") {}
    int takemeas(neighborstype* ns, const params& ps) override
    {
//        if (ps.size() != 0)
//        {
//            std::cout << "Wrong number of parameters to Chigreedyt\n";
//        }
        graphtype* g = ns->g;
        auto res = greedyColoring(convertadjacencymatrix(ns));
        return res;
    }
    int takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Chigreedytuple : public set
{
public:
    Chigreedytuple( mrecords* recin ) : set( recin, "Chigreedyp", "Coloring of a graph using greedy algorithm") {}
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
//        if (ps.size() != 0)
//        {
//            std::cout << "Wrong number of parameters to Chigreedyp\n";
//        }
        graphtype* g = ns->g;
        auto color = greedyColoringtuple(convertadjacencymatrix(ns));
        std::vector<int> tot;
        tot.resize(color.size());
        int i = 0;
        for (auto c : color )
        {
            // valms v;
            // v.t = mtdiscrete;
            // v.v.iv = c;
            tot[i++] = c;
        }
        auto res = new setitrtuple<int>(tot);
        return res;
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Chiprimetally : public tally
{
public:
    Chiprimetally( mrecords* recin ) : tally( recin, "Chiprimet", "Edge chromatic number of graph using back-tracking algorithm") {}
    int takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        auto gedge = edgegraph(ns);
        auto nsedge = neighborstype(gedge);
        nsedge.computeneighborslist();
        std::vector<std::vector<int>> graph = convertadjacencymatrix(&nsedge);
        int c = nsedge.dim == 0 ? 0 : 1;
        while (!graphColoring(graph,c) && c < nsedge.dim)
            ++c;
        // auto res = graphColoring(convertadjacencymatrix(ns), g->dim);
        delete gedge;
        return c;
    }
    int takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Chiprimegreedytally : public tally
{
public:
    Chiprimegreedytally( mrecords* recin ) : tally( recin, "Chiprimegreedyt", "Edge chromatic number of graph using greedy algorithm") {}
    int takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        auto gedge = edgegraph(ns);
        auto nsedge = neighborstype(gedge);
        nsedge.computeneighborslist();
        std::vector<std::vector<int>> graph = convertadjacencymatrix(&nsedge);
        auto res = greedyColoring(graph);
        delete gedge;
        return res;
    }
    int takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Separatescrit : public crit
// Diestel p. --
// FORALL (v1 IN A, FORALL (v2 IN B, FORALL (p IN Pathss(v1,v2), p CAP X != Nulls)))
{
public:
    bool takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        auto itrA = ps[0].seti->getitrpos();
        auto itrB = ps[1].seti->getitrpos();
        auto itrC = ps[2].seti->getitrpos();
        bool* C = (bool*)malloc(g->dim * sizeof(bool));
        memset (C, false, g->dim * sizeof(bool));
        while (!itrC->ended())
            C[itrC->getnext().v.iv] = true;;
        delete itrC;
        bool res = true;
        while (!itrA->ended() && res)
        {
            vertextype v1 = itrA->getnext().v.iv;
            while (!itrB->ended() && res)
            {
                vertextype v2 = itrB->getnext().v.iv;
                std::vector<std::vector<vertextype>> out {};
                pathsbetweentuples( g, ns, v1, v2, out );
                for (auto p : out)
                {
                    bool found = false;
                    for (auto v : p)
                    {
                        found = found || C[v];
                        if (found)
                            break;
                    }
                    res = res && found;
                    if (!res)
                        break;
                }
            }
            if (res)
                itrB->reset();
        }
        delete C;
        delete itrA;
        delete itrB; // C is deleted above
        return res;
    }

    bool takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    Separatescrit( mrecords* recin ) : crit( recin, "Separatesc", "Sets A and B separated by set X")
    {
        valms v1;
        v1.t = mtset;
        nps.push_back(std::pair{"A",v1});
        valms v2;
        v2.t = mtset;
        nps.push_back(std::pair{"B",v2});
        valms v3;
        v3.t = mtset;
        nps.push_back(std::pair{"X",v3});
        bindnamedparams();
    }
};

class connvsscrit : public crit
{
    public:
    bool takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        auto itrA = ps[0].seti->getitrpos();
        auto itrB = ps[1].seti->getitrpos();
        bool res = false;
        while (!itrA->ended() && !res)
        {
            vertextype v1 = itrA->getnext().v.iv;
            while (!itrB->ended() && !res)
            {
                vertextype v2 = itrB->getnext().v.iv;
                res = res || pathsbetweenmin( g, ns, v1, v2, 1);

            }
            if (!res)
                itrB->reset();
        }
        delete itrA;
        delete itrB;
        return res;
    }
    bool takemeas( const int idx, const params& ps) override {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns, ps);
    }
    connvsscrit( mrecords* recin ) : crit( recin, "connvssc", "Something in set A has a path to set B")
    {
        valms v;
        v.t = mtset;
        nps.push_back(std::pair{"A",v});
        nps.push_back(std::pair{"B",v});
        bindnamedparams();
    }
};


class connvcrit : public crit
{
public:
    bool takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        return pathsbetweenmin( g, ns, ps[0].v.iv, ps[1].v.iv, 1);
    }
    bool takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    connvcrit( mrecords* recin ) : crit( recin, "connvc", "exists at least one path from a to b")
    {
        valms v;
        v.t = mtdiscrete;
        nps.push_back(std::pair{"v1",v});
        nps.push_back(std::pair{"v2",v});
        bindnamedparams();
    }
};

class connvscrit : public crit
{
public:
    bool takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        auto vitr = ps[0].seti->getitrpos();
        auto edgeitr = ps[1].seti->getitrpos();
        bool* vertices = (bool*)malloc(g->dim * sizeof(bool));
        memset (vertices, false, g->dim * sizeof(bool));
        while (!vitr->ended())
            vertices[vitr->getnext().v.iv] = true;
        graphtype* gsub = new graphtype(g->dim);
        memset (gsub->adjacencymatrix, false, g->dim * g->dim * sizeof(bool));
        while (!edgeitr->ended())
        {
            valms e = edgeitr->getnext();
            auto eitr = e.seti->getitrpos();
            vertextype v1 = eitr->getnext().v.iv;
            vertextype v2 = eitr->getnext().v.iv;
            delete eitr;
            if (vertices[v1] && vertices[v2]) // ... or (different functionality) add "&& g->adjacencymatrix[v1*g->dim + v2]"
            {
                gsub->adjacencymatrix[v1 * g->dim + v2] = true;
                gsub->adjacencymatrix[v2 * g->dim + v1] = true;
            }
        }
        delete vitr;
        delete edgeitr;
        auto nssub = new neighborstype(gsub);
        nssub->computeneighborslist();
        bool res = connectedsubsetcount(gsub,nssub,vertices,2) <= 1;
        delete vertices;
        delete gsub;
        delete nssub;
        return res;
    }
    bool takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
    connvscrit( mrecords* recin ) : crit( recin, "connvsc", "Set is connected using given edges")
    {
        valms v;
        v.t = mtset;
        nps.push_back(std::pair{"vertexset",v});
        nps.push_back(std::pair{"edgeset",v});
        bindnamedparams();
    }
};


class CUDAnwalksbetweentuple : public set
{
public:
    CUDAnwalksbetweentuple( mrecords* recin ) :
        set( recin, "CUDAnwalksbetweenp", "CUDA array of walk counts of given length between each vertex pair")
    {
        valms v;
        v.t = mtdiscrete;
        nps.push_back(std::pair{"walk length",v});
        bindnamedparams();
    };
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        int dim = ns->g->dim;
        int* out;
        out = new int[dim*dim];
        CUDAcountpathsbetweenwrapper(out,ps[0].v.iv,ns->g->adjacencymatrix,dim);
        return new setitrtuple<int>(dim*dim,out);
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class nwalksbetweentuple : public set
{
public:
    nwalksbetweentuple( mrecords* recin ) :
        set( recin, "nwalksbetweenp", "Array of walk counts of given length between each vertex pair")
    {
        valms v;
        v.t = mtdiscrete;
        nps.push_back(std::pair{"walk length",v});
        bindnamedparams();
    };
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        int dim = ns->g->dim;
        int in1[dim*dim];
        int in2[dim*dim];
        int* out;
        out = new int[dim*dim];
        memset(in1,0,sizeof(int) * dim * dim);
        for (int i = 0; i < dim; ++i)
            in1[i*dim+i] = 1;
        for (int i = 0; i < dim*dim; i++)
            in2[i] = ns->g->adjacencymatrix[i];
        for (int i = 0; i < ps[0].v.iv; i++)
        {
            squarematrixmultiply(out,in1,in2,dim);
            memcpy(in1,out,dim * dim * sizeof(int));
        }
        return new setitrtuple<int>(dim*dim,out);
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};


class Connvtuple : public set
{
public:
    Connvtuple( mrecords* recin ) : set( recin, "Connv", "Matrix of connectedness") {}
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        const int dim = g->dim;
        bool* out = new bool[dim*dim];
        verticesconnectedmatrix( out, g, ns );
        std::vector<setitrtuple<bool>*> tuples {};
        tuples.resize(dim);
        for (int i = 0; i < dim; ++i)
            tuples[i] = new setitrtuple<bool>(dim,&out[i*dim]);
        auto res = new setitrtuple2d<bool>(tuples);
        return res;
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class CUDAConnvtuple : public set
{
public:
    CUDAConnvtuple( mrecords* recin ) :
        set( recin, "CUDAConnv", "CUDA boolean matrix of connected vertices") {};
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        const int dim = g->dim;
        bool* out = new bool[dim*dim];
        CUDAverticesconnectedmatrix( out, g, ns );
        std::vector<setitrtuple<bool>*> tuples {};
        tuples.resize(dim);
        for (int i = 0; i < dim; ++i)
            tuples[i] = new setitrtuple<bool>(dim,&out[i*dim]);
        auto res = new setitrtuple2d<bool>(tuples);
        return res;
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class conntally : public tally
{ // to deprecate: too slow
public:
    conntally(mrecords* recin) : tally(recin, "connt", "Connected components tally") {}
    int takemeas( neighborstype* ns, const params& ps ) override {
        graphtype* g = ns->g;
        const int dim = g->dim;
        const int pfactor = dim;
        auto partitions = (vertextype*)malloc(dim*pfactor*sizeof(vertextype));
        if (!partitions)
        {
            std::cout << "Error allocating enough memory in call to verticesconnectedmatrix\n";
            exit(1);
        }
        // for (int i = 0; i < dim; ++i)
        // memcpy(&partitions[i*pfactor],&ns->neighborslist[i*dim],dim*sizeof(vertextype));
        memcpy(partitions,ns->neighborslist,dim*pfactor*sizeof(vertextype));
        auto pindices = (int*)malloc(dim*sizeof(int));
        memcpy(pindices,ns->degrees,dim*sizeof(int));
        verticesconnectedlist( g, ns, partitions, pindices );
        int res = 0;
        for (int i = 0; i < dim; ++i)
            res += pindices[i] > 0;
        delete pindices;
        delete partitions;
        return res;
    }
    int takemeas( const int idx, const params& ps ) override {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Connc : public set
{
public:
    Connc( mrecords* recin ) :
        set( recin, "Connc", "Set of connected components") {};
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        const int dim = g->dim;

        std::vector<bool*> outv;

        connectedpartition(g,ns,outv);

        std::vector<valms> res {};
        for (auto elt : outv)
        {
            valms v;
            v.t = mtset;
            v.seti = new setitrint(g->dim-1,elt);
            res.push_back(v);
        }
        return new setitrmodeone(res);
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};

class Connmatrix : public set
{
public:
    Connmatrix( mrecords* recin ) :
        set( recin, "Connmatrix", "Matrix of vertex connectedness") {};
    setitr* takemeas( neighborstype* ns, const params& ps) override
    {
        graphtype* g = ns->g;
        const int dim = g->dim;

        std::vector<bool*> outv;
        connectedpartition(g,ns,outv);

        std::vector<setitrtuple<bool>*> tuples;
        std::vector<bool*> matrix;
        matrix.resize(dim);
        for (int i = 0; i < dim; ++i)
        {
            matrix[i] = new bool[dim];
            memset(matrix[i],false,dim*sizeof(bool));
        }
        for (int i = 0; i < outv.size(); ++i)
            for (int j = 0; j < dim; ++j)
                for (int k = 0; k < dim; ++k)
                    matrix[j][k] += outv[i][j] && outv[i][k];
        tuples.resize(dim);
        for (int i = 0; i < dim; ++i)
            tuples[i] = new setitrtuple<bool>(dim,matrix[i]);
        auto res = new setitrtuple2d<bool>(tuples);

        return res;
    }
    setitr* takemeas( const int idx, const params& ps) override
    {
        neighborstype* ns = (*rec->nsptrs)[idx];
        return takemeas(ns,ps);
    }
};
