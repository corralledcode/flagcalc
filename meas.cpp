//
// Created by peterglenn on 7/16/24.
//

#include "ameas.h"
#include "graphs.h"

#define GRAPH_PRECOMPUTECYCLESCNT 15

class kncrit : public crit
{
public:
    const int n;
    bool takemeas( const int idx, const params& ps) override {
        int mincnt;
        if (ps.empty() || ps[0].t != mtdiscrete)
            mincnt = 1;
        else
            mincnt = ps[0].v.iv;

        if (n <= 0)
            return true;

        graphtype* g = (*rec->gptrs)[idx];
        int dim = g->dim;

        std::vector<int> subsets {};
        enumsizedsubsets(0,n,nullptr,0,dim,&subsets);

        bool found = false;
        int foundcnt = 0;
        int j = 0;
        while ((foundcnt < mincnt) && (j < (subsets.size()/n))) {
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

    kncrit( mrecords* recin , const int nin) : crit(recin ,"kn"+std::to_string(nin),"embeds K_" + std::to_string(nin) + " criterion"), n{nin}
    {
        ps.clear();
        pssz = 1;
        valms p1;
        p1.t = measuretype::mtdiscrete;
        ps.push_back(p1);
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

    knpcrit( mrecords* recin  ) : crit( recin ,"Knc","Parameterized K_n criterion (parameter is complete set size)") {
        populatekns();
        ps.clear();
        valms p1;
        p1.t = measuretype::mtdiscrete;
        p1.v.iv = 0;
        ps.push_back(p1);
        ps.push_back(p1);
        pssz = 2;
    }
    ~knpcrit() {
        for (auto kn : kns)
            delete kn;
    }
};

class Kntally : public tally
{
public:
    int takemeas( const int idx, const params& ps ) override
    {
        int ksz = 0;
        if (ps.size() >= 1 && ps[0].t == mtdiscrete)
        {
            ksz = ps[0].v.iv;
        }

        if (ksz <= 0)
            return true;

        graphtype* g = (*rec->gptrs)[idx];
        int dim = g->dim;

        std::vector<int> subsets {};
        enumsizedsubsets(0,ksz,nullptr,0,dim,&subsets);

        bool found = false;
        int foundcnt = 0;
        int j = 0;
        while (j < (subsets.size()/ksz)) {
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

    Kntally( mrecords* recin ) : tally( recin, "Knt", "K_n embeddings tally" )
    {
        ps.clear();
        valms p1;
        p1.t = mtdiscrete;
        p1.v.iv = 0;
        ps.push_back(p1);
        pssz = 1;
    }

};


class embedscrit : public crit {
protected:

public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    embedscrit( mrecords* recin , neighbors* flagnsin,FP* fpin)
        : crit(recin , "embedsc","embeds type criterion"),
        flagg{flagnsin->g},flagns{flagnsin},fp{fpin}
    {
        pssz = 0;
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


class forestcrit : public crit
{
public:

    forestcrit( mrecords* recin  ) : crit(recin ,"forestc","Forest criterion") {}

    bool takemeas( const int idx ) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
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


};

class treecrit : public crit
{
public:

    treecrit(mrecords* recin ) : crit(recin,"treec","Tree criterion") {}

    bool takemeas( const int idx ) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
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
    bool takemeas( const int idx, const params& ps)
    {
        return takemeas(idx);
    }
    bool takemeas( const int idx ) override {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
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
    double takemeas( const int idx, const params& ps ) {
        return (*rec->gptrs)[idx]->dim;
    }

    dimmeas(mrecords* recin) : meas( recin, "dimm", "Graph's dimension") {}
};

class edgecntmeas : public meas {
public:

    double takemeas( const int idx, const params& ps ) {
        return edgecnt((*rec->gptrs)[idx]);
    }

    edgecntmeas(mrecords* recin) : meas( recin, "edgecm", "Graph's edge count") {}
};

class avgdegreemeas : public meas {
public:
    double takemeas( const int idx, const params& ps ) {
        neighborstype* ns = (*rec->nsptrs)[idx];
        int sum = 0;
        if (ns->dim == 0)
            return -1;
        for (int i = 0; i < ns->dim; ++i)
            sum += ns->degrees[i];
        return (double)sum / (double)ns->dim;
    }

    avgdegreemeas(mrecords* recin) : meas(recin, "dm", "Graph's average degree") {}
};

class mindegreemeas : public meas {
public:
    double takemeas( const int idx, const params& ps ) {
        neighborstype* ns = (*rec->nsptrs)[idx];
        int min = ns->maxdegree;
        for (auto n = 0; n < ns->dim; ++n) {
            min = ns->degrees[n] < min ? ns->degrees[n] : min;
        }
        return min;
    }

    mindegreemeas(mrecords* recin) : meas(recin, "deltam", "Graph's minimum degree") {}

};

class maxdegreemeas : public meas {
public:

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

class girthmeas : public meas
{
public:
    girthmeas(mrecords* recin) : meas(recin, "girthm","Graph's girth") {}

    double takemeas( const int idx, const params& ps ) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
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
        // if (r)
        // {
            // for (int i = 0; i < n; ++i)
                // std::cout << path[i] << "... ";
            // std::cout << "\n";
        // }
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
        ps.clear();
        valms p1;
        p1.t = measuretype::mtdiscrete;
        p1.v.iv = 0;
        ps.push_back(p1);
        pssz = 1;

    }
    int takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];

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

};


class maxcliquemeas : public meas {
public:

    maxcliquemeas(mrecords* recin) : meas(recin,"cliquem","Graph's largest clique") {}
    double takemeas( const int idx, const params& ps ) override {
        std::vector<kncrit*> kns {};
        neighborstype* ns = (*rec->nsptrs)[idx];
        for (int n = 2; n <= ns->dim; ++n) {
            auto kn = new kncrit(rec,n);
            kns.push_back(kn);
            if (!(kn->takemeas(idx,ps))) {
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


};





class connectedmeas : public meas
{
public:
    connectedmeas(mrecords* recin) : meas(recin, "connm", "Connected components") {}

    double takemeas( const int idx, const params& ps ) override {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];

        int breaksize = -1; // by default wait until all connected components are counted
        if (ps.size() > 0)
            breaksize = ps[0].v.iv;

        return connectedcount( g, ns, breaksize );
    }
};

class connectedcrit : public crit {
public:

    connectedcrit( mrecords* recin ) : crit( recin, "connc", "Graph connected components less than criterion")
    {
        // the parameter is "less than" cutoff
        ps.clear();
        pssz = 1;
        valms p1;
        p1.t = measuretype::mtdiscrete;
        ps.push_back(p1);
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

    connected1crit( mrecords* recin ) : crit( recin, "conn1c", "Graph 1-connected ")
    {
        ps.clear();
        pssz = 0;
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

    double takemeas( const int idx, const params& ps ) override {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        int breaksize = -1; // by default wait until finding the actual (minimal) radius
        if (ps.size() > 0)
            breaksize = ps[0].v.iv;

        int dim = g->dim;
        if (dim <= 0)
            return 0;


        int* distances = (int*)malloc(dim*dim*sizeof(int));
        int* dextremes = (int*)malloc(dim*sizeof(int));
        //std::vector<vertextype> nextvs {};

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
                                //nextvs.push_back(newv);
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

            /*
            int nextvidx = (int)((distance - 1) / 2);
            nextv = nextvs[nextvidx];
            int startingpoint = nextvidx > 0 ? nextvidx-1 : distance;
            while ((dextremes[nextv] > 0) && (nextvidx+1 != startingpoint)) {
                nextvidx++;
                if (nextvidx >= distance)
                    nextvidx = 0;
                nextv = nextvs[nextvidx];
            }
            if (nextvidx == startingpoint)
                v = -1;
            else {
                distance = 0;
                v = nextv;
                nextvs.clear();
            }*/
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

};

class radiuscrit : public crit
{
public:

    radiuscrit( mrecords* recin ) : crit( recin, "radiusc", "Radius less than criterion" )
    {
        // the parameter is "less than" cutoff
        ps.clear();
        pssz = 1;
        valms p1;
        p1.t = measuretype::mtdiscrete;
        ps.push_back(p1);
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


    // bool takemeas(const int idx, const params& ps) override
    // {
        // auto resf = rm->takemeas(g,ns);
        // std::cout << "resf == " << resf << "\n";
        // if (ps.size()>0 && is_number(ps[0]))
            // return ((resf >= 0) && (resf < stoi(ps[0])));
        // return (resf >= 0);
    // }

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
        ps.clear();
        pssz = 1;
        valms p1;
        p1.t = measuretype::mtdiscrete;
        ps.push_back(p1);
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

    double takemeas( const int idx, const params& ps ) override {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
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

            /*
            int nextvidx = (int)((distance - 1) / 2);
            nextv = nextvs[nextvidx];
            int startingpoint = nextvidx > 0 ? nextvidx-1 : distance;
            while ((dextremes[nextv] > 0) && (nextvidx+1 != startingpoint)) {
                nextvidx++;
                if (nextvidx >= distance)
                    nextvidx = 0;
                nextv = nextvs[nextvidx];
            }
            if (nextvidx == startingpoint)
                v = -1;
            else {
                distance = 0;
                v = nextv;
                nextvs.clear();
            }*/
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

};


class diametercrit : public crit
{
public:

    diametercrit( mrecords* recin ) : crit( recin, "diamc", "Diameter greater than criterion")
    {
        // the parameter is "greater than" cutoff
        ps.clear();
        pssz = 1;
        valms p1;
        p1.t = measuretype::mtdiscrete;
        ps.push_back(p1);
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


template<typename T> crit* critfactory(mrecords* recin)
{
    return new T(recin);
}

template<typename T> meas* measfactory(mrecords* recin)
{
    return new T(recin);
}

template<typename T> tally* tallyfactory(mrecords* recin)
{
    return new T(recin);
}

template<typename T> set* setfactory(mrecords* recin)
{
    return new T(recin);
}




class embedstally : public tally {
protected:

public:
    graphtype* flagg;
    neighbors* flagns;
    FP* fp;
    embedstally( mrecords* recin , neighbors* flagnsin,FP* fpin)
        : tally(recin , "embedst","embeds type tally"),
        flagg{flagnsin->g},flagns{flagnsin},fp{fpin} {}
    int takemeas( const int idx ) override {
        return (embedscount(flagns, fp, (*this->rec->nsptrs)[idx]));
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
// Diestel, Grath Theory, p. 6-7

    public:

    kconnectedcrit( mrecords* recin ) : crit( recin, "kconnc", "Graph k-connected ")
    {
        ps.clear();
        valms p1;
        p1.t = mtdiscrete;
        p1.v.iv = 1;
        ps.push_back(p1);
        pssz = 1;
    }

    bool takemeas(const int idx, const params& ps) override
    {
        int k = 0;
        if (ps.size() > 0 && ps[0].t == mtdiscrete)
            k = ps[0].v.iv;

        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];

        return kconnectedfn( g, ns, k);
    }
};


class kappatally : public tally {
// Diestel, Graph Theory, page 7

public:

    kappatally( mrecords* recin ) : tally( recin, "kappat", "kappa: max k-connectedness")
    {
        ps.clear();
        pssz = 0;
    }

    int takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];

        int k = g->dim;
        bool res = false;
        while (!res && k >= 0)
            res = kconnectedfn( g, ns, k-- );

        return k+1;
    }




};



class vdtally : public tally {

public:

    vdtally( mrecords* recin ) : tally( recin, "vdt", "vertex degree")
    {
        ps.clear();
        valms p1;
        p1.t = mtdiscrete;
        p1.v.iv = 0;
        ps.push_back(p1);
        pssz = 1;
    }

    int takemeas(const int idx, const params& ps) override
    {
        // graphtype* g = (*rec->gptrs)[idx];
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
        ps.clear();
        valms p1;
        p1.t = mtdiscrete;
        p1.v.iv = 0;
        ps.push_back(p1);
        ps.push_back(p1);
        pssz = 2;
    }

    bool takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        // neighborstype* ns = (*rec->nsptrs)[idx];
        vertextype v1 = 0;
        vertextype v2 = 0;
        if (ps.size() == 2) {
            v1 = ps[0].v.iv;
            v2 = ps[1].v.iv;
        }
        return g->adjacencymatrix[v1*g->dim + v2];
    }

};

class eadjcrit : public crit {

public:

    eadjcrit( mrecords* recin ) : crit( recin, "eadjc", "edges adjacent")
    {
        ps.clear();
        valms p1;
        p1.t = mtpair;
        ps.push_back(p1);
        ps.push_back(p1);
        pssz = 2;
    }

    bool takemeas(const int idx, const params& ps) override
    {
        if (ps.size() == 2) {
            bool imatch = ps[0].v.p->i == ps[1].v.p->i || ps[0].v.p->i == ps[1].v.p->j;
            bool jmatch = ps[0].v.p->j == ps[1].v.p->i || ps[0].v.p->j == ps[1].v.p->j;
            return (imatch || jmatch) && !(imatch && jmatch);
        }
        return false;
    }

};



class sizetally : public tally {

public:

    sizetally( mrecords* recin ) : tally( recin, "st", "set size tally")
    {
        ps.clear();
        valms p1;
        p1.t = mtset;
        ps.push_back(p1);
        pssz = 1;
    }

    int takemeas(const int idx, const params& ps) override
    {
        if (ps.size() == 1) {
            return ps[0].seti->getsize();
        }
        return 0;
    }

};

/*
class psizetally : public tally {

public:

    psizetally( mrecords* recin ) : tally( recin, "pst", "pair set size tally")
    {
        ps.clear();
        valms p1;
        p1.t = mtpairset;
        ps.push_back(p1);
        pssz = 1;
    }

    int takemeas(const int idx, const params& ps) override
    {
        if (ps.size() == 1) {
            return ps[0].seti->getsize();
        }
        return 0;
    }

};
*/

class pctally : public tally {

public:

    pctally( mrecords* recin ) : tally( recin, "pct", "vertices path connected tally")
    {
        ps.clear();
        valms p1;
        p1.t = mtdiscrete;
        ps.push_back(p1);
        ps.push_back(p1);
        pssz = 2;
    }

    int takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        //osadjacencymatrix(std::cout, g);
        if (ps.size() == 2) {
            // std::cout << ps[0].v.iv << " iv " << ps[1].v.iv << "\n";
            return pathsbetweentally(g,ns,ps[0].v.iv, ps[1].v.iv);
        }
        return 0;
    }

};

class cyclestally : public tally {

public:

    cyclestally( mrecords* recin ) : tally( recin, "cyclest", "cycles tally")
    {
        ps.clear();
        valms p1;
        p1.t = mtdiscrete;
        ps.push_back(p1);
        pssz = 1;
    }

    int takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        //osadjacencymatrix(std::cout, g);
        if (ps.size() == 1) {
            // std::cout << ps[0].v.iv << " iv " << ps[1].v.iv << "\n";
            return directedcyclestally(g,ns,ps[0].v.iv)/2; // undirected (cf. Diestel p 6-10) is half of directed
        }
        return 0;
    }

};



class ecrit : public crit
{
public:
    ecrit( mrecords* recin ) : crit( recin, "ec", "Edge holds in graph")
    {
        ps.clear();
        valms p1;
        p1.t = mtset;
        ps.push_back(p1);
        pssz = 1;
    }

    bool takemeas( const int idx, const params& ps) override
    {
        if (ps.size() != 1)
            return false;
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        auto itr = ps[0].seti->getitrpos();

        bool res = g->adjacencymatrix[g->dim * itr->getnext().v.iv + itr->getnext().v.iv];
        delete itr;
        return res;

        // return g->adjacencymatrix[g->dim*ps[0].v.p->i.v.iv + ps[0].v.p->j.v.iv];
    }
};

class idxtally : public tally
{
    public:

    idxtally( mrecords* recin ) : tally( recin, "idxt", "Index into an integer-comprised set")
    {
        ps.clear();
        valms p1;
        p1.t = mtset;
        ps.push_back(p1);
        p1.t = mtdiscrete;
        ps.push_back(p1);
        pssz = 2;
    }

    int takemeas( const int idx, const params& ps) override
    {
        if (ps.size() != 2)
        {
            std::cout << "Wrong number of parameters to idxt\n";
        }
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
        ps.clear();
        valms p1;
        p1.t = mtset;
        ps.push_back(p1);
        p1.t = mtdiscrete;
        ps.push_back(p1);
        pssz = 2;
    }

    setitr* takemeas( const int idx, const params& ps) override
    {
        if (ps.size() != 2)
        {
            std::cout << "Wrong number of parameters to idxs\n";
        }
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

/* class pairfirsttally : public tally
{
public:
    pairfirsttally( mrecords* recin ) : tally( recin, "pfirstt", "Pair first")
    {
        ps.clear();
        valms p1;
        p1.t = mtpair;
        ps.push_back(p1);
        pssz = 1;
    }

    int takemeas( const int idx, const params& ps) override
    {
        return ps[0].v.p->i.v.iv;
    }
};

class pairsecondtally : public tally
{
public:
    pairsecondtally( mrecords* recin ) : tally( recin, "psecondt", "Pair second")
    {
        ps.clear();
        valms p1;
        p1.t = mtpair;
        ps.push_back(p1);
        pssz = 1;
    }

    int takemeas( const int idx, const params& ps) override
    {
        return ps[0].v.p->j.v.iv;
    }
};
*/

class bipcrit : public crit
{
public:
    bipcrit( mrecords* recin ) : crit( recin, "bipc", "Bipartite")
    {
        ps.clear();
        valms p1;
        p1.t = mtset;
        ps.push_back(p1);
        ps.push_back(p1);
        pssz = 2;
    }

    bool takemeas( const int idx, const params& ps) override
    {
        if (ps.size() != 2)
            return false;
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        bool all = true;
        if (setitrbool* sl = dynamic_cast<setitrsubset*>(ps[0].seti)->itrbool)
        {
            if (setitrbool* sr = dynamic_cast<setitrsubset*>(ps[1].seti)->itrbool)
            {
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

                for (int i = 0; all && (i < sl->maxint); ++i)
                    for (int j = i+1; all && (j <= sl->maxint); ++j)
                        all = !sl->elts[i] || !sl->elts[j] || !(g->adjacencymatrix[i*g->dim + j]);
                for (int i = 0; all && (i < sr->maxint); ++i)
                    for (int j = i+1; all && (j <= sr->maxint); ++j)
                        all = !sr->elts[i] || !sr->elts[j] || !(g->adjacencymatrix[i*g->dim + j]);
                return all;
            }
        }
        std::cout << "Non-integer (non-vertex) set passed to bipcrit\n";
        return false;
    }
};

class Ntally : public tally {
// Diestel p. ??
public:

    Ntally( mrecords* recin ) : tally( recin, "Nt", "Neighbors of a vertex set")
    {
        ps.clear();
        valms p1;
        p1.t = mtset;
        ps.push_back(p1);
        pssz = 1;
    }

    int takemeas(const int idx, const params& ps) override
    {
        graphtype* g = (*rec->gptrs)[idx];
        neighborstype* ns = (*rec->nsptrs)[idx];
        if (ps.size() == 1) {
            if (setitrint* s = dynamic_cast<setitrint*>(ps[0].seti))
            {
                bool* S = s->elts;
                bool* N = (bool*)malloc(g->dim * sizeof(bool));
                memset(N,false,g->dim * sizeof(bool));
                int cnt = 0;
                for (int i = 0; i < s->maxint; ++i)
                    if (S[i])
                        for (int j = 0; j < ns->degrees[i]; ++j)
                        {
                            vertextype nbr = ns->neighborslist[i*g->dim + j];
                            cnt += (!N[nbr] && !S[nbr]) ? 1 : 0;
                            N[nbr] = true;
                        }
                delete N;
                return cnt;
            }
        }
        std::cout << "Wrong number of parameters or parameter types passed to Ntally\n";
        return 0;
    }

};
