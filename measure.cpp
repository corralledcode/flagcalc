//
// Created by peterglenn on 6/29/24.
//


#include "graphs.h"
#include "asymp.h"
// how many cycles to compute in advance, e.g. in girthmeasure
#define GRAPH_PRECOMPUTECYCLESCNT 15




class boolmeasure : public measure {
public:
    virtual std::string shortname() override {return "truem";}

    double takemeasure( const graphtype* g, const neighbors* ns ) override {
        return (double)true;
    }


    boolmeasure() : measure( "Graph's pass/fail of criterion") {}
};



class dimmeasure : public measure {
public:
    virtual std::string shortname() {return "dimm";}

    double takemeasure( const graphtype* g, const neighbors* ns ) {
        return g->dim;
    }


    dimmeasure() : measure( "Graph's dimension") {}
};

class edgecntmeasure : public measure {
public:
    virtual std::string shortname() {return "edgecm";}

    double takemeasure( const graphtype* g, const neighbors* ns ) {
        return edgecnt(g);
    }

    edgecntmeasure() : measure( "Graph's edge count") {}
};

class avgdegreemeasure : public measure {
public:
    virtual std::string shortname() {return "dm";}

    double takemeasure( const graphtype* g, const neighbors* ns ) {
        int sum = 0;
        if (ns->dim == 0)
            return -1;
        for (int i = 0; i < ns->dim; ++i)
            sum += ns->degrees[i];
        return sum / ns->dim;
    }

    avgdegreemeasure() : measure( "Graph's average degree") {}
};

class mindegreemeasure : public measure {
public:
    virtual std::string shortname() {return "deltam";}

    double takemeasure( const graphtype* g, const neighbors* ns ) {
        int min = ns->maxdegree;
        for (auto n = 0; n < ns->dim; ++n) {
            min = ns->degrees[n] < min ? ns->degrees[n] : min;
        }
        return min;
    }

    mindegreemeasure() : measure( "Graph's minimum degree") {}

};

class maxdegreemeasure : public measure {
public:
    virtual std::string shortname() {return "Deltam";}

    double takemeasure( const graphtype* g, const neighbors* ns ) {
        return ns->maxdegree;
    }


    maxdegreemeasure() : measure( "Graph's maximum degree") {}

};

class girthmeasure : public measure {
    // "girth" is shortest cycle (Diestel p. 8)
protected:
    std::vector<graphtype*> cyclegraphs {};
    std::vector<neighbors*> cycleneighbors {};
    std::vector<FP*> cyclefps {};
public:
    virtual std::string shortname() {return "girthm";}

    girthmeasure() : measure("Graph's girth") {
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
    ~girthmeasure() {
        for (int i = 0; i < cyclefps.size(); ++i ) {
            freefps(cyclefps[i],i);
            free (cyclefps[i]);
        }
        for (int i = 0; i < cyclegraphs.size(); ++i) {
            delete cyclegraphs[i];
            delete cycleneighbors[i];
        }
    }
    double takemeasure( const graphtype* g, const neighbors* ns ) {
        int n = 3;
        bool embedsbool = false;
        if (g->dim < cyclegraphs.size()) {
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


class maxcliquemeasure : public measure {
public:
    virtual std::string shortname() {return "cliquem";}

    maxcliquemeasure() : measure("Graph's largest clique") {}
    double takemeasure( const graphtype* g, const neighbors* ns ) {
        std::vector<kncriterion*> kns {};
        for (int n = 2; n <= ns->dim; ++n) {
            auto kn = new kncriterion(n);
            kns.push_back(kn);
            if (!(kn->takemeasure(g,ns))) {
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





template<typename T>
class measurenonzerocriterion : public criterion {
public:
    std::string name = "_measurenonzerocriterion";
    abstractmemorymeasure<T>* am;

    virtual std::string shortname() {return "_mnzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return am->takemeasure(g,ns) != 0;
    }
    measurenonzerocriterion(abstractmemorymeasure<T>* amin)
        : criterion(amin->name + " measure non-zero"), am{amin} {}
};

template<typename T>
class measurezerocriterion : public criterion {
public:
    std::string name = "_measurezerocriterion";
    abstractmemorymeasure<T>* am;

    virtual std::string shortname() {return "_mzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return (abs(am->takemeasure(g,ns))) < 0.1;
    }
    measurezerocriterion(abstractmemorymeasure<T>* amin)
        : criterion(amin->name + " measure zero"), am{amin} {}
};


template<typename T>
class measuregreaterthancriterion : public criterion
{
public:
    std::string name = "_measuregreaterthancriterion";
    abstractmemorymeasure<T>* am;
    virtual std::string shortname() override {return "_mgtpc";}

    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        double limit = 0;
        if (ps.size() > 0)
            limit = ps[0];
        return am->takemeasure(g,ns) > limit;
    }

    void setparams(const std::vector<double>& pin) override
    {
        criterion::setparams(pin);
        am->setparams(pin);
    }
    measuregreaterthancriterion(abstractmemorymeasure<T>* amin)
        : criterion(amin->name + " measure greater than"), am{amin} {}
};

template<typename T>
class measurelessthancriterion : public criterion
{
public:
    std::string name = "_measurelessthanparameterizedcriterion";
    abstractmemorymeasure<T>* am;
    virtual std::string shortname() override {return "_mltpc";}

    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        double limit = 0;
        if (ps.size() > 0)
            limit = ps[0];
        return am->takemeasure(g,ns) < limit;
    }

    void setparams(const std::vector<double>& pin) override
    {
        criterion::setparams(pin);
        am->setparams(pin);
    }
    measurelessthancriterion(abstractmemorymeasure<T>* amin)
        : criterion(amin->name + " measure less than"), am{amin} {}
};






class legacyforestcriterion : public measurezerocriterion<double>{
protected:
public:
    std::string name = "legacyforestcriterion";

    virtual std::string shortname() {return "ltreec";}

    legacyforestcriterion() : measurezerocriterion<double>(new girthmeasure()) {}
    ~legacyforestcriterion() {
        delete am;
    }
};


class connectedmeasure : public measure
{
public:
    virtual std::string shortname() override {return "connm";}

    connectedmeasure() : measure("Connected components") {}

    double takemeasure( const graphtype* g, const neighbors* ns ) override
    {
        int breaksize = -1; // by default wait until all connected components are counted
        if (ps.size() > 0)
            breaksize = ps[0];

        int dim = g->dim;
        if (dim <= 0)
            return 0;


        int* visited = (int*)malloc(dim*sizeof(int));

        for (int i = 0; i < dim; ++i)
        {
            visited[i] = -1;
        }

        visited[0] = 0;
        int res = 1;
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
                    return res;
                }
                res++;
                if (breaksize >=0 && res >= breaksize)
                {
                    free (visited);
                    return res;
                }
                visited[firstunvisited-1] = firstunvisited-1;
                changed = true;
            }

        }

    }


};

class connectedcriterion : public measurelessthancriterion<double>
{
public:
    std::string shortname() override {return "connc";}

    connectedcriterion() : measurelessthancriterion<double>(new connectedmeasure())
    {
        setparams({2});
    }
    ~connectedcriterion()
    {
        delete am;
    }
};



class radiusmeasure : public measure
{
public:
    virtual std::string shortname() override {return "radiusm";}

    radiusmeasure() : measure("Graph radius") {}

    double takemeasure( const graphtype* g, const neighbors* ns ) override {
        int breaksize = -1; // by default wait until finding the actual (minimal) radius
        if (ps.size() > 0)
            breaksize = ps[0];

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

class radiuscriterion : public measurelessthancriterion<double>
{
public:
    std::string shortname() override {return "radiusc";}

    radiuscriterion() : measurelessthancriterion<double>(new radiusmeasure)
    {
        setparams({-1});
    }

    // bool takemeasure(const graphtype* g, const neighbors* ns) override
    // {
        // auto resf = rm->takemeasure(g,ns);
        // std::cout << "resf == " << resf << "\n";
        // if (ps.size()>0 && is_number(ps[0]))
            // return ((resf >= 0) && (resf < stoi(ps[0])));
        // return (resf >= 0);
    // }

    ~radiuscriterion()
    {
        delete am;
    }
};

inline int recursecircumferencemeasure( int* path, int pathsize, bool* visited, const graphtype* g, const neighbors* ns, const int breaksize ) {
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
                int newpl = recursecircumferencemeasure(newpath,1,visited,g,ns,breaksize);
                if (breaksize > 2 && newpl > breaksize)
                    return newpl;
                //visited[n-1] = false;
                //for (int n = 0; n < g->dim; ++n) {
                //    int newpath[] = {n};
                //    int newpl = recursecircumferencemeasure(newpath, 1, g, ns,breaksize);
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
            newpl = recursecircumferencemeasure(newpath,pathsize+1,visited, g, ns,breaksize);
            //visited[newv] = false;
        } else {
            newpl = pathsize-j + 1;
        }
        if (breaksize > 2 && newpl > breaksize)
            return newpl;
        respl = (respl < newpl ? newpl : respl);
    }
    delete newpath;
    respl = respl == 2 ? 0 : respl;
    return respl;
}

inline int legacyrecursecircumferencemeasure( int* path, int pathsize, const graphtype* g, const neighbors* ns, const int breaksize ) {

    if (pathsize == 0) {
        int respl = 0;
        for (int n = 0; n < g->dim; ++n) {
            int newpath[] = {n};
            int newpl = legacyrecursecircumferencemeasure(newpath, 1, g, ns,breaksize);
            respl = (respl < newpl ? newpl : respl);
        }
        return respl;
    }

    int respl = 0;
    for (int i = 0; i < ns->degrees[path[pathsize-1]]; ++i ) {
        int newpl = 0;
        int* newpath = new int[pathsize+1];
        int newv = ns->neighborslist[path[pathsize-1]*g->dim + i];
        bool found = false;
        int j = 0;
        for (; !found && (j < pathsize); ++j) {
            newpath[j] = path[j];
            found = (path[j] == newv);
        }
        if (!found) {
            newpath[pathsize] = newv;
            newpl = legacyrecursecircumferencemeasure(newpath,pathsize+1, g, ns,breaksize);
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

class circumferencemeasure: public measure {
public:

    virtual std::string shortname() override {return "circm";}

    circumferencemeasure() : measure("Graph circumference") {}

    double takemeasure( const graphtype* g, const neighbors* ns ) {
        int breaksize = -1; // by default compute the largest circumference possible
        if (ps.size() > 0)
            breaksize = ps[0];

        int dim = g->dim;
        if (dim <= 0)
            return 0;

        bool* visited = new bool[g->dim];
        for (int i = 0; i < g->dim; ++i)
            visited[i] = false;
        auto res = recursecircumferencemeasure(nullptr,0,visited,g,ns,breaksize);
        delete visited;
        return res;
    }
};

class legacycircumferencemeasure: public measure {
public:

    virtual std::string shortname() override {return "lcircm";}

    legacycircumferencemeasure() : measure("Legacy Graph circumference") {}



    double takemeasure( const graphtype* g, const neighbors* ns ) {
        int breaksize = -1; // by default compute the largest circumference possible
        if (ps.size() > 0)
            breaksize = ps[0];

        int dim = g->dim;
        if (dim <= 0)
            return 0;

        return legacyrecursecircumferencemeasure(nullptr,0,g,ns,breaksize);
    }
};


class circumferencecriterion : public measuregreaterthancriterion<double>
{
public:
    std::string shortname() override {return "circc";}

    circumferencecriterion() : measuregreaterthancriterion<double>(new circumferencemeasure)
    {
        setparams({-1});
    }

    // return res > 3

    ~circumferencecriterion()
    {
        delete am;
    }
};


class diametermeasure : public measure
{
public:
    virtual std::string shortname() override {return "diamm";}

    diametermeasure() : measure("Graph diameter") {}

    double takemeasure( const graphtype* g, const neighbors* ns ) override {
        int breaksize = -1; // by default wait until finding the actual (maximal) diameter
        if (ps.size() > 0)
            breaksize = ps[0];

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

class diametercriterion : public measuregreaterthancriterion<double>
{
public:
    std::string shortname() override {return "diamc";}

    diametercriterion() : measuregreaterthancriterion<double>(new diametermeasure)
    {
        am->setparams({-1});
    }

    ~diametercriterion()
    {
        delete am;
    }
};


/*
class formulameasure : public measure {
protected:
    //std::vector<abstractmemorymeasure<bool>*> cs;
    formulaclass* fc;
    std::vector<double*>* variables {};
    const std::map<int,int> litnumps;
    evalidxedformula* ef;

public:

    std::string shortname() override {return "msFORMULA";}
    double takemeasureidxed( const int idx ) override {
        if (!computed[idx]) {
            std::vector<double> tmpres {};
            tmpres.resize(variables->size());
            for (int i = 0; i < variables->size(); ++i )
                tmpres[i] = (*variables)[i][idx];
            ef->idx = idx;
            ef->literals = tmpres;
            res[idx] = ef->eval(*fc);  // check for speed cost
            computed[idx] = true;
        }
        return res[idx]; //abstractmemorymeasure::takemeasureidxed(idx);
    }

    formulameasure( std::vector<iteration*>* iterin, const std::map<int,int>& litnumpsin,
        std::vector<double*>* variablesin, const int szin, std::string formula,
        std::string stringin )
        : measure(stringin == "" ? "logical sentence of several criteria" : stringin),
            variables{variablesin}, ef{new evalidxedformula(iterin)}, litnumps{litnumpsin}, fc{parseformula(formula,litnumpsin)} {
        setsize(szin);
    }

    ~formulameasure() {
        delete fc;
        delete ef;
    }
};
*/

/* class equationcriteria : public criterion {
protected:
    //std::vector<criterion*> cs;
    formulaclass* lhs;
    formulaclass* rhs;
    int eqtype = 0; // 0: equal; 1: >= ; -1: <=; 2: >; -2: <
    std::vector<double*> variables {};
    const int sz2;

public:

    std::string shortname() override {return "crEQUATION";}
    bool takemeasureidxed( const int idx ) override {
        if (!computed[idx]) {
            std::vector<double> tmpres {};
            tmpres.resize(variables.size());
            for (int i = 0; i < variables.size(); ++i )
                tmpres[i] = variables[i][idx];
            switch (eqtype) {
                case 0: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) == evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case 1: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) >= evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case 2: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) >  evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case -1: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) <= evalformula(*rhs,tmpres,nullptr);
                    break;
                }
                case -2: {
                    res[idx] = evalformula(*lhs,tmpres,nullptr) < evalformula(*rhs,tmpres,nullptr);
                    break;
                }
            }
            computed[idx] = true;
        }
        return res[idx]; //abstractmemorymeasure::takemeasureidxed(idx);
    }

    equationcriteria( std::vector<double*> variablesin, const int szin, std::string equationin, std::string stringin )
        : criterion(stringin == "" ? "equation" : stringin),
            variables{variablesin}, sz2{szin} {
        setsize(sz2);
        std::string lhstr;
        std::string rhstr;
        parseequation(&equationin,&lhstr,&rhstr,&eqtype);
        lhs = parseformula(lhstr,nullptr);
        rhs = parseformula(rhstr,nullptr);
    }

    ~equationcriteria() {
        delete lhs;
        delete rhs;
    }

};
*/

template<typename T> measure* measurefactory(void)
{
    return new T;
}

