//
// Created by peterglenn on 6/29/24.
//


#include "graphs.h"
#include "asymp.h"
// how many cycles to compute in advance, e.g. in girthmeasure
#define GRAPH_PRECOMPUTECYCLESCNT 15



class boolmeasure : public abstractmemorymeasure<float> {
public:
    virtual std::string shortname() override {return "bm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) override {
        return (float)true;
    }


    boolmeasure() : abstractmemorymeasure<float>( "Graph's pass/fail of criterion") {}
};



class dimmeasure : public abstractmemorymeasure<float> {
public:
    virtual std::string shortname() {return "dm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        return g->dim;
    }


    dimmeasure() : abstractmemorymeasure<float>( "Graph's dimension") {}
};

class edgecntmeasure : public abstractmemorymeasure<float> {
public:
    virtual std::string shortname() {return "ecm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        return edgecnt(g);
    }

    edgecntmeasure() : abstractmemorymeasure<float>( "Graph's edge count") {}
};

class avgdegreemeasure : public abstractmemorymeasure<float> {
public:
    virtual std::string shortname() {return "adm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        int sum = 0;
        if (ns->dim == 0)
            return -1;
        for (int i = 0; i < ns->dim; ++i)
            sum += ns->degrees[i];
        return sum / ns->dim;
    }

    avgdegreemeasure() : abstractmemorymeasure<float>( "Graph's average degree") {}
};

class mindegreemeasure : public abstractmemorymeasure<float> {
public:
    virtual std::string shortname() {return "mdm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        int min = ns->maxdegree;
        for (auto n = 0; n < ns->dim; ++n) {
            min = ns->degrees[n] < min ? ns->degrees[n] : min;
        }
        return min;
    }

    mindegreemeasure() : abstractmemorymeasure<float>( "Graph's minimum degree") {}

};

class maxdegreemeasure : public abstractmemorymeasure<float> {
public:
    virtual std::string shortname() {return "Mdm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        return ns->maxdegree;
    }


    maxdegreemeasure() : abstractmemorymeasure<float>( "Graph's maximum degree") {}

};

class girthmeasure : public abstractmemorymeasure<float> {
    // "girth" is shortest cycle (Diestel p. 8)
protected:
    std::vector<graphtype*> cyclegraphs {};
    std::vector<neighbors*> cycleneighbors {};
    std::vector<FP*> cyclefps {};
public:
    virtual std::string shortname() {return "gm";}

    girthmeasure() : abstractmemorymeasure<float>("Graph's girth") {
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
    float takemeasure( const graphtype* g, const neighbors* ns ) {
        int n = 3;
        bool embedsbool = false;
        if (g->dim < cyclegraphs.size()) {
            while (!embedsbool && n <= g->dim) {
                embedsbool |= embeds(cycleneighbors[n], cyclefps[n], ns, 1);
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
                embedsbool |= embeds(cycleneighbors[n], cyclefps[n], ns,1);
                ++n;
            }
            if (embedsbool)
                return n-1;
        }
        if (!embedsbool) {
            return 0;
        }

    }

};


class maxcliquemeasure : public abstractmemorymeasure<float> {
public:
    virtual std::string shortname() {return "cm";}

    maxcliquemeasure() : abstractmemorymeasure<float>("Graph's largest clique") {}
    float takemeasure( const graphtype* g, const neighbors* ns ) {
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
class measurenonzerocriterion : public abstractmemorymeasure<bool> {
public:
    std::string name = "_measurenonzerocriterion";
    abstractmemorymeasure<float>* am;

    virtual std::string shortname() {return "_mnzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return am->takemeasure(g,ns) != 0;
    }
    measurenonzerocriterion(abstractmemorymeasure<T>* amin)
        : abstractmemorymeasure<bool>(amin->name + " measure non-zero"), am{amin} {}
};

template<typename T>
class measurezerocriterion : public abstractmemorymeasure<bool> {
public:
    std::string name = "_measurezerocriterion";
    abstractmemorymeasure<T>* am;

    virtual std::string shortname() {return "_mzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return (abs(am->takemeasure(g,ns))) < 0.1;
    }
    measurezerocriterion(abstractmemorymeasure<T>* amin)
        : abstractmemorymeasure<bool>(amin->name + " measure zero"), am{amin} {}
};

template<typename T>
class measurenonzeroparameterizedcriterion : public abstractmemoryparameterizedmeasure<bool>
{
public:
    std::string name = "_measurenonzeroparameterizedcriterion";
    abstractmemoryparameterizedmeasure<T>* apm;
    virtual std::string shortname() override {return "_mnzpc";}

    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        return apm->takemeasure(g,ns) != 0;
    }
    measurenonzeroparameterizedcriterion(abstractmemoryparameterizedmeasure<T>* apmin)
        : abstractmemoryparameterizedmeasure<bool>(apmin->name + " measure non-zero"), apm{apmin} {}
};

template<typename T>
class measuregreaterthanparameterizedcriterion : public abstractmemoryparameterizedmeasure<bool>
{
public:
    std::string name = "_measuregreaterthanparameterizedcriterion";
    abstractmemoryparameterizedmeasure<T>* apm;
    virtual std::string shortname() override {return "_mgtpc";}

    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        float limit = 0;
        if (ps.size() > 0)
            limit = stof(ps[0]);
        return apm->takemeasure(g,ns) > limit;
    }

    void setparams(const std::vector<std::string> pin) override
    {
        abstractmemoryparameterizedmeasure::setparams(pin);

    }
    measuregreaterthanparameterizedcriterion(abstractmemoryparameterizedmeasure<T>* apmin)
        : abstractmemoryparameterizedmeasure<bool>(apmin->name + " measure greater than"), apm{apmin} {}
};

template<typename T>
class measurelessthanparameterizedcriterion : public abstractmemoryparameterizedmeasure<bool>
{
public:
    std::string name = "_measurelessthanparameterizedcriterion";
    abstractmemoryparameterizedmeasure<T>* apm;
    virtual std::string shortname() override {return "_mltpc";}

    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        float limit = 0;
        if (ps.size() > 0)
            limit = stof(ps[0]);
        return apm->takemeasure(g,ns) < limit;
    }

    void setparams(const std::vector<std::string> pin) override
    {
        abstractmemoryparameterizedmeasure::setparams(pin);
        apm->setparams(pin);
    }
    measurelessthanparameterizedcriterion(abstractmemoryparameterizedmeasure<T>* apmin)
        : abstractmemoryparameterizedmeasure<bool>(apmin->name + " measure less than"), apm{apmin} {}
};






class legacytreecriterion : public measurezerocriterion<float>{
protected:
public:
    std::string name = "legacytreecriterion";

    virtual std::string shortname() {return "ltc";}

    legacytreecriterion() : measurezerocriterion<float>(new girthmeasure()) {}
    ~legacytreecriterion() {
        delete am;
    }
};


class connectedmeasure : public  abstractmemoryparameterizedmeasure<float>
{
public:
    virtual std::string shortname() override {return "cnm";}

    connectedmeasure() : abstractmemoryparameterizedmeasure<float>("Connected components") {}

    float takemeasure( const graphtype* g, const neighbors* ns ) override
    {
        int breaksize = -1; // by default wait until all connected components are counted
        if (ps.size() > 0 && is_number(ps[0]))
            breaksize = stoi(ps[0]);

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
                    allvisited &= (visited[firstunvisited] != -1);
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

class connectedcriterion : public measurelessthanparameterizedcriterion<float>
{
public:
    std::string shortname() override {return "cc";}

    connectedcriterion() : measurelessthanparameterizedcriterion<float>(new connectedmeasure())
    {
        setparams({"2"});
    }
    ~connectedcriterion()
    {
        delete apm;
    }
};



class radiusmeasure : public  abstractmemoryparameterizedmeasure<float>
{
public:
    virtual std::string shortname() override {return "rm";}

    radiusmeasure() : abstractmemoryparameterizedmeasure<float>("Graph radius") {}

    float takemeasure( const graphtype* g, const neighbors* ns ) override {
        int breaksize = -1; // by default wait until finding the actual (minimal) radius
        if (ps.size() > 0 && is_number(ps[0]))
            breaksize = stoi(ps[0]);

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
                return -1; // the graph isn't connected
            }
            if ((breaksize >= 0) && (distance <= breaksize)) {
                int res = distance;
                for (int i = 0; i < dim; ++i)
                    if (distances[v*dim + i] < 0)
                        res = -1;
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

class radiuscriterion : public abstractmemoryparameterizedmeasure<bool>
{
public:
    radiusmeasure* rm;
    std::string shortname() override {return "rltc";}

    radiuscriterion() : abstractmemoryparameterizedmeasure<bool>("Radius less than")
    {
        rm = new radiusmeasure;
        rm->setparams({"-1"});
    }
    void setparams(const std::vector<std::string> pin) override
    {
        abstractmemoryparameterizedmeasure::setparams(pin);
        rm->setparams(pin);
    }
    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        auto resf = rm->takemeasure(g,ns);
        //std::cout << "resf == " << resf << "\n";
        if (ps.size()>0 && is_number(ps[0]))
            return ((resf >= 0) && (resf <= stoi(ps[0])));
        return (resf >= 0);
    }

    ~radiuscriterion()
    {
        delete rm;
    }
};

inline int recursecircumferencemeasure( int* path, int pathsize, const graphtype* g, const neighbors* ns, const int breaksize ) {
    if (pathsize == 0) {
        int respl = 0;
        for (int n = 0; n < g->dim; ++n) {
            int newpath[] = {n};
            int newpl = recursecircumferencemeasure(newpath, 1, g, ns,breaksize);
            respl = (respl < newpl ? newpl : respl);
        }
        return respl;
    }

    int respl = 0;
    for (int i = 0; i < ns->degrees[path[pathsize-1]]; ++i ) {
        int newpl = 0;
        int newpath[pathsize+1];
        int newv = ns->neighborslist[path[pathsize-1]*g->dim + i];
        bool found = false;
        int j = 0;
        for (; !found && (j < pathsize); ++j) {
            newpath[j] = path[j];
            found = (path[j] == newv);
        }
        if (!found) {
            newpath[pathsize] = newv;
            newpl = recursecircumferencemeasure(newpath,pathsize+1, g, ns,breaksize);
        } else
            newpl = pathsize-j + 1;
        respl = (respl < newpl ? newpl : respl);
    }
    respl = respl == 2 ? 0 : respl;
    if (breaksize >= 3 && respl >= breaksize)
        return respl;

    return respl;
}

class circumferencemeasure: public abstractmemoryparameterizedmeasure<float> {
public:

    virtual std::string shortname() override {return "circm";}

    circumferencemeasure() : abstractmemoryparameterizedmeasure<float>("Graph circumference") {}



    float takemeasure( const graphtype* g, const neighbors* ns ) {
        int breaksize = -1; // by default compute the largest circumference possible
        if (ps.size() > 0 && is_number(ps[0]))
            breaksize = stoi(ps[0]);

        int dim = g->dim;
        if (dim <= 0)
            return 0;

        return recursecircumferencemeasure(nullptr,0,g,ns,breaksize);
    }
};


class circumferencecriterion : public abstractmemoryparameterizedmeasure<bool>
{
public:
    circumferencemeasure* cm;
    std::string shortname() override {return "circc";}

    circumferencecriterion() : abstractmemoryparameterizedmeasure<bool>("Circumference greater than")
    {
        cm = new circumferencemeasure;
        cm->setparams({"-1"});
    }
    void setparams(const std::vector<std::string> pin) override
    {
        abstractmemoryparameterizedmeasure::setparams(pin);
        cm->setparams(pin);
    }
    bool takemeasure(const graphtype* g, const neighbors* ns) override
    {
        auto resf = cm->takemeasure(g,ns);
        //std::cout << "resf == " << resf << "\n";
        if (ps.size()>0 && is_number(ps[0]))
            return (resf >= stoi(ps[0]));
        return (resf >= 3);
    }

    ~circumferencecriterion()
    {
        delete cm;
    }
};

