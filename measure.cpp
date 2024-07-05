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





class measurenonzerocriterion : public abstractmemorymeasure<bool> {
public:
    std::string name = "_measurenonzerocriterion";
    abstractmemorymeasure<float>* am;

    virtual std::string shortname() {return "_mnzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return am->takemeasure(g,ns) != 0;
    }
    measurenonzerocriterion(abstractmemorymeasure<float>* amin)
        : abstractmemorymeasure<bool>(amin->name + " measure non zero"), am{amin} {}
};

class measurezerocriterion : public abstractmemorymeasure<bool> {
public:
    std::string name = "_measurezerocriterion";
    abstractmemorymeasure<float>* am;

    virtual std::string shortname() {return "_mzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return (abs(am->takemeasure(g,ns))) < 0.1;
    }
    measurezerocriterion(abstractmemorymeasure<float>* amin)
        : abstractmemorymeasure<bool>(amin->name + " measure zero"), am{amin} {}
};





class legacytreecriterion : public measurezerocriterion{
protected:
public:
    std::string name = "legacytreecriterion";

    virtual std::string shortname() {return "ltc";}

    legacytreecriterion() : measurezerocriterion(new girthmeasure()) {}
    ~legacytreecriterion() {
        delete am;
    }
};


