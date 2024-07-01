//
// Created by peterglenn on 6/29/24.
//


#include "graphs.h"
#include "asymp.h"
// how many cycles to compute in advance, e.g. in girthmeasure
#define GRAPH_PRECOMPUTECYCLESCNT 20



class boolmeasure : public abstractmeasure<float> {
public:
    virtual std::string shortname() override {return "bm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) override {
        return (float)true;
    }

    boolmeasure() : abstractmeasure<float>( "Graph's pass/fail of criterion") {}
};



class dimmeasure : public abstractmeasure<float> {
public:
    virtual std::string shortname() {return "dm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        return g->dim;
    }

    dimmeasure() : abstractmeasure<float>( "Graph's dimension") {}
};

class edgecntmeasure : public abstractmeasure<float> {
public:
    virtual std::string shortname() {return "ecm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        return edgecnt(g);
    }
    edgecntmeasure() : abstractmeasure<float>( "Graph's edge count") {}
};

class avgdegreemeasure : public abstractmeasure<float> {
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

    avgdegreemeasure() : abstractmeasure<float>( "Graph's average degree") {}
};

class mindegreemeasure : public abstractmeasure<float> {
public:
    virtual std::string shortname() {return "mdm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        int min = ns->maxdegree;
        for (auto n = 0; n < ns->dim; ++n) {
            min = ns->degrees[n] < min ? ns->degrees[n] : min;
        }
        return min;
    }
    mindegreemeasure() : abstractmeasure<float>( "Graph's minimum degree") {}

};

class maxdegreemeasure : public abstractmeasure<float> {
public:
    virtual std::string shortname() {return "Mdm";}

    float takemeasure( const graphtype* g, const neighbors* ns ) {
        return ns->maxdegree;
    }

    maxdegreemeasure() : abstractmeasure<float>( "Graph's maximum degree") {}

};

class girthmeasure : public abstractmeasure<float> {
    // "girth" is shortest cycle (Diestel p. 8)
protected:
    std::vector<graphtype*> cyclegraphs {};
    std::vector<neighbors*> cycleneighbors {};
    std::vector<FP*> cyclefps {};
public:
    virtual std::string shortname() {return "gm";}

    girthmeasure() : abstractmeasure<float>("Graph's girth") {
        for (int n = 0; n < GRAPH_PRECOMPUTECYCLESCNT; ++n) {
            graphtype* cycleg = new graphtype(n);
            *cycleg = cyclegraph(n);
            neighbors* cyclens = new neighbors(cycleg);
            cyclegraphs.push_back(cycleg);
            cycleneighbors.push_back(cyclens);
            FP* cyclefp = (FP*)malloc(n*sizeof(FP));
            for (int i = 0; i < n; ++i) {
                cyclefp[i].ns = nullptr;
                cyclefp[i].nscnt = 0;
                cyclefp[i].v = i;
                cyclefp[i].parent = nullptr;
                cyclefp[i].invert = cyclens->degrees[i] >= int(n+1/2); // = 2
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
        if (g->dim < GRAPH_PRECOMPUTECYCLESCNT) {
            while (!embedsbool && n < g->dim) {
                embedsbool |= embeds(cycleneighbors[n], cyclefps[n], ns);
                ++n;
            }
            if (embedsbool)
                return n-1;
        }

        if (g->dim >= GRAPH_PRECOMPUTECYCLESCNT) {
            while (!embedsbool && n < g->dim) {


                graphtype cycleg = cyclegraph(n);
                neighbors* cyclens = new neighbors(&cycleg);
                FP* cyclefp = (FP*)malloc(n*sizeof(FP));
                for (int i = 0; i < n; ++i) {
                    cyclefp[i].ns = nullptr;
                    cyclefp[i].nscnt = 0;
                    cyclefp[i].v = i;
                    cyclefp[i].parent = nullptr;
                    cyclefp[i].invert = cyclens->degrees[i] >= int(n+1/2); // = 2
                }
                takefingerprint(cyclens,cyclefp,n);
                embedsbool |= embeds(cyclens, cyclefp, ns);
                ++n;
            }
            if (embedsbool)
                return n-1;
        }
        if (!embedsbool) {
            //std::cout << "Error in girthmeasure: no cycles embed\n";
            return 0;
        }

    }

};


class maxcliquemeasure : public abstractmeasure<float> {
public:
    virtual std::string shortname() {return "cm";}

    maxcliquemeasure() : abstractmeasure<float>("Graph's largest clique") {}
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
    abstractmeasure<float>* am;

    virtual std::string shortname() {return "_mnzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return am->takemeasure(g,ns) != 0;
    }
    measurenonzerocriterion(abstractmeasure<float>* amin)
        : abstractmemorymeasure<bool>(amin->name + " measure non zero"), am{amin} {}
};

class measurezerocriterion : public abstractmemorymeasure<bool> {
public:
    std::string name = "_measurezerocriterion";
    abstractmeasure<float>* am;

    virtual std::string shortname() {return "_mzc";}

    bool takemeasure(const graphtype *g, const neighbors *ns) override {
        return am->takemeasure(g,ns) == 0;
    }
    measurezerocriterion(abstractmeasure<float>* amin)
        : abstractmemorymeasure<bool>(amin->name + " measure zero"), am{amin} {}
};

class treecriterion : public measurezerocriterion{
protected:
public:
    std::string name = "treecriterion";

    virtual std::string shortname() {return "tc";}

    treecriterion() : measurezerocriterion(new girthmeasure()) {}
    ~treecriterion() {
        delete am;
    }
};


