//
// Created by peterglenn on 6/29/24.
//


#include "graphs.h"

// how many cycles to compute in advance, e.g. in girthmeasure
#define GRAPH_PRECOMPUTECYCLESCNT 20


template<typename T>
class abstractmeasure {
public:
    std::vector<graphtype*>* gptrs {};
    std::vector<neighbors*>* nsptrs {};
    std::vector<T>* res {};
    std::string name = "_abstractmeasure";
    virtual std::string shortname() {return "_am";}

    virtual T takemeasure( const graphtype* g, const neighbors* ns ) {return {};}
    virtual T takemeasureidxed( const int idx ) {
        if ((*res)[idx] == -1) {
            (*res)[idx] = takemeasure((*gptrs)[idx],(*nsptrs)[idx]);
        }
        return (*res)[idx];
    }
    abstractmeasure( std::string namein ) :name{namein} {}
    ~abstractmeasure() {
        //delete res; // crashes...
    }
};

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


/* superceded code
class edgecountmeasure : public abstractmeasure {
public:
    float takemeasure( const graphtype* g, const neighbors* ns ) override {
        int edgecnt = 0;
        for (int n = 0; n < g->dim-1; ++n) {
            for (int i = n+1; i < g->dim; ++i) {
                if (g->adjacencymatrix[n*g->dim + i]) {
                    edgecnt++;
                }
            }
        }
        return edgecnt;
    }
};
*/