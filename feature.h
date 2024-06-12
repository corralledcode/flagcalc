//
// Created by peterglenn on 6/12/24.
//

#ifndef FEATURE_H
#define FEATURE_H
#include <iostream>
#include <string>

#include "asymp.h"
#include "graphs.h"
#include "EdgesforHelly.cpp"
#include "Graph.cpp"
#include "Formatgraph.cpp"
#include "Formatdata.cpp"
#include "prob.h"


class feature {
protected:
    std::istream* _is;
    std::ostream* _os;

public:
    virtual std::string cmdlineoption() {return "";};
    virtual std::string cmdlineoptionlong() { return "";};
    virtual void execute(int argc, char* argv[], int* cnt) {};
    feature( std::istream* is, std::ostream* os ) {
        _is = is;
        _os = os;
    }
};

class samplerandomgraphsfeature : public feature {
public:
    std::string cmdlineoption() {
        return "r";
    }
    std::string cmdlineoptionlong() {
        return "samplerandomgraphs";
    }
    samplerandomgraphsfeature( std::istream* is, std::ostream* os ) : feature( is, os) {
        //cmdlineoption = "r";
        //cmdlineoptionlong = "samplerandomgraphs";
    };
    void execute(int argc, char* argv[], int* cnt) override {
        int outof = 1000;
        int dim = 5;
        float edgecnt = dim*(dim-1)/4.0;
        *cnt=0;
        int rgidx = 0;
        if (argc > 1) {
            dim = std::stoi(argv[1]);
            if (argc > 2) {
                edgecnt = std::stoi(argv[2]);
                if (argc > 3) {
                    outof = std::stoi(argv[3]);
                    (*cnt)++;
                }
                (*cnt)++;
            }
            (*cnt)++;
        }
        abstractrandomgraph* rg1 = new stdrandomgraph((int)edgecnt);
        abstractrandomgraph* rg2 = new randomgraphonnedges((int)edgecnt);
        abstractrandomgraph* rg3 = new randomconnectedgraph((int)edgecnt);
        abstractrandomgraph* rg4 = new randomconnectedgraphfixededgecnt((int)edgecnt);
        std::vector<weightstype> weights;
        weights = computeweights(dim);
        abstractrandomgraph* rg5 = new weightedrandomconnectedgraph(weights);
        std::vector<abstractrandomgraph*> rgs {};
        rgs.push_back(rg1);
        rgs.push_back(rg2);
        rgs.push_back(rg3);
        rgs.push_back(rg4);
        rgs.push_back(rg5);
        if (argc > 4) {
            for (int i = 0; i < rgs.size(); ++i) {
                if (argv[4] == rgs[i]->shortname()) {
                    rgidx = i;
                }
            }
            (*cnt)++;
        }
        samplematchingrandomgraphs(rgs[rgidx],dim,outof,*_os);
            // --- yet a third functionality: randomly range over connected graphs (however, the algorithm should be checked for the right sense of "randomness"
            // note the simple check of starting with a vertex, recursively obtaining sets of neighbors, then checking that all
            // vertices are obtained, is rather efficient too.
            // Note also this definition of "randomness" is not correct: for instance, on a graph on three vertices, it doesn't run all the way
            //  up to and including three edges; it stops as soon as the graph is connected, i.e. at two vertices.

        for (int i = 0; i < rgs.size();++i) {
            delete rgs[i];
        }
    }

};


class enumisomorphismsfeature: public feature {
public:
    std::string cmdlineoption() { return "i"; }
    std::string cmdlineoptionlong() { return "enumisomorphisms"; }
    enumisomorphismsfeature( std::istream* is, std::ostream* os ) : feature( is, os) {
        //cmdlineoption = "i";
        //cmdlineoptionlong = "enumisomorphisms";
    };
    void execute(int argc, char* argv[], int* cnt) override {
        std::ifstream ifs;
        std::istream* is = &std::cin;
        std::ostream* os = _os;
        if (argc > 1) {
            std::string filename = argv[1];
            std::cout << "Opening file " << filename << "\n";
            ifs.open(filename);
            if (!ifs) {
                std::cout << "Couldn't open file for reading \n";
                return;
            }
            is = &ifs;
            *cnt = 1;
        } else {
            std::cout << "Enter a filename or enter T for terminal mode: ";
            std::string filename;
            std::cin >> filename;
            if (filename != "T") {
                ifs.open(filename);
                if (!ifs)
                    std::cout << "Couldn't open file for reading \n";
                is = &ifs;
            }
            *cnt = 0;
        }


        auto V = new Vertices();
        auto EV = new Batchprocesseddata<strtype>();
        auto FV = new Formatvertices(V,EV);
        FV->pause();
        FV->readdata(*is);
        FV->resume();
        int sV = FV->size();
        for( int n = 0; n<sV; ++n)
            *os << n << "{" << FV->idata->getdata(n) << ":" << FV->edata->getdata(n) << "}, ";
        *os << "\b\b  \n";

        auto E = new EdgesforHelly();
        auto EE = new Batchprocesseddata<Edgestr>();
        auto FE = new Formatedges(E,EE);
        FE->pause();
        FE->readdata(*is);
        FE->setvertices(FV);
        FE->resume();
        int sE = FE->size();

        for (int m = 0; m < sE; ++m) {
            *os << "{" << FE->idata->getdata(m).first << ", " << FE->idata->getdata(m).second;
            *os << ":" << FE->edata->getdata(m).first << ", " << FE->edata->getdata(m).second << "}, ";
        }
        *os << "\b\b  \n";

        // repeat the code above for a second graph...

        auto V2 = new Vertices();
        auto EV2 = new Batchprocesseddata<strtype>();
        auto FV2 = new Formatvertices(V2,EV2);
        FV2->pause();
        FV2->readdata(*is);
        FV2->resume();
        int sV2 = FV2->size();
        for( int n = 0; n<sV2; ++n)
            *os << n << "{" << FV2->idata->getdata(n) << ":" << FV2->edata->getdata(n) << "}, ";
        *os << "\b\b  \n";

        auto E2 = new EdgesforHelly();
        auto EE2 = new Batchprocesseddata<Edgestr>();
        auto FE2 = new Formatedges(E2,EE2);
        FE2->pause();
        FE2->readdata(*is);
        FE2->setvertices(FV2);
        FE2->resume();
        int sE2 = FE2->size();

        for (int m = 0; m < sE2; ++m) {
            *os << "{" << FE2->idata->getdata(m).first << ", " << FE2->idata->getdata(m).second;
            *os << ":" << FE2->edata->getdata(m).first << ", " << FE2->edata->getdata(m).second << "}, ";
        }
        *os << "\b\b  \n";


        graph g1;
        g1.dim = V->size();

        graph g2;
        g2.dim = V2->size();

        // below is rather embarrassing hack around some omissions in the general purpose intent of the HellyTool code...
        // (recall that HellyTool does not make use of the vertexadjacency matrix, hence the lack of debugging around its use...)
        E->maxvertex = g1.dim-1;
        E->vertexadjacency = new bool[(g1.dim) * g1.dim];
        E->computevertexadjacency();
        g1.adjacencymatrix = E->vertexadjacency;

        E2->maxvertex = g2.dim-1;
        E2->vertexadjacency = new bool[(g2.dim) * g2.dim];
        E2->computevertexadjacency();
        g2.adjacencymatrix = E2->vertexadjacency;

        osadjacencymatrix( *os, g1 );
        osadjacencymatrix( *os, g2 );

        neighbors ns1;
        ns1 = computeneighborslist(g1);
        osneighbors(*os,ns1);

        neighbors ns2;
        ns2 = computeneighborslist(g2);
        osneighbors(*os,ns2);

        FP fps1[g1.dim];
        for (vertextype n = 0; n < g1.dim; ++n) {
            fps1[n].v = n;
            fps1[n].ns = nullptr;
            fps1[n].nscnt = 0;
            fps1[n].parent = nullptr;
        }

        takefingerprint(ns1,fps1,g1.dim);

        osfingerprint(*os,ns1,fps1,g1.dim);

        FP fps2[g1.dim];
        for (vertextype n = 0; n < g2.dim; ++n) {
            fps2[n].v = n;
            fps2[n].ns = nullptr;
            fps2[n].nscnt = 0;
            fps2[n].parent = nullptr;
        }

        takefingerprint(ns2,fps2,g2.dim);

        FP fpstmp1;
        fpstmp1.parent = nullptr;
        fpstmp1.ns = fps1;
        fpstmp1.nscnt = g1.dim;

        FP fpstmp2;
        fpstmp2.parent = nullptr;
        fpstmp2.ns = fps2;
        fpstmp2.nscnt = g2.dim;

        osfingerprint(*os,ns2,fps2,g2.dim);
        if (FPcmp(ns1,ns2,fpstmp1,fpstmp2) == 0) {
            *os << "Fingerprints MATCH\n";
        } else {
            *os << "Fingerprints DO NOT MATCH\n";
        }

        std::vector<graphmorphism> maps = enumisomorphisms(ns1,ns2);
        osgraphmorphisms(*os, maps);

        free(FE);
        free(EE);
        free(E);
        free(FV);
        free(EV);
        free(V);
        free(ns1.neighborslist);
        free(ns1.degrees);
        freefps(fps1, g1.dim);

        free(FE2);
        free(EE2);
        free(E2);
        free(FV2);
        free(EV2);
        free(V2);
        free(ns2.neighborslist);
        free(ns2.degrees);
        freefps(fps2, g2.dim);

    }
};

class mantelstheoremfeature : public feature {
public:
    std::string cmdlineoption() { return "m"; }
    std::string cmdlineoptionlong() { return "mantels"; }
    mantelstheoremfeature( std::istream* is, std::ostream* os ) : feature( is, os) {}
    void execute(int argc, char *argv[], int *cnt) override {
        int outof = 100;
        int limitdim = 10;
        *cnt=0;
        if (argc > 1) {
            limitdim = std::stoi(argv[1]);
            if (argc > 2) {
                outof = std::stoi(argv[2]);
                (*cnt)++;
            }
            (*cnt)++;
        }

        asymp* as = new asymp();
        trianglefreecriterion* cr = new trianglefreecriterion();
        edgecountmeasure* ms = new edgecountmeasure();;
        float max = as->computeasymptotic(cr,ms,outof,limitdim);
        *_os << "Asymptotic approximation at limitdim == " << limitdim << ", outof == " << outof << ": " << max << "\n";
        *_os << "(n^2/4) == " << limitdim * limitdim / 4.0 << "\n";
        delete ms;
        delete cr;
        delete as;

    }
};




#endif //FEATURE_H
