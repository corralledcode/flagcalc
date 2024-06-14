//
// Created by peterglenn on 6/12/24.
//

#ifndef WORKSPACE_H
#define WORKSPACE_H
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include "graphs.h"

#define VERBOSE_DONTLISTISOS 7
#define VERBOSE_LISTGRAPHS 2
#define VERBOSE_LISTFINGERPRINTS 3
#define VERBOSE_ISOS 5
#define VERBOSE_RUNTIMES 11

class workitems {
public:
    std::string classname;
    std::string name;
    int verbosityfactor = 0;
    virtual void ositem( std::ostream& os, int verbositylevel ) {
        os << classname << " " << name << ":\n";
    }
    virtual void isitem( std::istream& is ) {}
    virtual void freemem() {}
    workitems() {
        classname = "Unnamed class";
    }
};

class workspace {
    int namesused = 0;
public:
    std::vector<workitems*> items {};
    std::string getuniquename() {
        bool unique = true;
        std::string tmpname;
        int namesused = 0;
        do {
            tmpname = "WORKITEM" + std::to_string(namesused);
            unique = true;
            for (int n = 0; unique && (n < items.size()); ++n) {
                unique = unique && (items[n]->name != tmpname);
            }
            namesused++;
        } while (!unique);
        return tmpname;
    }
    workspace() {
    }
};

class graphitem : public workitems {
public:
    graph g;
    neighbors ns;
    graphitem() : workitems() {
        g.adjacencymatrix = nullptr;
        ns.neighborslist = nullptr;
        verbosityfactor = VERBOSE_LISTGRAPHS;
        classname = "Graph";
    }
    void freemem() override {

        if (g.adjacencymatrix != nullptr) {
            free(g.adjacencymatrix);
            g.adjacencymatrix = nullptr;
        }
        if (ns.neighborslist != nullptr) {
            free(ns.neighborslist);
            ns.neighborslist = nullptr;
        }

    }
    void ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        osadjacencymatrix(os,g);
        osneighbors(os,ns);
    }
    /*
    void isitem( std::istream& is ) override {
        std::string tmp;
        int tmpdim =0;
        int dim = 0;
        int i=0;
        int n=0;
        if (g.adjacencymatrix != nullptr) {
            free(g.adjacencymatrix);
            g.adjacencymatrix = nullptr;
        }
        if (ns.neighborslist != nullptr) {
            free(ns.neighborslist);
            ns.neighborslist = nullptr;
        }
        std::vector<bool> tmpadj {};
        do {
            is >> tmp;
            if (tmp != "\n") {
                ++tmpdim;
                tmpadj.push_back((bool)std::stoi(tmp));
                continue;
            }
            if (g.dim != 0 && g.dim != tmpdim) {
                std::cout << "Error in dims while reading graph\n";
                return;
            }
            g.dim = tmpdim;
            if (g.adjacencymatrix == nullptr) {
                g.adjacencymatrix = (bool*)malloc(g.dim*g.dim*sizeof(bool));
            }
            g.adjacencymatrix[i*g.dim + i] = false;
            for (int j = 0; j < g.dim; ++j) {
                g.adjacencymatrix[i*g.dim + j] = tmpadj[j];
            }
            ++n;
        } while (n < dim);
        for (int i = 0; i < g.dim-1; ++i ) {
            for (int n = i+1; n < g.dim; ++n) {
                if (g.adjacencymatrix[n*g.dim + i] != g.adjacencymatrix[i*g.dim + n]) {
                    std::cout << "Error in symmetry while reading graph\n";
                    return;
                }
            }

        }

    }*/

};


class enumisomorphismsitem : public workitems {
public:
    graph g1;
    graph g2;
    neighbors ns1;
    neighbors ns2;
    std::vector<graphmorphism> gm;
    enumisomorphismsitem() : workitems() {
        classname = "Graph isomorphisms";
        verbosityfactor = VERBOSE_ISOS; // use primes
    }
    void freemem() override {
/*
        if (g1.adjacencymatrix != nullptr) {
            free(g1.adjacencymatrix);
            g1.adjacencymatrix = nullptr;
        }
        if (ns1.neighborslist != nullptr) {
            free(ns1.neighborslist);
            ns1.neighborslist = nullptr;
        }
        if (g2.adjacencymatrix != nullptr) {
            free(g2.adjacencymatrix);
            g2.adjacencymatrix = nullptr;
        }
        if (ns2.neighborslist != nullptr) {
            free(ns2.neighborslist);
            ns2.neighborslist = nullptr;
        }
        */
    }
    void ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        if ((verbositylevel % VERBOSE_DONTLISTISOS) == 0) {
            os << "Total number of isomorphisms == " << gm.size() << "\n";
        } else {
            osgraphmorphisms(os, gm);
        }
    }
};

class cmpfingerprintsitem : public workitems {
public:
    graph g1;
    graph g2;
    neighbors ns1;
    neighbors ns2;
    FP* fps1;
    FP* fps2;
    int fps1cnt = 0;
    int fps2cnt = 0;
    bool fingerprintsmatch;
    cmpfingerprintsitem() : workitems() {
        classname = "Graph fingerprint comparison";
        verbosityfactor = VERBOSE_LISTFINGERPRINTS;
    }
    void freemem() override {
        if (g1.adjacencymatrix != nullptr) {
            free(g1.adjacencymatrix);
            g1.adjacencymatrix = nullptr;
        }
        if (ns1.neighborslist != nullptr) {
            free(ns1.neighborslist);
            ns1.neighborslist = nullptr;
        }
        if (g2.adjacencymatrix != nullptr) {
            free(g2.adjacencymatrix);
            g2.adjacencymatrix = nullptr;
        }
        if (ns2.neighborslist != nullptr) {
            free(ns2.neighborslist);
            ns2.neighborslist = nullptr;
        }
        if (fps1cnt > 0)
            freefps(fps1,fps1cnt);
        if (fps2cnt > 0)
            freefps(fps2,fps2cnt);

    }
    void ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        os << "fingerprint of first graph\n";
        osfingerprint(os,ns1, fps1,fps1cnt);
        os << "fingerprint of second graph\n";
        //osfingerprint(os,ns2, fps2,fps2cnt);
        if (fingerprintsmatch) {
            os << "Fingerprints MATCH\n";
        } else {
            os << "Fingerprints DO NOT MATCH\n";
        }
    }
};

class timedrunitem : public workitems {
public:
    float duration;

    timedrunitem() : workitems() {
        classname = "TimedRun";
        verbosityfactor = VERBOSE_RUNTIMES;
    }
    void ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        os << duration << "\n";
    }

};



#endif //WORKSPACE_H
