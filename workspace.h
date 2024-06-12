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

class workitems {
public:
    std::string name;
    virtual void ositem( std::ostream& os ) {}
    virtual void isitem( std::istream& is ) {}
    virtual void freemem() {}
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
};

class graphitem : public workitems {
public:
    graph g;
    neighbors ns;
    graphitem() : workitems() {
        g.adjacencymatrix = nullptr;
        ns.neighborslist = nullptr;
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
    void ositem( std::ostream& os ) override {
        osadjacencymatrix(os,g);
        osneighbors(os,ns);
    }
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

    }

};



#endif //WORKSPACE_H
