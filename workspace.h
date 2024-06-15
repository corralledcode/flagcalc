//
// Created by peterglenn on 6/12/24.
//

#ifndef WORKSPACE_H
#define WORKSPACE_H
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <regex>
//#include <bits/regex.h>

#include "graphs.h"

#define VERBOSE_DONTLISTISOS 7
#define VERBOSE_LISTGRAPHS 2
#define VERBOSE_LISTFINGERPRINTS 3
#define VERBOSE_ISOS 5
#define VERBOSE_RUNTIMES 11
#define VERBOSE_VERBOSITYRUNTIME 13

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




    void isitem( std::istream& is) {
        //auto p = paused();
        //pause();
        //if (!idata || !edata)
        //    throw std::exception();
        int s = 0;

        std::vector<std::string> eres{};
        std::string tmp = "";
        std::string tmp2 = "";
        while ((is >> tmp) && (tmp != "END") && (tmp != "###")) {
            eres.push_back(tmp);
            tmp2 += tmp + " ";
            tmp = "";
            s++;
        }
        g.dim = s;
        std::vector<std::string> vertexlabels {}; // ultimately store this in workitem to use for readouts
        if (tmp2.size() > 0) {
            std::regex pat{"([\\w]+)"};

            for (std::sregex_iterator p(tmp2.begin(), tmp2.end(), pat); p != std::sregex_iterator{}; ++p) {
                std::string tmp3;
                tmp3 = (*p)[1];
                vertexlabels.push_back(tmp3);
            }

            // idata->removeduplicates(); must not remove duplicates yet... wait until setvertices has been called
            g.dim = vertexlabels.size();
            g.adjacencymatrix = (bool*)malloc(g.dim*g.dim*sizeof(bool));
            for (int i = 0; i <  g.dim; ++i)
                for (int n = 0; n< g.dim; ++n)
                    g.adjacencymatrix[n*g.dim + i] = false;
        } else {
            g.dim = 0;
            g.adjacencymatrix = nullptr;
            return;   // in case only one graph is given, default to computing automorphisms
        }

        //for (int n = 0; n < vertexlabels.size(); ++n) {
        //    std::cout << vertexlabels[n] << ", ";
        //}
        //std::cout << "\b\b\n";
        s = 0;

        tmp = "";
        tmp2 = "";

        eres.clear();
        while ((is >> tmp) && (tmp != "END") && (tmp != "###")) {
            eres.push_back(tmp);
            tmp2 += tmp + " ";
            tmp = "";
            s++;
        }
        std::vector<std::string> edgecommands {};
        if (tmp2.size() > 0) {
            std::regex pat{"([[:punct:]]*[\\w]+)"};

            for (std::sregex_iterator p(tmp2.begin(), tmp2.end(), pat); p != std::sregex_iterator{}; ++p) {
                std::string tmp3;
                tmp3 = (*p)[1];
                edgecommands.push_back(tmp3);
                //std::cout << tmp3 << ", ";
            }
            //std::cout << "\n";

            // idata->removeduplicates(); must not remove duplicates yet... wait until setvertices has been called
        }

        if (edgecommands.size() >= 0) {
            std::regex pat {"([a-zA-Z]{1}[\\d_]*)"};
            for( int n = 0;n< edgecommands.size(); ++n) {
                std::vector<std::string> v;
                bool cmdomit = false;
                bool cmdcomplete = true;
                bool cmdline = false;
                if (edgecommands[n].size() > 0) {
                    if (edgecommands[n][0] == '*')
                        cmdcomplete=true;
                    if (edgecommands[n][0] == '!') {
                        cmdomit = true;
                        if (edgecommands[n].size()>1) {
                            if (edgecommands[n][1] == '-') {
                                cmdline = true;
                                cmdcomplete = false;
                            }
                        }
                    }
                    if (edgecommands[n][0] == '-') {
                        cmdline = true;
                        cmdcomplete = false;
                    }
                }
                for (std::sregex_iterator p(edgecommands[n].begin(),edgecommands[n].end(),pat); p != std::sregex_iterator{};++p)
                    v.push_back((*p)[1]);
                if (!cmdline)
                    std::sort(v.begin(), v.end());
                int sz = v.size();
                //std::cout<< "v.size == " << v.size() << "\n";
                if (cmdcomplete) {
                    // connect all pairs within the sequence of vertices
                    for (int m = 0; m < sz; ++m) {
                        for (int n = m+1; n < sz; ++n) {
                            int i = 0;
                            while( i < vertexlabels.size() && vertexlabels[i] != v[m])
                                ++i;
                            int j = 0;
                            while( j < vertexlabels.size() && vertexlabels[j] != v[n])
                                ++j;
                            if (j < vertexlabels.size() && i < vertexlabels.size()) {
                                if (vertexlabels[j] == v[n] && vertexlabels[i] == v[m]) {
                                    if (j != i) {
                                        g.adjacencymatrix[i*g.dim + j] = !cmdomit;
                                        g.adjacencymatrix[j*g.dim + i] = !cmdomit;
                                        //std::cout << "v[m]: " << v[m] << " v[n]: " << v[n] << "\n";
                                    }
                                }
                            }
                        }
                    }
                }
                if (cmdline) {
                    for (int m = 0; m < (sz-1); ++m) {
                        int i = 0;
                        while( i < vertexlabels.size() && vertexlabels[i] != v[m])
                            ++i;
                        int j = 0;
                        while( j < vertexlabels.size() && vertexlabels[j] != v[m+1])
                            ++j;
                        if (j < vertexlabels.size() && i < vertexlabels.size()) {
                            if (vertexlabels[i] == v[m] && vertexlabels[j] == v[m+1]) {
                                if (i != j) {
                                    g.adjacencymatrix[i*g.dim + j] = !cmdomit;
                                    g.adjacencymatrix[j*g.dim + i] = !cmdomit;
                                    //std::cout << "v[m]: " << v[m] << " v[m+1]: " << v[m+1] << "\n";
                                }
                            }
                        }
                    }
                }
            }
        }
        ns = computeneighborslist(g);
    }





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
        osfingerprint(os,ns2, fps2,fps2cnt);
        if (fingerprintsmatch) {
            os << "Fingerprints MATCH\n";
        } else {
            os << "Fingerprints DO NOT MATCH\n";
        }
    }
};

class timedrunitem : public workitems {
public:
    long duration;

    timedrunitem() : workitems() {
        classname = "TimedRun";
        verbosityfactor = VERBOSE_RUNTIMES;
    }
    void ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        os << ((float)duration)/1000000<< "\n";
    }

};



#endif //WORKSPACE_H
