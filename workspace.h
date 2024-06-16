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
#define VERBOSE_VERBOSITYFILEAPPEND 17
#define VERBOSE_MINIMAL 19
#define VERBOSE_MANTELSTHEOREM 23
#define VERBOSE_FINGERPRINT 27

class workitems {
public:
    std::string classname;
    std::string name;
    int verbosityfactor = 0;
    virtual bool ositem( std::ostream& os, int verbositylevel ) {
        os << classname << " " << name << ":\n";
        return true;
    }
    virtual bool isitem( std::istream& is ) {return true;}
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
    bool ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem( os, verbositylevel );

        //to do: would be nice to have a list of edges
        //and a labelling for the adjacency matrix;
        //add vertexlabels to graph struct type

        osadjacencymatrix(os,g);
        osneighbors(os,ns);
        return true;
    }




    bool isitem( std::istream& is) {
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
            return false;   // in case only one graph is given, default to computing automorphisms
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
        return g.dim > 0;   // for now no support for trivial empty graphs
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
/* already freed by graphitem
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
        }*/
    }

    bool ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        if ((verbositylevel % VERBOSE_DONTLISTISOS) == 0) {
            os << "Total number of isomorphisms == " << gm.size() << "\n";
        } else {
            osgraphmorphisms(os, gm);
        }
        return true;
    }
};

class cmpfingerprintsitem : public workitems {
public:
    std::vector<graph> glist;
    std::vector<neighbors> nslist;
    std::vector<FP> fpslist;

    bool fingerprintsmatch;
    std::vector<int> sorted {};
    cmpfingerprintsitem() : workitems() {
        classname = "Graph fingerprint comparison";
        verbosityfactor = VERBOSE_LISTFINGERPRINTS;
    }
    void freemem() override {

        // graph items are already freed by graphitem freemem

        for (int n = 0; n < fpslist.size(); ++n) {
            if (fpslist[n].nscnt > 0) {
                freefps(fpslist[n].ns,fpslist[n].nscnt);
                free(fpslist[n].ns);
            }
        }

    }
    bool ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem(os,verbositylevel);
        for (int n = 0; n < sorted.size(); ++n) {
            if (verbositylevel % VERBOSE_FINGERPRINT == 0) {
                os << "fingerprint of graph ordered number " << n+1 << " out of " << sorted.size() << "\n";
                osfingerprint(os,nslist[sorted[n]], fpslist[sorted[n]].ns, fpslist[sorted[n]].nscnt);
            }
        }
        if (fingerprintsmatch) {
            os << "Fingerprints MATCH\n";
        } else {
            os << "Some fingerprints DO NOT MATCH\n";
        }
        return true;
    }
};

class timedrunitem : public workitems {
public:
    long duration;

    timedrunitem() : workitems() {
        classname = "TimedRun";
        verbosityfactor = VERBOSE_RUNTIMES;
    }
    bool ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        os << ((float)duration)/1000000<< "\n";
        return true;
    }

};

class mantelstheoremitem : public workitems {
public:
    float max;
    int limitdim;
    int outof;

    mantelstheoremitem() : workitems() {
        classname = "MantelsTheorem";
        verbosityfactor = VERBOSE_MANTELSTHEOREM;
    }
    bool ositem( std::ostream& os, int verbositylevel ) override {
        workitems::ositem( os, verbositylevel );
        os << "Asymptotic approximation at limitdim == " << limitdim << ", outof == " << outof << ": " << max << "\n";
        os << "(n^2/4) == " << limitdim * limitdim / 4.0 << "\n";
        return true;
    }

};



#endif //WORKSPACE_H
