//
// Created by peterglenn on 6/12/24.
//

#ifndef FEATURE_H
#define FEATURE_H
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>

#include "asymp.h"
#include "graphs.h"
#include "prob.h"
#include "workspace.h"


class feature {
protected:
    std::istream* _is;
    std::ostream* _os;
    workspace* _ws;
public:
    virtual std::string cmdlineoption() {return "";};
    virtual std::string cmdlineoptionlong() { return "";};
    virtual void execute(int argc, char* argv[], int* cnt) {};
    feature( std::istream* is, std::ostream* os, workspace* ws ) {
        _is = is;
        _os = os;
        _ws = ws;
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
    samplerandomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
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

class verbosityfeature : public feature {
public:
    int verbositylevel = 0;
    std::string ofname {};
    std::string cmdlineoption() {
        return "v";
    }
    std::string cmdlineoptionlong() {
        return "verbosity";
    }
    verbosityfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "r";
        //cmdlineoptionlong = "samplerandomgraphs";
    };
    void execute(int argc, char* argv[], int* cnt) override {
        std::ofstream ofs;
        if (argc > 1) {
            verbositylevel = std::stoi(argv[1]);
            if (argc > 2) {
                ofname = argv[2];
                (*cnt)++;
                ofs.open(ofname);
                if (!ofs) {
                    std::cout << "Couldn't open file for writing \n";
                    (*cnt)++;
                    return;
                }
                _os = &ofs;

            } else
            {
                _os = &std::cout;
            }
            //(*cnt)++;
        }


        auto starttime = std::chrono::high_resolution_clock::now();

        for (int n = 0; n < _ws->items.size(); ++n) {
            if (verbositylevel % _ws->items[n]->verbosityfactor == 0) {
                _ws->items[n]->ositem(*_os,verbositylevel);
            }
        }

        auto stoptime = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(stoptime - starttime);

        if (verbositylevel % VERBOSE_VERBOSITYRUNTIME == 0)
        {
            timedrunitem* tr = new timedrunitem();
            tr->duration = duration.count();
            tr->name = "TimedRunVerbosity";
            tr->ositem(*_os,verbositylevel);
        }

    }

};



class cmpfingerprintsfeature: public feature {
public:
    graph internalg;
    bool useinternalg = false;
    std::string cmdlineoption() { return "f"; }
    std::string cmdlineoptionlong() { return "cmpfingerprints"; }
    cmpfingerprintsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "f";
        //cmdlineoptionlong = "cmpfingerprints";
    };
    void execute(int argc, char* argv[], int* cnt) override {
        std::ifstream ifs;
        std::istream* is = &std::cin;
        std::ostream* os = _os;
        if (argc > 1) {
            std::string filename = argv[1];
            *_os << "Opening file " << filename << "\n";
            ifs.open(filename);
            if (!ifs) {
                std::cout << "Couldn't open file for reading \n";
                (*cnt)++;
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
                if (!ifs) {
                    std::cout << "Couldn't open file for reading \n";
                    (*cnt)++;
                    return;
                }
                is = &ifs;
            }
            *cnt = 0;
        }

        graphitem* gi1 = new graphitem();
        gi1->isitem(*is);

        graphitem* gi2 = new graphitem();
        if (useinternalg) {
            gi2->g = internalg;  // a better approach is to find it on the workspace
        }
        gi2->isitem(*is);
        if (gi2->g.dim == 0) {
            gi2->g.dim = gi1->g.dim;
            gi2->g.adjacencymatrix = (bool*)malloc(gi2->g.dim*gi2->g.dim*sizeof(bool));
            for (int n = 0; n < gi2->g.dim; ++n) {
                for (int i = 0; i < gi2->g.dim; ++i) {
                    gi2->g.adjacencymatrix[n*gi2->g.dim + i] = gi1->g.adjacencymatrix[n*gi2->g.dim + i];
                }
            }
            gi2->ns = computeneighborslist(gi2->g);

            //gi2->g = gi1->g;
        }

        FP* fps1 = (FP*)malloc(gi1->g.dim * sizeof(FP));
        for (vertextype n = 0; n < gi1->g.dim; ++n) {
            fps1[n].v = n;
            fps1[n].ns = nullptr;
            fps1[n].nscnt = 0;
            fps1[n].parent = nullptr;
        }

        takefingerprint(gi1->ns,fps1,gi1->g.dim);

        //osfingerprint(*os, gi1->ns, fps1, gi1->g.dim);

        FP* fps2 = (FP*)malloc(gi2->g.dim * sizeof(FP));
        for (vertextype n = 0; n < gi2->g.dim; ++n) {
            fps2[n].v = n;
            fps2[n].ns = nullptr;
            fps2[n].nscnt = 0;
            fps2[n].parent = nullptr;
        }

        takefingerprint(gi2->ns,fps2,gi2->g.dim);

        FP fpstmp1;
        fpstmp1.parent = nullptr;
        fpstmp1.ns = fps1;
        fpstmp1.nscnt = gi1->g.dim;

        FP fpstmp2;
        fpstmp2.parent = nullptr;
        fpstmp2.ns = fps2;
        fpstmp2.nscnt = gi2->g.dim;


        bool res = (FPcmp(gi1->ns,gi2->ns,fpstmp1,fpstmp2) == 0);

        cmpfingerprintsitem* wi = new cmpfingerprintsitem();
        wi->fingerprintsmatch = res;
        wi->name = _ws->getuniquename();
        wi->g1 = gi1->g;
        wi->ns1 = gi1->ns;


        //wi->fps1 = (FP*)malloc(gi1->g.dim * sizeof(FP));


        wi->fps1 = fps1;
        /*for (int n = 0; n < gi1->g.dim; ++n) {
            wi->fps1[n].ns = nullptr; //fps1[n].ns;
            wi->fps1[n].v = fps1[n].v;
            wi->fps1[n].nscnt = 0; //fps1[n].nscnt;
            wi->fps1[n].parent = nullptr;

        }*/

        wi->fps1cnt = gi1->g.dim;

        wi->g2 = gi2->g;
        wi->ns2 = gi2->ns;

        //wi->fps2 = (FP*)malloc(gi2->g.dim * sizeof(FP));


        wi->fps2 = fps2;

        /*for (int n = 0; n < gi2->g.dim; ++n) {
            wi->fps2[n].ns = nullptr; //fps2[n].ns;
            wi->fps2[n].v = fps2[n].v;
            wi->fps2[n].nscnt = 0; //fps2[n].nscnt;
            wi->fps2[n].parent = nullptr;
        }*/

        wi->fps2cnt = gi2->g.dim;

        _ws->items.push_back(wi);
    }
};


class enumisomorphismsfeature: public feature {
public:
    graph internalg;
    bool useinternalg = false;
    std::string cmdlineoption() { return "i"; }
    std::string cmdlineoptionlong() { return "enumisomorphisms"; }
    enumisomorphismsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "i";
        //cmdlineoptionlong = "enumisomorphisms";
    };
    void execute(int argc, char* argv[], int* cnt) override {
        std::ifstream ifs;
        std::istream* is = &std::cin;
        std::ostream* os = _os;
        if (argc > 1) {
            std::string filename = argv[1];
            *_os << "Opening file " << filename << "\n";
            ifs.open(filename);
            if (!ifs) {
                std::cout << "Couldn't open file for reading \n";
                (*cnt)++;
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
                if (!ifs) {
                    std::cout << "Couldn't open file for reading \n";
                    (*cnt)++;
                    return;
                }
                is = &ifs;
            }
            *cnt = 0;
        }

        graphitem* gi1 = new graphitem();
        gi1->isitem(*is);

        graphitem* gi2 = new graphitem();
        if (useinternalg) {
            gi2->g = internalg;  // a better approach is to find it on the workspace
        }
        gi2->isitem(*is);
        if (gi2->g.dim == 0) {
            gi2->g.dim = gi1->g.dim;
            gi2->g.adjacencymatrix = (bool*)malloc(gi2->g.dim*gi2->g.dim*sizeof(bool));
            for (int n = 0; n < gi2->g.dim; ++n) {
                for (int i = 0; i < gi2->g.dim; ++i) {
                    gi2->g.adjacencymatrix[n*gi2->g.dim + i] = gi1->g.adjacencymatrix[n*gi1->g.dim + i];
                }
            }
            gi2->ns = computeneighborslist(gi2->g);

            //gi2->g = gi1->g;
        }

        FP fps1[gi1->g.dim];
        for (vertextype n = 0; n < gi1->g.dim; ++n) {
            fps1[n].v = n;
            fps1[n].ns = nullptr;
            fps1[n].nscnt = 0;
            fps1[n].parent = nullptr;
        }

        takefingerprint(gi1->ns,fps1,gi1->g.dim);

        FP fps2[gi2->g.dim];
        for (vertextype n = 0; n < gi2->g.dim; ++n) {
            fps2[n].v = n;
            fps2[n].ns = nullptr;
            fps2[n].nscnt = 0;
            fps2[n].parent = nullptr;
        }

        takefingerprint(gi2->ns,fps2,gi2->g.dim);

        FP fpstmp1;
        fpstmp1.parent = nullptr;
        fpstmp1.ns = fps1;
        fpstmp1.nscnt = gi1->g.dim;

        FP fpstmp2;
        fpstmp2.parent = nullptr;
        fpstmp2.ns = fps2;
        fpstmp2.nscnt = gi2->g.dim;

        std::vector<graphmorphism> maps = enumisomorphisms(gi1->ns,gi2->ns);

        gi1->name = _ws->getuniquename();
        _ws->items.push_back(gi1);

        gi2->name = _ws->getuniquename();
        _ws->items.push_back(gi2);

        enumisomorphismsitem* wi = new enumisomorphismsitem();
        wi->name = _ws->getuniquename();
        wi->g1 = gi1->g;
        wi->ns1 = gi1->ns;
        wi->g2 = gi2->g;
        wi->ns2 = gi2->ns;

        wi->gm = maps;

        _ws->items.push_back(wi);


    }

};

class mantelstheoremfeature : public feature {
public:
    std::string cmdlineoption() { return "m"; }
    std::string cmdlineoptionlong() { return "mantels"; }
    mantelstheoremfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {}
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
        float max = as->computeasymptotic(cr,ms,outof,limitdim, *_os, _ws);
        *_os << "Asymptotic approximation at limitdim == " << limitdim << ", outof == " << outof << ": " << max << "\n";
        *_os << "(n^2/4) == " << limitdim * limitdim / 4.0 << "\n";
        delete ms;
        delete cr;
        delete as;

    }
};

class mantelsverifyfeature : public feature {
public:
    std::string cmdlineoption() { return "M"; }
    std::string cmdlineoptionlong() { return "mantelsverify"; }
    mantelsverifyfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {}
    void execute(int argc, char *argv[], int *cnt) override {
        mantelstheoremfeature* ms = new mantelstheoremfeature(_is,_os,_ws);
        ms->execute(argc, argv, cnt);
        std::cout << "cnt == " << *cnt << "\n";
        enumisomorphismsfeature* ei = new enumisomorphismsfeature(_is,_os,_ws);
        ei->internalg = ((graphitem*)(_ws->items[_ws->items.size()-1]))->g;
        ei->useinternalg = true;
        int tmpcnt = *cnt;
        ei->execute(argc-tmpcnt,argv+(tmpcnt*sizeof(char)),cnt);
        *cnt = *cnt + tmpcnt;
        delete ms;
        delete ei;
    }


};




#endif //FEATURE_H
