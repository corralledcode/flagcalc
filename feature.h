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

//default is to enumisomorphisms
#define DEFAULTCMDLINESWITCH "i"


class feature {
protected:
    std::istream* _is;
    std::ostream* _os;
    workspace* _ws;
public:
    virtual std::string cmdlineoption() {return "";};
    virtual std::string cmdlineoptionlong() { return "";};
    virtual void execute(std::vector<std::string>) {};
    feature( std::istream* is, std::ostream* os, workspace* ws ) {
        _is = is;
        _os = os;
        _ws = ws;
    }
    virtual ~feature() {}
};

class _sandboxfeature : public feature { // to be used to code hack jobs for testing purposes
public:
    std::string cmdlineoption() {
        return "+";
    }
    std::string cmdlineoptionlong() {
        return "sandboxfeature";
    }

    _sandboxfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws ) {
        //
    }
    void execute(std::vector<std::string> args) override {
/*
        std::ifstream ifs;
        std::istream* is = &std::cin;
        std::ostream* os = _os;
        graphitem* gi1 = new graphitem();
        gi1->isitem(*is);
        trianglefreecriterion tf;
        std::cout << "Triangle free returns " << tf.checkcriterion(gi1->g,gi1->ns) << "\n";
*/
    }

};

class abstractrandomgraphsfeature : public feature {
protected:
    std::vector<abstractrandomgraph*> rgs {};

public:
    abstractrandomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {

        // add any new abstractrandomgraph types to the list here...

        auto rg1 = new stdrandomgraph();
        auto rg2 = new randomgraphonnedges();
        auto rg3 = new randomconnectedgraph();
        auto rg4 = new randomconnectedgraphfixededgecnt();
        auto rg5 = new weightedrandomconnectedgraph();
        rgs.push_back(rg1);
        rgs.push_back(rg2);
        rgs.push_back(rg3);
        rgs.push_back(rg4);
        rgs.push_back(rg5);
    }

    ~abstractrandomgraphsfeature() {
        for (int i = 0; i < rgs.size();++i) {
            delete rgs[i];
        }
    }

};

class samplerandomgraphsfeature : public abstractrandomgraphsfeature {
public:
    float percent = -1;
    std::string cmdlineoption() {
        return "R";
    }
    std::string cmdlineoptionlong() {
        return "samplerandomgraphs";
    }

    samplerandomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractrandomgraphsfeature( is, os, ws) {}

    void execute(std::vector<std::string> args) override {
        int outof = 1000;
        int dim = 5;
        float edgecnt = dim*(dim-1)/4.0;
        int rgidx = 0;
        if (args.size() > 1) {
            dim = std::stoi(args[1]);
            if (args.size() > 2) {
                edgecnt = std::stof(args[2]);
                if (args.size() > 3) {
                    outof = std::stoi(args[3]);
                }
            }
        }
        if (args.size() > 4) {
            for (int i = 0; i < rgs.size(); ++i) {
                if (args[4] == rgs[i]->shortname()) {
                    rgidx = i;
                }
            }
        }
        int cnt = samplematchingrandomgraphs(rgs[rgidx],dim,edgecnt, outof);
            // --- yet a third functionality: randomly range over connected graphs (however, the algorithm should be checked for the right sense of "randomness"
            // note the simple check of starting with a vertex, recursively obtaining sets of neighbors, then checking that all
            // vertices are obtained, is rather efficient too.
            // Note also this definition of "randomness" is not correct: for instance, on a graph on three vertices, it doesn't run all the way
            //  up to and including three edges; it stops as soon as the graph is connected, i.e. at two vertices.

        auto wi = new samplerandommatchinggraphsitem;
        wi->dim = dim;
        wi->cnt = cnt;
        wi->outof = outof;
        wi->name = _ws->getuniquename(wi->classname);

        wi-> percent = ((float)wi->cnt / (float)wi->outof);
        wi->rgname = rgs[rgidx]->name;
        _ws->items.push_back(wi);

        // to do: add a work item class and feature that announces which random alg was used to produce the graphs

    }

};

class randomgraphsfeature : public abstractrandomgraphsfeature {
public:
    std::string cmdlineoption() {
        return "r";
    }
    std::string cmdlineoptionlong() {
        return "outputrandomgraphs";
    }

    randomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractrandomgraphsfeature( is, os, ws) {
    }

    void execute(std::vector<std::string> args) override {
        int dim = 5;
        int cnt = 2;
        float edgecnt = dim*(dim-1)/4.0;
        int rgidx = 0;
        if (args.size() > 1) {
            dim = std::stoi(args[1]);
            edgecnt = dim*(dim-1)/4.0;
            if (args.size() > 2) {
                edgecnt = std::stof(args[2]);
                if (args.size() > 3) {
                    cnt = std::stoi(args[3]);
                }
            }
        }
        if (args.size() > 4) {
            for (int i = 0; i < rgs.size(); ++i) {
                if (args[4] == rgs[i]->shortname()) {
                    rgidx = i;
                }
            }
        }
        std::vector<graph> gv {};
        gv = randomgraphs(rgs[rgidx],dim,edgecnt,cnt);

        for (int i = 0; i < gv.size(); ++i) {
            auto wi = new graphitem;
            wi->g.dim = gv[i].dim;
            wi->g.adjacencymatrix = gv[i].adjacencymatrix;
            wi->ns = computeneighborslist(wi->g);
            wi->name = _ws->getuniquename(wi->classname);
            _ws->items.push_back(wi);
        }
    }
};



class verbosityfeature : public feature {
public:
    std::string verbositylevel = "";
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
    ~verbosityfeature() {
        if (verbositylevel == "") {  // if hasn't been run yet
            std::vector<std::string> args {};
            args.push_back(cmdlineoption());
            args.push_back("std::cout");
            args.push_back(VERBOSE_DEFAULT);
            execute(args);
        }
    }
    void execute(std::vector<std::string> args) override {
        std::ofstream ofs;
        bool ofsrequiresclose = false;
        verbositylevel = VERBOSE_DEFAULT;
        if (args.size() > 2) {
            verbositylevel = args[2];
        }
        if (args.size() > 1) {
            if (args[1] == "std::cout") {
                _os = &std::cout;
            } else {
                ofname = args[1];
                std::ifstream infile(ofname);
                if (infile.good() && !verbositycmdlineincludes(verbositylevel, VERBOSE_VERBOSITYFILEAPPEND)) {
                    std::cout << "Output file " << ofname << " already exists; use verbosity \"" << VERBOSE_VERBOSITYFILEAPPEND << "\" to append it.\n";
                    return;
                }
                ofs.open(ofname,  std::ios::app);
                if (!ofs) {
                    std::cout << "Couldn't open file for writing \n";
                    return;
                }
                _os = &ofs;
                ofsrequiresclose = true;
            }
        }

        auto starttime = std::chrono::high_resolution_clock::now();

        for (int n = 0; n < _ws->items.size(); ++n) {
            if (verbositycmdlineincludes(verbositylevel, _ws->items[n]->verbositylevel)) {
                _ws->items[n]->ositem(*_os,verbositylevel);
            }
        }

        auto stoptime = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(stoptime - starttime);

        if (verbositycmdlineincludes(verbositylevel,VERBOSE_VERBOSITYRUNTIME))
        {
            timedrunitem* tr = new timedrunitem();
            tr->duration = duration.count();
            tr->name = "TimedRunVerbosity";
            tr->ositem(*_os,verbositylevel);
        }
        if (ofsrequiresclose)
            ofs.close();

    }

};

class readgraphsfeature : public feature {
public:
    std::string cmdlineoption() { return "d"; }
    std::string cmdlineoptionlong() { return "readgraphs"; }
    readgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "d";
        //cmdlineoptionlong = "readgraphs";
    }

    void execute(std::vector<std::string> args) override {
        int filenameidx = 0;
        bool oncethrough = true;
        while ((args.size() <= 1 && oncethrough) || filenameidx < args.size()-1) {
            ++filenameidx;
            oncethrough = false;
            std::ifstream ifs;
            std::istream* is = &std::cin;
            std::ostream* os = _os;
            if (args.size() > filenameidx) {
                std::string filename = args[filenameidx];
                *_os << "Opening file " << filename << "\n";
                ifs.open(filename);
                if (!ifs) {
                    std::cout << "Couldn't open file for reading \n";
                    return;
                }
                is = &ifs;
            } else {
                std::cout << "Using std::cin for input\n"; // recode so this is possible
            }


            graphitem* gi = new graphitem();
            while (gi->isitem(*is)) {
                gi->name = _ws->getuniquename(gi->classname);
                _ws->items.push_back(gi);
                gi = new graphitem();
            }
            delete gi;
        }
    }

};


class cmpfingerprintsfeature: public feature {
public:
    std::string cmdlineoption() { return "f"; }
    std::string cmdlineoptionlong() { return "cmpfingerprints"; }
    cmpfingerprintsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "f";
        //cmdlineoptionlong = "cmpfingerprints";
    };
    void execute(std::vector<std::string> args) override {

        std::vector<int> items {}; // a list of indices within workspace of the graph items to FP and sort


        bool takeallgraphitems = true;
        int numofitemstotake = 0;

        if (args.size() > 1) {
            if (args[1] == "all")
                takeallgraphitems = true;
            else {
                takeallgraphitems = false;
                numofitemstotake = 2;   // to do: code the regex of say "c=3" to mean go back three graphs on the workspace
                // also code a list of indices in reverse chrono order
            }
        }

        int idx = _ws->items.size();
        while (idx > 0 && (takeallgraphitems || numofitemstotake > 0)) {
            idx--;
            if (_ws->items[idx]->classname == "GRAPH")  {
                items.push_back(idx);
                --numofitemstotake;

            } else {
                //std::cout << idx << ": " << _ws->items[idx]->classname << "\n";
            }
        }

        auto wi = new cmpfingerprintsitem;
        wi->nslist.resize(items.size());
        wi->fpslist.resize(items.size());
        wi->glist.resize(items.size());
        wi->gnames.resize(items.size());
        for (int i = 0; i < items.size(); ++i) {
            auto gi = (graphitem*)(_ws->items[items[i]]);
            //std::cout << idx << ": " << _ws->items[idx]->classname << "\n";
            //gi->ositem(*_os,11741730);

            FP* fpsptr = (FP*)malloc(gi->g.dim * sizeof(FP));
            wi->fpslist[i].ns = fpsptr;
            wi->fpslist[i].parent = nullptr;
            wi->fpslist[i].nscnt = gi->g.dim;

            for (vertextype n = 0; n < gi->g.dim; ++n) {
                fpsptr[n].v = n;
                fpsptr[n].ns = nullptr;
                fpsptr[n].nscnt = 0;
                fpsptr[n].parent = nullptr;
            }
            takefingerprint(gi->ns,wi->fpslist[i].ns,gi->g.dim);
            wi->nslist[i] = gi->ns;
            wi->glist[i] = gi->g;
            wi->gnames[i] = gi->name;
        }
        //osfingerprint(*os, gi1->ns, fps1, gi1->g.dim);

        bool changed = true;
        bool overallres = true;
        wi->sorted.resize(items.size());
        for (int i = 0; i < items.size(); ++i) {
            wi->sorted[i] = i;
        }
        std::vector<int> res {};
        res.resize(items.size());
        overallres = true;
        while (changed) {
            // to do: use threads to sort quickly
            changed = false;
            for (int i = 0; i < items.size()-1; ++i) {
                res[i] = FPcmp(wi->nslist[wi->sorted[i]],wi->nslist[wi->sorted[i+1]],wi->fpslist[wi->sorted[i]],wi->fpslist[wi->sorted[i+1]]);
                if (res[i] < 0) {
                    int tmp;
                    tmp = wi->sorted[i+1];
                    wi->sorted[i+1] = wi->sorted[i];
                    wi->sorted[i] = tmp;
                    changed = true;
                }
                if (res[i] != 0)
                    overallres = false;
            }
        }
        wi->fingerprintsmatch = overallres;
        wi->res = res;
        wi->name = _ws->getuniquename(wi->classname);
        _ws->items.push_back(wi);

    }
};


class enumisomorphismsfeature: public feature {
public:
    bool useworkspaceg2 = false;
    std::string cmdlineoption() { return "i"; }
    std::string cmdlineoptionlong() { return "enumisomorphisms"; }
    enumisomorphismsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "i";
        //cmdlineoptionlong = "enumisomorphisms";
    };
    void execute(std::vector<std::string> args) override {
        std::vector<int> items {}; // a list of indices within workspace of the graph items to FP and sort


        bool takeallgraphitems = false;
        int numofitemstotake = 2;

        /*
        if (args.size() > 1) {
            if (args[1] == "all")
                takeallgraphitems = true;
            else {
                takeallgraphitems = false;
                numofitemstotake = 2;   // to do: code the regex of say "c=3" to mean go back three graphs on the workspace
                // also code a list of indices in reverse chrono order
            }
        }*/

        int idx = _ws->items.size();
        while (idx > 0 && (takeallgraphitems || numofitemstotake > 0)) {
            idx--;
            if (_ws->items[idx]->classname == "GRAPH")  {
                items.push_back(idx);
                --numofitemstotake;

            } else {
                //std::cout << idx << ": " << _ws->items[idx]->classname << "\n";
            }
        }

        if (items.size() == 0) {
            std::cout << "No graphs available to enum isomorphisms\n";
            return;
        }
        auto wi = new enumisomorphismsitem;
        std::vector<neighbors> nslist {};
        nslist.resize(items.size());
        std::vector<graph> glist {};
        glist.resize(items.size());
        std::vector<FP> fpslist {};
        fpslist.resize(items.size());
        // code this to run takefingerprint only once if graph 1 = graph 2
        for (int i = 0; i < items.size(); ++i) {
            auto gi = (graphitem*)(_ws->items[items[i]]);
            //std::cout << idx << ": " << _ws->items[idx]->classname << "\n";
            //gi->ositem(*_os,11741730);

            FP* fpsptr = (FP*)malloc(gi->g.dim * sizeof(FP));
            fpslist[i].ns = fpsptr;
            fpslist[i].parent = nullptr;
            fpslist[i].nscnt = gi->g.dim;

            for (vertextype n = 0; n < gi->g.dim; ++n) {
                fpsptr[n].v = n;
                fpsptr[n].ns = nullptr;
                fpsptr[n].nscnt = 0;
                fpsptr[n].parent = nullptr;
            }
            takefingerprint(gi->ns,fpslist[i].ns,gi->g.dim);
            nslist[i] = gi->ns;
            glist[i] = gi->g;
        }
        //osfingerprint(*os, gi1->ns, fps1, gi1->g.dim);

        if (items.size() == 1)
        {
            wi->gm = enumisomorphisms(nslist[0],nslist[0]);
        } else {
            wi->gm = enumisomorphisms(nslist[0],nslist[1]);
        }
        wi->name = _ws->getuniquename(wi->classname);
        _ws->items.push_back(wi);

        for (int i = 0; i < fpslist.size(); ++i) {
            freefps(fpslist[i].ns,wi->g1.dim);
            free(fpslist[i].ns);
        }
    }

};







class mantelstheoremfeature : public feature {
public:
    std::string cmdlineoption() { return "m"; }
    std::string cmdlineoptionlong() { return "mantels"; }
    mantelstheoremfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {}
    void execute(std::vector<std::string> args) override {
        int outof = 100;
        int limitdim = 10;
        if (args.size() > 1) {
            limitdim = std::stoi(args[1]);
            if (args.size() > 2) {
                outof = std::stoi(args[2]);
            }
        }

        asymp* as = new asymp();
        trianglefreecriterion* cr = new trianglefreecriterion();
        edgecountmeasure* ms = new edgecountmeasure();;
        float max = as->computeasymptotic(cr,ms,outof,limitdim, *_os, _ws);

        auto mi = new mantelstheoremitem();
        mi->limitdim = limitdim;
        mi->outof = outof;
        mi->max = max;

        _ws->items.push_back(mi);

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
    void execute(std::vector<std::string> args) override {
        mantelstheoremfeature* ms = new mantelstheoremfeature(_is,_os,_ws);
        ms->execute(args);
        enumisomorphismsfeature* ei = new enumisomorphismsfeature(_is,_os,_ws);
        ei->useworkspaceg2 = true;
        ei->execute(args); // or send it a smaller subset of args
        delete ms;
        delete ei;
    }
};




#endif //FEATURE_H
