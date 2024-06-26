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
#include "graphoutcome.h"
#include "graphs.h"
#include "prob.h"
#include "workspace.h"
#include "mantel.h"

//default is to enumisomorphisms
#define DEFAULTCMDLINESWITCH "i"

//Choose ONE of the two following
//#define NAIVESORT
#define QUICKSORT

//Choose ONE of the two following
#define THREADPOOL4
//#define NOTTHREADED4

// Leave this ON
#define THREADPOOL5

//#define THREADED7

class feature {
protected:
    std::istream* _is;
    std::ostream* _os;
    workspace* _ws;
public:
    virtual std::string cmdlineoption() {return "";};
    virtual std::string cmdlineoptionlong() { return "";};
    virtual void listoptions() {
        *_os << "Options for command line \"-" << cmdlineoption() << "\": " << cmdlineoptionlong() << "\n";
    }
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

    void listoptions() override {}; // don't publicize _sandboxfeature
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

        /*
        std::vector<std::pair<std::string,std::string>> res = cmdlineparseiterationtwo(args);
        for (int i = 0; i < res.size(); ++i) {
            std::cout << res[i].first << " === " << res[i].second << "\n";
        }*/

        int dim1 = 3;
        int dim2 = 10;
        int numberofsubsets = nchoosek(dim2,dim1);
        //int* subsets = (int*)malloc(numberofsubsets*dim1*sizeof(int));
        std::vector<int> subsets {};
        enumsizedsubsets(0,dim1,nullptr,0,dim2,&subsets);

        for (int n = 0; n < numberofsubsets*dim1; n+=dim1) {
            std::cout << "n " << n <<", ";

            for (int i = 0; i < dim1; ++i ) {
                std::cout << subsets[n+i] << ", ";
            }
            std::cout << "\n";
        }
        //free(subsets);


    }

};


class userguidefeature : public feature {
public:
    std::vector<feature*> featureslist {};
    std::string cmdlineoption() {return "h";};
    std::string cmdlineoptionlong() { return "userguide";};
    void listoptions() override {
        feature::listoptions();
        *_os << "  " << "(this feature)\n";
    }
    virtual void execute(std::vector<std::string> args) {
        feature::execute(args);
        for (int n = 0; n < featureslist.size(); ++n) {
            featureslist[n]->listoptions();
        }
    };
    userguidefeature( std::istream* is, std::ostream* os, workspace* ws ) : feature(is,os,ws) {}

};


class clearworkspacefeature : public feature {
public:
    std::string cmdlineoption() {return "c";};
    std::string cmdlineoptionlong() { return "clear";};
    void listoptions() override {
        feature::listoptions();
        *_os << "  " << "(no options)" << " \t\t clears the workspace\n";
    }
    virtual void execute(std::vector<std::string> args) {
        feature::execute(args);
        for (int n = 0; n < _ws->items.size(); ++n) {
            _ws->items[n]->freemem();  // figure out if this is a memory leak
            delete _ws->items[n];
        }
        _ws->items.clear();

    };
    clearworkspacefeature( std::istream* is, std::ostream* os, workspace* ws ) : feature(is,os,ws) {}

};

class abstractrandomgraphsfeature : public feature {
protected:
    std::vector<abstractrandomgraph*> rgs {};

public:
    virtual void listoptions() override {
        feature::listoptions();
    }
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
    void listoptions() override {
        abstractrandomgraphsfeature::listoptions();
        *_os << "\t" << "<dim>: \t\t\t\t dimension of the graph\n";
        *_os << "\t" << "<edgecount>: \t\t edge count, where probability is <edgecount>/maxedges, maxedges = (dim-choose-2)\n";
        *_os << "\t" << "<outof>: \t\t\t how many samples to take\n";
        *_os << "\t" << "<randomalgorithm>:\t which algorithm to use, standard options are:\n";
        for (int n = 0; n < rgs.size(); ++n) {
            *_os << "\t\t\"" << rgs[n]->shortname() << "\": " << rgs[n]->name << "\n";
        }
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

    void listoptions() override {
        abstractrandomgraphsfeature::listoptions();
        *_os << "\t" << "<dim>: \t\t\t\t dimension of the graph\n";
        *_os << "\t" << "<edgecount>: \t\t edge count, where probability is <edgecount>/maxedges, maxedges = (dim-choose-2)\n";
        *_os << "\t" << "<count>: \t\t\t how many random graphs to output to the workspace\n";
        *_os << "\t" << "<randomalgorithm>:\t which algorithm to use, standard options are r1,...,r5:\n";
        for (int n = 0; n < rgs.size(); ++n) {
            *_os << "\t\t\"" << rgs[n]->shortname() << "\": " << rgs[n]->name << "\n";
        }
    }

    randomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractrandomgraphsfeature( is, os, ws) {
    }

    void execute(std::vector<std::string> args) override {
        abstractrandomgraphsfeature::execute(args);
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


        std::vector<graphtype*> gv {};
        gv.resize(cnt);
#ifdef THREADED7
        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        // Note: MUST play safe in the abstractrandomgraph class so it won't contend

        std::vector<std::future<std::vector<graphtype>>> t {};
        t.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            t[j] = std::async(&randomgraphs,rgs[rgidx],dim,edgecnt,int(float(cnt)/float(thread_count)));
        }
        std::vector<std::vector<graphtype>> gvv {};
        gvv.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            gvv[j] = t[j].get();
        }
        for (int j = 0; j < thread_count; ++j) {
            for (int i = 0; i < gvv[j].size(); ++i) {
                gv.push_back(gvv[j][i]);
            }

        }
#else

        //gv = randomgraphs(rgs[rgidx],dim,edgecnt,cnt);

#endif


/*
        std::vector<std::future<neighbors>> t {};
        t.resize(gv.size());
        for (int j = 0; j < gv.size(); ++j) {
            t[j] = std::async(&computeneighborslist,gv[j]);
        }
        std::vector<neighbors> nv {};
        nv.resize(gv.size());
        for (int j = 0; j < gv.size(); ++j) {
            nv[j] = t[j].get();
        }
        for (int j = 0; j < gv.size(); ++j) {
            auto wi = new graphitem;
            wi->g.dim = gv[j].dim;
            wi->g.adjacencymatrix = gv[j].adjacencymatrix;
            wi->ns = nv[j];
            wi->name = _ws->getuniquename(wi->classname);
            _ws->items.push_back(wi);
        }


*/
        gv = randomgraphs(rgs[rgidx],dim,edgecnt,cnt);

        /*
        auto starttime = std::chrono::high_resolution_clock::now();
        std::vector<std::chrono::time_point<std::chrono::system_clock>> starray {};
*/
        //int s = _ws->items.size();

        //_ws->items.resize(s + cnt);

        std::vector<std::string> vertexlabels {};
        char ch = 'a';
        int idx = 1;
        for (int i = 0; i < dim; ++i) {
            if (idx == 1)
                vertexlabels.push_back( std::string{ ch++ });
            else
                vertexlabels.push_back(std::string{ch++} + std::to_string(idx));
            if (ch == 'z') {
                idx++;
                ch = 'a';
            }
        }

        for (int i = 0; i < cnt; ++i) {

            //starray.push_back(std::chrono::high_resolution_clock::now());


            auto wi = new graphitem;
            wi->ns = new neighbors(gv[i]);
            wi->g = gv[i];
            wi->name = _ws->getuniquename(wi->classname);
            gv[i]->vertexlabels = vertexlabels;
            _ws->items.push_back( wi );
        }
/*        for (int i = 0; i < cnt; ++i) {
            _ws->items[s+i]->name = _ws->getuniquename(_ws->items[s+i]->classname);
        }*/
/*        for (int i =1; i < starray.size(); ++i) {
            std::cout << (starray[i]- starray[i-1])/1000000.0 << ",";
        }
        std::cout << "\n";*/
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

    void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "<filename>: \t\t output filename, or \"std::cout\"\n";
        *_os << "\t" << "o=<filename>: \t\t output filename, or \"std::cout\"\n";
        *_os << "\t" << "i=<filename>: \t\t input filename (use to prepackage verbosity commands)\n";
        *_os << "\t" << "<verbosityname>: \t any levels can be listed, delimited by spaces;\n";
        *_os << "\t\t\t\t\t\t in addition to what's optionally in the input file\n";
        // eventually do a loop which calls on each verbosity option to identify itself
    }


    verbosityfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "r";
        //cmdlineoptionlong = "samplerandomgraphs";
    }

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
        std::string ifname {};
        bool ofsrequiresclose = false;
        verbositylevel = "";
        std::vector<std::pair<std::string,std::string>> cmdlineoptions = cmdlineparseiterationtwo(args);
        for (int n = 0; n < cmdlineoptions.size(); ++n) {
            if (cmdlineoptions[n].first == "o") {
                ofname = cmdlineoptions[n].second;
                continue;
            }
            if (cmdlineoptions[n].first == "i") {
                ifname = cmdlineoptions[n].second;
                std::ifstream infile(ifname);
                if (infile.good()) {
                    std::ifstream ifs;
                    ifs.open(ifname, std::fstream::in );
                    while (!ifs.eof()) {
                        std::string tmp {};
                        ifs >> tmp;
                        if (verbositylevel != "") {
                            verbositylevel += " ";
                        }
                        verbositylevel += tmp;
                    }
                    ifs.close();
                    //std::cout << "Verbosity: " << verbositylevel << "\n";
                } else {
                    std::cout << "Couldn't open file for reading " << ifname << "\n";
                }
                continue;
            }
            if (cmdlineoptions[n].first == "default") {
                if (verbositylevel != "") {
                    verbositylevel += " ";
                }
                verbositylevel += cmdlineoptions[n].second;
                continue;
            }
        }
        if (verbositylevel == "")
            verbositylevel = VERBOSE_DEFAULT;
        if (ofname != "") {
            if (ofname != "std::cout") {
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
            } else {
                _os = &std::cout;
            }
        } else
            _os = &std::cout;

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


class writegraphsfeature: public feature {
public:
    std::string ofname {};
    std::string cmdlineoption() {
        return "g";
    }
    std::string cmdlineoptionlong() {
        return "writegraphs";
    }

    void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "<filename>: \t writes machine-readable graphs to output filename \n";
        *_os << "\t\t\t\t\t or \"std::cout\"\n";
        *_os << "\t" << "\"o=<filename>\":" << "\t same as just a filename\n";
        *_os << "\t" << "\"append\":" << "\t\t Appends to an existing file (default)\n";
        *_os << "\t" << "\"overwrite\":" << "\t Overwrites if file exists already\n";
    }


    writegraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {}

    void execute(std::vector<std::string> args) override {
        std::ofstream ofs;
        bool ofsrequiresclose = false;
        bool append = true;
        bool overwrite = false;
        std::vector<std::pair<std::string,std::string>> cmdlineoptions = cmdlineparseiterationtwo(args);
        for (int n = 0; n < cmdlineoptions.size(); ++n) {
            if (cmdlineoptions[n].first == "default" && cmdlineoptions[n].second == "append") {
                append = true;
                overwrite = false;
                continue;
            }
            if (cmdlineoptions[n].first == "default" && cmdlineoptions[n].second == "overwrite") {
                append = false;
                overwrite = true;
                continue;
            }
            if (cmdlineoptions[n].first == "o") {
                ofname = cmdlineoptions[n].second;
                continue;
            }
            if (cmdlineoptions[n].first == "default") {
                ofname = cmdlineoptions[n].second;
                continue;
            }
        }
        if (ofname != "") {
            if (ofname != "std::cout") {
                std::ifstream infile(ofname);
                if (infile.good() && !(append || overwrite)) {
                    std::cout << "Graph machine-readable output file " << ofname << " already exists; use option \"append\" to append it or option \"overwrite\" to overwrite.\n";
                    return;
                }
                if (append)
                    ofs.open(ofname,  std::ios::app);
                if (!append && overwrite)
                    ofs.open(ofname, std::ios::trunc);
                if (!ofs) {
                    std::cout << "Couldn't open file for writing \n";
                    return;
                }
                _os = &ofs;
                ofsrequiresclose = true;
            } else {
                _os = &std::cout;
            }
        } else
            _os = &std::cout;

        bool first = true;
        for (auto wi : _ws->items) {
            if (wi->classname == "GRAPH") {
                auto gi = (graphitem*)wi;
                if (!first)
                    *_os << "\n";
                gi->osmachinereadablegraph(*_os);
                first = false;
            }
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

    void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "<filename>: \t input filename, or \"std::cin\"; <filename> can be repeated any number of times\n";
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
                if (gi->name == "") {
                    gi->name = _ws->getuniquename(gi->classname);
                }
                _ws->items.push_back(gi);
                gi = new graphitem();
            }
            delete gi;
        }
    }

};


inline int partition( std::vector<int> &arr, int start, int end, std::vector<neighbors*>* nslist, std::vector<FP*>* fpslist ) {
    int pivot = arr[start];
    int count = 0;
    for (int i = start+1;i <= end; i++) {
        if (FPcmp((*nslist)[arr[i]],(*nslist)[pivot],(*fpslist)[arr[i]],(*fpslist)[pivot]) >= 0) {
            count++;
        }
    }

    int pivotIndex = start + count;
    std::swap(arr[pivotIndex],arr[start]);

    int i = start;
    int j = end;
    while (i < pivotIndex && j > pivotIndex) {
        while (FPcmp((*nslist)[arr[i]],(*nslist)[pivot],(*fpslist)[arr[i]],(*fpslist)[pivot]) >= 0) {
            i++;
        }
        while (FPcmp((*nslist)[arr[j]],(*nslist)[pivot],(*fpslist)[arr[j]],(*fpslist)[pivot]) < 0) {
            j--;
        }
        if (i < pivotIndex && j > pivotIndex) {
            std::swap(arr[i++],arr[j--]);
        }
    }
    return pivotIndex;
}

inline void quickSort( std::vector<int> &arr, int start, int end,std::vector<neighbors*>* nslist, std::vector<FP*>* fpslist ) {

    if (start >= end)
        return;

    int p = partition(arr,start,end,nslist,fpslist);

    quickSort(arr, start, p-1,nslist,fpslist);
    quickSort(arr, p+1, end,nslist,fpslist);
}

class cmpfingerprintsfeature: public feature {
public:
    std::string cmdlineoption() { return "f"; }
    std::string cmdlineoptionlong() { return "cmpfingerprints"; }
    cmpfingerprintsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "f";
        //cmdlineoptionlong = "cmpfingerprints";
    };

    void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "\"all\": \t\t computes fingerprints and sorts ALL graphs found on the workspace\n";
        *_os << "\t" << "\"c=<n>\": \t (not implemented yet) computes fingerprints for the last n graphs on the workspace\n";
    }

    void execute(std::vector<std::string> args) override {
        std::vector<int> items {}; // a list of indices within workspace of the graph items to FP and sort


        bool takeallgraphitems = true;
        int numofitemstotake = 0;

        if (args.size() > 1) {
            if (args[1] == CMDLINE_ALL)
                takeallgraphitems = true;
            else {
                takeallgraphitems = false;
                numofitemstotake = 2;   // to do: code the regex of say "c=3" to mean go back three graphs on the workspace
                // also code a list of indices in reverse chrono order
            }
        }

        if (!takeallgraphitems) {
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
        } else {
            for (int n = 0; n < _ws->items.size(); ++n) {
                if (_ws->items[n]->classname == "GRAPH") {
                    items.push_back(n);
                }
            }
        }
#ifdef THREADPOOL5


        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        std::vector<std::future<void>> t {};
        t.resize(items.size());
#endif

        auto wi = new cmpfingerprintsitem;
        wi->nslist.resize(items.size());
        wi->fpslist.resize(items.size());
        wi->glist.resize(items.size());
        wi->gnames.resize(items.size());

        for (int i = 0; i < items.size(); ++i) {
            auto gi = (graphitem*)(_ws->items[items[i]]);
            //std::cout << idx << ": " << _ws->items[idx]->classname << "\n";
            //gi->ositem(*_os,11741730);

            wi->fpslist[i] = (FP*)malloc(sizeof(FP));
            FP* fpsptr = (FP*)malloc(gi->g->dim * sizeof(FP));
            wi->fpslist[i]->ns = fpsptr;
            wi->fpslist[i]->parent = nullptr;
            wi->fpslist[i]->nscnt = gi->g->dim;
            wi->fpslist[i]->invert = false;
            wi->nslist[i] = gi->ns;

            //osadjacencymatrix(std::cout, gi->g);
            //std::cout << "(" << i << ")\n";


            for (vertextype n = 0; n < gi->g->dim; ++n) {
                fpsptr[n].v = n;
                fpsptr[n].ns = nullptr;
                fpsptr[n].nscnt = 0;
                fpsptr[n].parent = nullptr;
                fpsptr[n].invert = wi->nslist[i]->degrees[n] >= (gi->g->dim+1)/2;
            }

#ifdef THREADPOOL5

            //wi->nslist[i] = gi->ns;
            t[i] = std::async(&takefingerprint,(wi->nslist[i]),fpsptr,gi->g->dim);
            wi->glist[i] = gi->g;
            wi->gnames[i] = gi->name;

#endif

#ifndef THREADPOOL5
            takefingerprint(gi->ns,wi->fpslist[i]->ns,gi->g->dim);
            //wi->nslist[i] = gi->ns;
            wi->glist[i] = gi->g;
            wi->gnames[i] = gi->name;

#endif
        }
#ifdef THREADPOOL5
        for (int m = 0; m < items.size(); ++m) {
            t[m].get();
            //t[m].detach();
            //threadgm[m] = t[m].get();
        }
#endif
        /*
        for (int m = 0; m < items.size(); ++m)
        {
            sortneighbors(wi->nslist[m],wi->fpslist[m]->ns,wi->fpslist[m]->nscnt);
        }*/
        bool changed = true;
        bool overallres = true;
        bool overallresdont = true;
        int overallmatchcount = 0;
        wi->sorted.resize(items.size());
        for (int i = 0; i < items.size(); ++i) {
            wi->sorted[i] = i;
        }
        std::vector<int> res {};
        res.resize(items.size());

#ifdef QUICKSORT
        quickSort( wi->sorted,0,wi->sorted.size()-1, &wi->nslist, &wi->fpslist);

        for (int i = 0; i < items.size()-1; ++i) {
            res[i] = FPcmp(wi->nslist[wi->sorted[i]],wi->nslist[wi->sorted[i+1]],wi->fpslist[wi->sorted[i]],wi->fpslist[wi->sorted[i+1]]);
            if (res[i] != 0)
                overallres = false;
            else {
                overallresdont = false;
                ++overallmatchcount;
            }
        }

#endif
#ifndef QUICKSORT

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
                    overallmatchcount = 0;
                }
                if (res[i] != 0)
                    overallres = false;
                else {
                    overallresdont = false;
                    ++overallmatchcount;
                }
            }
        }
#endif

        wi->fingerprintsmatch = overallres;
        wi->fingerprintsdontmatch = overallresdont;
        wi->overallmatchcount = overallmatchcount;
        wi->res = res;
        wi->name = _ws->getuniquename(wi->classname);

        for (auto i = 0; i < items.size(); ++i) {
            if (_ws->items[items[wi->sorted[i]]]->classname != "GRAPH") {
                std::cout << "Error in cmpfingerprintsfeature";
                return;
            }
            auto egi = (graphitem*)_ws->items[items[wi->sorted[i]]];
            egi->intitems.push_back(new fpoutcome(wi->fpslist[i],egi,wi->res[i]));
        }
        std::vector<workitems*> tmpws {};
        tmpws.resize(_ws->items.size());
        for (auto i=0; i < items.size();++i) {
            tmpws[items[i]] = _ws->items[items[wi->sorted[i]]];
        }
        for (auto i =0; i < items.size(); ++i) {
            _ws->items[items[i]] = tmpws[items[i]];
        }

        _ws->items.push_back(wi);


    }
};


class enumisomorphismsfeature: public feature {
public:
    std::string cmdlineoption() { return "i"; }
    std::string cmdlineoptionlong() { return "enumisomorphisms"; }
    enumisomorphismsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {
        //cmdlineoption = "i";
        //cmdlineoptionlong = "enumisomorphisms";
    };

    void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "\"" << CMDLINE_ALL << "\": \t\t\t computes automorphisms for ALL graphs found on the workspace\n";
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTED << "\": \t\t computes automorphisms once for each fingerprint-equivalent class\n";
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f\"\n";
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTEDVERIFY << "\":  computes automorphisms once within each fingerprint-equivalence class\n";
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f\"\n";
        *_os << "\t" << "\"c=<n>\": \t\t (not implemented yet) computes automorphisms for the last n graphs on the workspace\n";
        *_os << "\t" << "\t\t\t\t (the default is to compute isomorphisms between the last two graphs found on the workspace\n";
        *_os << "\t" << "\t\t\t\t or if only one is found, to compute its automorphisms)\n";
    }


    void execute(std::vector<std::string> args) override {
        std::vector<int> items {}; // a list of indices within workspace of the graph items to FP and sort


        bool takeallgraphitems = false;
        bool computeautomorphisms = false;
        int numofitemstotake = 2;
        bool sortedbool = false;
        bool sortedverify = false;

        if (args.size() > 1) {
            if (args[1] == CMDLINE_ALL) {
                takeallgraphitems = true;
                computeautomorphisms = true;
            } else {
                if (args[1] == CMDLINE_ENUMISOSSORTED) {
                    sortedbool = true;
                    numofitemstotake = 0;
                    takeallgraphitems = true;
                    computeautomorphisms = true;
                }
                else {
                    if (args[1] == CMDLINE_ENUMISOSSORTEDVERIFY) {
                        sortedverify = true;
                        numofitemstotake = 0;
                        takeallgraphitems = true;
                        computeautomorphisms = false;
                    } else {
                        takeallgraphitems = false;
                        numofitemstotake = 2;   // to do: code the regex of say "c=3" to mean go back three graphs on the workspace
                        // also code a list of indices in reverse chrono order
                    }
                }
            }
        }

        if (!takeallgraphitems) {
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
        } else {
            for (int n = 0; n < _ws->items.size(); ++n) {
                if (_ws->items[n]->classname == "GRAPH") {
                    items.push_back(n);
                }
            }
        }
        if (!sortedbool && !sortedverify)
            if (items.size() == 0) {
                std::cout << "No graphs available to enum isomorphisms\n";
                return;
            }

        std::vector<neighbors*> nslist {};
        nslist.resize(items.size());
        // code this to run takefingerprint only once if graph 1 = graph 2

        for (int i = 0; i < items.size(); ++i)
        {
            auto gi = (graphitem*)(_ws->items[items[i]]);
            nslist[i] = gi->ns;
        }
        if (computeautomorphisms || (items.size() == 1)) {





#ifdef THREADPOOL4


            //unsigned const thread_count = std::thread::hardware_concurrency();
            //unsigned const thread_count = 1;

            std::vector<int> eqclass {};
            if (sortedbool) {
                if (items.size()==0) {
                    std::cout << "No graphs to enumerate isomorphisms over\n";
                    return;
                }

                eqclass.push_back(0);
                for (int m = 0; m < items.size()-1; ++m) {
                    auto gi = (graphitem*)_ws->items[items[m]];
                    bool found = false;
                    for (int r = 0; !found && (r < gi->intitems.size()); ++r) {
                        if (gi->intitems[r]->name() == "FP") {
                            auto fpo = (fpoutcome*)gi->intitems[r];
                            if (fpo->value == 1) {
                                eqclass.push_back(m+1);
                                found = true;
                            }
                        }
                    }
                }
            } else {
                for (int m = 0; m < items.size(); ++m) {
                    eqclass.push_back(m);
                }
            }

            std::vector<std::future<std::vector<graphmorphism>*>> t {};
            t.resize(eqclass.size());
            for (int m = 0; m < eqclass.size(); ++m) {
                    t[m] = std::async(&enumisomorphisms,nslist[eqclass[m]],nslist[eqclass[m]]);
            }
            std::vector<std::vector<graphmorphism>*> threadgm {};
            threadgm.resize(eqclass.size());
            for (int m = 0; m < eqclass.size(); ++m) {
                //t[m].join();
                //t[m].detach();
                threadgm[m] = t[m].get();
            }
            for (int j=0; j < eqclass.size(); ++j) {
                auto wi = new enumisomorphismsitem;
                wi->g1 = nslist[eqclass[j]]->g;
                wi->g2 = nslist[eqclass[j]]->g;
                wi->gm = threadgm[j];
                wi->name = _ws->getuniquename(wi->classname);
                _ws->items.push_back(wi);

                //freefps(fpslist[j].ns,glist[j]->dim);
                //delete fpslist[j].ns;
                auto gi = (graphitem*)_ws->items[items[eqclass[j]]];
                gi->intitems.push_back(new automorphismsoutcome(gi,wi->gm->size()));
            }

#endif

#ifdef NOTTHREADED4

            for (int j = 0; j < items.size(); ++j) {
                auto wi = new enumisomorphismsitem;
                wi->gm = enumisomorphisms(nslist[j],nslist[j]);
                wi->name = _ws->getuniquename(wi->classname);
                _ws->items.push_back(wi);

                freefps(fpslist[j].ns,glist[j].dim);
                free(fpslist[j].ns);
            }

#endif

        } else {
            if (sortedverify) {
                std::vector<int> eqclass {};
                if (items.size()==0) {
                    std::cout << "No graphs to enumerate isomorphisms over\n";
                    return;
                }
                //eqclass.push_back(0);

                eqclass.push_back(0);
                for (int m = 0; m < items.size()-1; ++m) {
                    auto gi = (graphitem*)_ws->items[items[m]];
                    bool found = false;
                    for (int r = 0; !found && (r < gi->intitems.size()); ++r) {
                        if (gi->intitems[r]->name() == "FP") {
                            auto fpo = (fpoutcome*)gi->intitems[r];
                            if (fpo->value == 1) {
                                eqclass.push_back(m+1);
                                found = true;
                            }
                        }
                    }
                }

                std::vector<std::future<std::vector<graphmorphism>*>> t {};
                t.resize(items.size());
                int eqclassidx = 1;
                for (int m = 0; m < items.size()-1; ++m) {
                    if ((m+1) != eqclass[eqclassidx]) {
                        t[m] = std::async(&enumisomorphisms,nslist[m],nslist[m+1]);
                    } else
                        eqclassidx++;
                }
                std::vector<std::vector<graphmorphism>*> threadgm {};
                threadgm.resize(items.size());
                eqclassidx = 1;
                for (int m = 0; m < items.size()-1; ++m) {
                    //t[m].join();
                    //t[m].detach();
                    if ((m +1) != eqclass[eqclassidx])
                        threadgm[m] = t[m].get();
                    else
                        eqclassidx++;
                }
                eqclassidx = 1;
                for (int j=0; j < items.size()-1; ++j) {
                    if ((j+1) != eqclass[eqclassidx]) {
                        auto wi = new enumisomorphismsitem;
                        wi->g1 = nslist[j]->g;
                        wi->g2 = nslist[j+1]->g;
                        wi->gm = threadgm[j];
                        wi->name = _ws->getuniquename(wi->classname);
                        _ws->items.push_back(wi);

                        //freefps(fpslist[j].ns,glist[j]->dim);
                        //delete fpslist[j].ns;
                        auto gi1 = (graphitem*)_ws->items[items[j]];
                        auto gi2 = (graphitem*)_ws->items[items[j+1]];
                        gi1->intitems.push_back(new isomorphismsoutcome(gi2,gi1,wi->gm->size()));
                        gi2->intitems.push_back(new isomorphismsoutcome(gi1,gi2,wi->gm->size()));
                    } else
                        eqclassidx++;
                }
            } else {
                if (items.size() == 2) {
                    auto wi = new enumisomorphismsitem;
                    wi->g1 = nslist[0]->g;
                    wi->g1 = nslist[1]->g;
                    wi->gm = enumisomorphisms(nslist[0],nslist[1]);
                    wi->name = _ws->getuniquename(wi->classname);
                    auto gi = (graphitem*)_ws->items[items[0]];
                    gi->intitems.push_back(new automorphismsoutcome(gi,wi->gm->size()));
                    _ws->items.push_back(wi);
                }
            }
        }
    }

};

template<typename T>
class abstractcheckcriterionfeature : public feature {
protected:
    std::vector<abstractcriterion<T>*> crs {};
public:
    virtual void listoptions() override {
        feature::listoptions();
    }
    abstractcheckcriterionfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {

        // add any new criterion types to the list here...

        auto cr1 = new trianglefreecriterion();
        crs.push_back(cr1);
        // ...
    }

    ~abstractcheckcriterionfeature() {

        for (int i = 0; i < crs.size();++i) {
            delete crs[i];
        }
    }


};

class checkcriterionfeature : public abstractcheckcriterionfeature<bool> {
public:
    std::vector<abstractcriterion<bool>*> cs;
    std::string cmdlineoption() override { return "a"; }
    std::string cmdlineoptionlong() { return "checkcriteria"; }
    checkcriterionfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractcheckcriterionfeature( is, os, ws) {
        cs.clear();
    }

    void listoptions() override {
        abstractcheckcriterionfeature::listoptions();
        *_os << "\t" << "\"" << CMDLINE_ALL << "\": \t\t\t checks criteria for ALL graphs found on the workspace\n";
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTED << "\": \t\t checks criteria for each fingerprint-equivalent class\n";
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f\"\n";
        *_os << "\t" << "<criterion>:\t which criterion to use, standard options are:\n";
        for (int n = 0; n < crs.size(); ++n) {
            *_os << "\t\t\"" << crs[n]->shortname() << "\": " << crs[n]->name << "\n";
        }
    }

    void execute(std::vector<std::string> args) override {
        std::vector<int> items {}; // a list of indices within workspace of the graph items to FP and sort


        bool takeallgraphitems = false;
        int numofitemstotake = 1;
        bool sortedbool = false;
        bool sortedverify = false;


        for (int i = 1; i < args.size(); ++i) {
            if (args[i] == CMDLINE_ALL) {
                takeallgraphitems = true;
            } else {
                if (args[i] == CMDLINE_ENUMISOSSORTED) {
                    sortedbool = true;
                    takeallgraphitems = true;
                }
            }
            for (int n = 0; n < crs.size(); ++n) {
                if (args[i] == crs[n]->shortname())
                    cs.push_back(crs[n]);
            }
            if (cs.size()==0)
                cs.push_back(crs[0]);
        }

        if (!takeallgraphitems) {
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
        } else {
            for (int n = 0; n < _ws->items.size(); ++n) {
                if (_ws->items[n]->classname == "GRAPH") {
                    items.push_back(n);
                }
            }
        }

        if (!sortedbool && !sortedverify)
            if (items.size() == 0) {
                std::cout << "No graphs available to check criterion\n";
                return;
            }

        std::vector<neighbors*> nslist {};
        nslist.resize(items.size());
        std::vector<graphtype*> glist {};
        glist.resize(items.size());
        // code this to run takefingerprint only once if graph 1 = graph 2

        for (int i = 0; i < items.size(); ++i)
        {
            auto gi = (graphitem*)(_ws->items[items[i]]);
            nslist[i] = gi->ns;
            glist[i] = gi->g;
        }
        if (items.size() >= 1) {





#ifdef THREADPOOL4


            //unsigned const thread_count = std::thread::hardware_concurrency();
            //unsigned const thread_count = 1;






            std::vector<int> eqclass {};
            if (sortedbool) {
                if (items.size()==0) {
                    std::cout << "No graphs to check criterion over\n";
                    return;
                }
                //eqclass.push_back(0);



                eqclass.push_back(0);
                for (int m = 0; m < items.size()-1; ++m) {
                    auto gi = (graphitem*)_ws->items[items[m]];
                    bool found = false;
                    for (int r = 0; !found && (r < gi->intitems.size()); ++r) {
                        if (gi->intitems[r]->name() == "FP") {
                            auto fpo = (fpoutcome*)gi->intitems[r];
                            if (fpo->value == 1) {
                                eqclass.push_back(m+1);
                                found = true;
                            }
                        }
                    }
                }
            } else {
                for (int m = 0; m < items.size(); ++m) {
                    eqclass.push_back(m);
                }
            }


            for (int k = 0; k < cs.size(); ++k) {
                std::vector<std::future<bool>> t {};
                t.resize(eqclass.size());
                for (int m = 0; m < eqclass.size(); ++m) {
                    t[m] = std::async(&abstractcriterion<bool>::checkcriterion,cs[k],glist[eqclass[m]],nslist[eqclass[m]]);
                }
                std::vector<bool> threadbool {};
                threadbool.resize(eqclass.size());
                for (int m = 0; m < eqclass.size(); ++m) {
                    //t[m].join();
                    //t[m].detach();
                    threadbool[m] = t[m].get();
                }
                auto wi = new checkcriterionitem<bool>;

                wi->res.resize(eqclass.size());
                wi->fpslist = {};
                wi->glist.resize(eqclass.size());
                wi->sorted.resize(eqclass.size());
                wi->gnames.resize(eqclass.size());
                wi->nslist.resize(eqclass.size());
                for (int j=0; j < eqclass.size(); ++j) {
                    wi->res[j] = threadbool[j];
                    wi->glist[j] = glist[j];
                    wi->sorted[j] = j;
                    wi->nslist[j] = nslist[j];
                    auto gi = (graphitem*)_ws->items[items[eqclass[j]]];
                    gi->boolitems.push_back(new abstractcriterionoutcome<bool>(cs[k],gi,wi->res[j]));
                    wi->gnames[j] = gi->name;
                }
                wi->name = _ws->getuniquename(wi->classname);
                _ws->items.push_back(wi);

            }


#endif

#ifdef NOTTHREADED4

            for (int j = 0; j < items.size(); ++j) {
                auto wi = new enumisomorphismsitem;
                wi->gm = enumisomorphisms(nslist[j],nslist[j]);
                wi->name = _ws->getuniquename(wi->classname);
                _ws->items.push_back(wi);

                freefps(fpslist[j].ns,glist[j].dim);
                free(fpslist[j].ns);
            }

#endif

        }
    }
};


class checkembedcriteriafeature : public checkcriterionfeature {
protected:
public:
    std::string cmdlineoption() override { return "e"; }
    std::string cmdlineoptionlong() { return "checkembedscriteria"; }
    checkembedcriteriafeature( std::istream* is, std::ostream* os, workspace* ws ) : checkcriterionfeature( is, os, ws) {}
    ~checkembedcriteriafeature() {
        for (int i = 0; i < cs.size(); ++i)
            delete cs[i];
    }

    void listoptions() override {
        abstractcheckcriterionfeature::listoptions();
        *_os << "\t" << "\"" << CMDLINE_ALL << "\": \t\t\t checks embeds for ALL graphs found on the workspace\n";
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTED << "\": \t\t checks embeds for each fingerprint-equivalent class\n";
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f\"\n";
        *_os << "\t" << "f=<filename>:\t reads graphs from <filename> and uses them as the embeddings sought\n";
    }


    void execute(std::vector<std::string> args) {
        std::vector<std::pair<std::string,std::string>> parsedargs = cmdlineparseiterationtwo(args);

        for (int i = 0; i < parsedargs.size(); ++i) {
            if (parsedargs[i].first == "f") {

                std::ifstream ifs;
                std::istream* is = &std::cin;
                std::ostream* os = _os;
                std::string filename = parsedargs[i].second;
                if (filename == "std::cin" || filename == "") {
                    std::cout << "Using std::cin for input\n"; // recode so this is possible
                } else {
                    *_os << "Opening file " << filename << "\n";
                    ifs.open(filename);
                    if (!ifs) {
                        std::cout << "Couldn't open file for reading \n";
                        return;
                    }
                    is = &ifs;
                }

                graphitem* gi = new graphitem();
                while (gi->isitem(*is)) {
                    gi->name = _ws->getuniquename(gi->classname) + "FLAG";
                    _ws->items.push_back(gi);
                    cs.push_back(new embedscriterion(gi->ns));
                    gi = new graphitem();
                }
                delete gi;

            }
        }
        checkcriterionfeature::execute(args);
    }

};




class mantelstheoremfeature : public feature {
public:
    std::string cmdlineoption() { return "m"; }
    std::string cmdlineoptionlong() { return "mantels"; }
    mantelstheoremfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {}

    void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "<limitdim>: \t the dimension of graph being analyzed\n";
        *_os << "\t" << "<outof>: \t\t uses up to outof random graphs looking for edge-count-maximal triangle-free graphs\n";
    }


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
        //ei->useworkspaceg2 = true;
        ei->execute(args); // or send it a smaller subset of args
        delete ms;
        delete ei;
    }
};




#endif //FEATURE_H
