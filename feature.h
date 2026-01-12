//
// Created by peterglenn on 6/12/24.
//

#ifndef FEATURE_H
#define FEATURE_H

#include "config.h"

#ifdef FLAGCALC_CUDA
#include "cudameas.cu"
#endif

#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <wchar.h>

#include "asymp.h"
#include "graphio.h"
#include "graphoutcome.h"
#include "graphs.h"
#include "prob.h"
#include "workspace.h"
#include "mantel.h"
#include "thread_pool.cpp"
#include "ameas.h"
#include "cudaengine.cuh"
#include "meas.cu"

#ifdef FLAGCALCWITHPYTHON
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

#include "probsub.cu"
#include "math.h"

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

#define THREADCHECKCRITERION

#define COMMENTSTARTDELIMITER "/*"
#define COMMENTENDDELIMITER "*/"
#define COMMENTENTIRELINEDELIMITER "////"
// please don't use entire line delimiter for comments until rewriting readfromfile to include newlines;
// as it stands, using the entirelinedelimiter comments until the END line is reached



class feature {
protected:
    std::istream* _is;
    std::ostream* _os;
    workspace* _ws;
public:
    unsigned thread_count = std::thread::hardware_concurrency();
    virtual void setthread_count (const unsigned thread_countin) {
        thread_count = thread_countin;
    }
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
    void execute(std::vector<std::string> args) override
    {
        /*
                std::ifstream ifs;
                std::istream* is = &std::cin;
                std::ostream* os = _os;
                graphitem* gi1 = new graphitem();
                gi1->isitem(*is);
                trianglefreecriterion tf;
                std::cout << "Triangle free returns " << tf.checkcriterion(gi1->g,gi1->ns) << "\n";
        */


        std::string tmpstr {};
        std::vector<std::string> input {};
        std::cin >> tmpstr;
        while (tmpstr != "END") {
            input.push_back(tmpstr);
            std::cin >> tmpstr;
        }

        std::string f {};
        for (auto s : input)
            f += s + " ";


        const std::regex r {"\\[([[:alpha:]]\\w*)\\]"};
        std::vector<std::string> out {};
        for (std::sregex_iterator p(f.begin(),f.end(),r); p!=std::sregex_iterator{}; ++p)
        {
            out.push_back((*p)[1]);
        }
        for (auto o : out)
            std::cout << o << ", ";
        std::cout << " END\n";


        const std::regex r2 {"([[:alpha:]]\\w*)"};
        std::vector<std::string> out2 {};
        for (std::sregex_iterator p2(f.begin(),f.end(),r2); p2!=std::sregex_iterator{}; ++p2)
        {
            out2.push_back((*p2)[1]);
        }
        for (auto o : out2)
            std::cout << o << ", ";
        std::cout << " END2\n";


        // std::string replacement = "1";
        // std::string result = std::regex_replace(f,r,replacement);
        // std::cout << result << "\n";




        //auto cgs = new completegraphstyle();

        //auto vgs = new verticesgraphstyle();
        //auto ngs = new negationgraphstyle();
        //std::string outstr {};
        //bool out = cgs->applies(inputstr,&outstr);
        //std::cout << "out = " << out << ", outstr = " << outstr << "\n";

        //delete ngs;
        //delete cgs;

        // auto g = igraphstyle(input);
        // auto ns = new neighbors(g);
        // osadjacencymatrix(std::cout,g);
        // osneighbors(std::cout,ns);

//        auto tc = new forestcriterion();
//        std::cout << "tree criterion == " << tc->takemeasure(g,ns) << "\n";

//        auto cm = new connectedmeasure();
//        std::cout << "connected measure == " << cm->takemeasure(g,ns) << "\n";
//        auto cc = new connectedcriterion();
//        std::cout << "connected criterion == " << cc->takemeasure(g,ns) << "\n";

        // auto rm = new radiusmeasure();
        // auto tmp = rm->takemeasure(g,ns);
        // std::cout << "radius measure == " << tmp << "\n";

        //auto cm = new circumferencemeasure;
        // auto tmp = cm->takemeasure(g,ns);

        // auto dm = new diametermeasure;
        // auto tmp = dm->takemeasure(g,ns);
        // std::cout << "diameter measure == " << tmp << "\n";

        //delete cm;
        // std::string f {};
        // for (auto s : input)
            // f += s + " ";
        // std::vector<double> literals = {0,1,5,10,-25};
        // formulaclass* fout = parseformula(f); // don't use function names yet
        // if (fout != nullptr)
            // std::cout << "Result: " << evalformula(*fout,literals) << "\n";
        // else
            // std::cout << "Null result\n";
        // delete fout;

        // delete rm;
        // delete ns;
        // delete g;
//        delete tc;
        // delete cm;
        // delete cc;

        /*
        std::vector<std::pair<std::string,std::string>> res = cmdlineparseiterationtwo(args);
        for (int i = 0; i < res.size(); ++i) {
            std::cout << res[i].first << " === " << res[i].second << "\n";
            std::vector<std::pair<std::string,std::vector<std::string>>> arg = cmdlineparseiterationthree(res[i].second);
            for (int j = 0; j < arg.size(); ++j) {
                std::cout << "\t" << arg[j].first << " : ";
                for (auto s : arg[j].second) {
                    std::cout << s << ", " ;
                }
                std::cout << "\b\b\n";
            }
            std::cout << "\n";
        }
*/

        /*
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
*/
/*
        std::string sentence = "((NOT (3 AND (2 OR 5))) AND ((NOT -4) AND 4))";
        std::vector<bool> variables = {true,true,true,false,true,false};
        //sentence = res[0].second;
        logicalsentence ls = parsesentence(sentence);
        std::cout << "Result: " << sentence<< " evals to " << evalsentence(ls,variables) << "\n";
*/
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

// in addition to feature->thread_count, some quantifiermanager in particular, need thread_count
inline unsigned global_thread_count = std::thread::hardware_concurrency();

class threadsfeature : public feature {
public:
    std::vector<feature*> featureslist {};
    std::string cmdlineoption() {return "j";};
    std::string cmdlineoptionlong() {return "threads";};
    void listoptions() override {
        feature::listoptions();
        *_os << "  <n>" << " global set thread count" << " \t\t \n";
        *_os << "  p=<f>" << " global set thread portion (0 < f <= 1)" << " \t\t \n";
    }
    void execute(std::vector<std::string> args) {
        std::vector<std::pair<std::string,std::string>> cmdlineoptions = cmdlineparseiterationtwo(args);
        int cnt = 1;
        for (int n = 0; n < cmdlineoptions.size(); ++n) {
            if (cmdlineoptions[n].first == "p") {
                double f = stod(cmdlineoptions[n].second);
                double availthreads = (double)std::thread::hardware_concurrency();
                if (0 < f && f <= 1)
                    cnt = (int)(availthreads * f);
                else
                    cnt = 1;
                continue;
            }
            if (cmdlineoptions[n].first == "default") {
                cnt = stoi(cmdlineoptions[n].second);
                continue;
            }
        }
        for (auto f : featureslist)
            f->setthread_count(cnt);
        global_thread_count = cnt;
        *_os << "Threads globally: " << cnt << "\n";
    }
    threadsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature(is,os,ws) {}

};

class abstractrandomgraphsfeature : public feature {
protected:
    std::vector<abstractparameterizedrandomgraph*> rgs {};

public:
    virtual void listoptions() override {
        feature::listoptions();
    }
    abstractrandomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {

        // add any new abstractrandomgraph types to the list here...

        auto rg1 = new legacyrandomgraph<legacystdrandomgraph>();
        auto rg2 = new legacyrandomgraph<legacyrandomgraphonnedges>();
        auto rg3 = new legacyrandomgraph<legacyrandomconnectedgraph>();
        auto rg4 = new legacyrandomgraph<legacyrandomconnectedgraphfixededgecnt>();
        auto rg5 = new legacyrandomgraph<legacyweightedrandomconnectedgraph>();
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


class legacysamplerandomgraphsfeature : public abstractrandomgraphsfeature {
public:
    double percent = -1;
    std::string cmdlineoption() {
        return "L";
    }
    std::string cmdlineoptionlong() {
        return "legacysamplerandomgraphsforpairwiseequiv";
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

    legacysamplerandomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractrandomgraphsfeature( is, os, ws) {}

    void execute(std::vector<std::string> args) override {
        int outof = 1000;
        int dim = 5;
        double edgecnt = dim*(dim-1)/4.0;
        //std::vector<abstractparameterizedrandomgraph> rs {};
        int rgsidx = 0;
        std::vector<std::string> rgparams {};

        std::vector<std::pair<std::string,std::string>> parsedargs = cmdlineparseiterationtwo(args);
        if (parsedargs.size() >= 1 && parsedargs[0].first == "default" && is_number(parsedargs[0].second)) {
            dim = std::stoi(parsedargs[0].second);
        }
        if (parsedargs.size() >= 2 && parsedargs[1].first == "default" && is_number(parsedargs[1].second)) {
            edgecnt = std::stof(parsedargs[1].second);
        }
        if (parsedargs.size() >= 3 && parsedargs[2].first == "default" && is_number(parsedargs[2].second)) {
            outof = std::stoi(parsedargs[2].second);
        }

        for (int i = 0; i < parsedargs.size(); ++i) {
            if (parsedargs[i].first == "default" || parsedargs[i].first == "r") {
                std::vector<std::pair<std::string,std::vector<std::string>>> parsedargs2 = cmdlineparseiterationthree(parsedargs[i].second);
                for (int k = 0; k < parsedargs2.size(); ++k) {
                    for (int l = 0; l < rgs.size(); ++l) {
                        if (parsedargs2[k].first == rgs[l]->shortname()) {
                            rgsidx = l;
                            rgparams = parsedargs2[k].second;
                            //for (int k = 0; k < rgparams.size(); ++k)
                            //    std::cout << "rgparam " << rgparams[k] << ", ";
                            //std::cout << "\n";

                        }
                    }
                }
            }
        }

        if (rgparams.size() > 0 && is_number(rgparams[0]))
            dim = stoi(rgparams[0]);
        if (rgparams.size() > 1) // what is function to check if double
            edgecnt = std::stof(rgparams[1]);
        if (rgparams.size() > 2 && is_number(rgparams[2]))
            outof = stoi(rgparams[2]);
        //if (legacyrandomgraph<abstractrandomgraph>* larg = dynamic_cast<legacyrandomgraph<abstractrandomgraph>*>(rgs[rgsidx])) {
        //    std::cout << "LEGACY ARG\n";
        if (rgparams.empty()) {
            rgparams.clear();
            rgparams.resize(3);
            rgparams[0] = std::to_string(dim);
            rgparams[1] = std::to_string(edgecnt);
            rgparams[2] = std::to_string(outof);
        }
        rgs[rgsidx]->setparams(rgparams);


        // unsigned const thread_count = 1;

        int cnt = 0;
        const double section = double(outof) / double(thread_count);
        std::vector<std::future<int>> t;
        t.resize(thread_count);
        for (int m = 0; m < thread_count; ++m) {
            const int startidx = int(m*section);
            const int stopidx = int((m+1.0)*section);
            t[m] = std::async(&samplematchingrandomgraphs,rgs[rgsidx],dim,edgecnt,stopidx-startidx);
        }
        for (int m = 0; m < thread_count; ++m)
        {
            cnt += t[m].get();
        }



        // int cnt = samplematchingrandomgraphs(rgs[rgsidx],dim,edgecnt, outof);
            // --- yet a third functionality: randomly range over connected graphs (however, the algorithm should be checked for the right sense of "randomness"
            // note the simple check of starting with a vertex, recursively obtaining sets of neighbors, then checking that all
            // vertices are obtained, is rather efficient too.
            // Note also this definition of "randomness" is not correct: for instance, on a graph on three vertices, it doesn't run all the way
            //  up to and including three edges; it stops as soon as the graph is connected, i.e. at two vertices.



        auto wi = new legacysamplerandommatchinggraphsitem;
        wi->dim = dim;
        wi->cnt = cnt;
        wi->outof = outof;
        wi->name = _ws->getuniquename(wi->classname);

        wi-> percent = ((double)wi->cnt / (double)wi->outof);
        wi->rgname = rgs[rgsidx]->name;
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
        *_os << "\t" << "p=<f>:\t percentage in place of <edgecount>, 0<=f<=1 (must be in position of <edgecount>: second of the three inputs\n";
        *_os << "\t" << "<randomalgorithm>:\t which algorithm to use, standard options are r1,...,r5:\n";
        for (int n = 0; n < rgs.size(); ++n) {
            *_os << "\t\t\"" << rgs[n]->shortname() << "\": " << rgs[n]->name << "\n";
        }
    }

    randomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractrandomgraphsfeature( is, os, ws) {}

    void execute(std::vector<std::string> args) override {
        abstractrandomgraphsfeature::execute(args);
        int dim = 5;
        //std::vector<abstractparameterizedrandomgraph> rs {};
        int rgsidx = 0;
        std::vector<std::string> rgparams {};

        std::vector<std::pair<std::string,std::string>> parsedargs = cmdlineparseiterationtwo(args);
        if (parsedargs.size() >= 1 && parsedargs[0].first == "default" && is_number(parsedargs[0].second)) {
            dim = std::stoi(parsedargs[0].second);
        }
        double edgecnt = dim*(dim-1)/4.0;
        if (parsedargs.size() >= 2 && parsedargs[1].first == "default" && is_real(parsedargs[1].second)) {
            edgecnt = std::stof(parsedargs[1].second);
        }
        long int cnt = 100; // the default count when count is omitted
        if (parsedargs.size() >= 3 && parsedargs[2].first == "default" && is_number(parsedargs[2].second)) {
            cnt = std::stoi(parsedargs[2].second);
        }

        for (int i = 0; i < parsedargs.size(); ++i) {
            if (parsedargs[i].first == "default" || parsedargs[i].first == "r") {
                std::vector<std::pair<std::string,std::vector<std::string>>> parsedargs2 = cmdlineparseiterationthree(parsedargs[i].second);
                for (int k = 0; k < parsedargs2.size(); ++k) {
                    for (int l = 0; l < rgs.size(); ++l) {
                        if (parsedargs2[k].first == rgs[l]->shortname()) {
                            rgsidx = l;
                            rgparams = parsedargs2[k].second;
                            //for (int k = 0; k < rgparams.size(); ++k)
                            //    std::cout << "rgparam " << rgparams[k] << ", ";
                            //std::cout << "\n";

                        }
                    }
                }
            }
        }

        if (rgparams.size() > 0 && is_number(rgparams[0]))
            dim = stoi(rgparams[0]);
        if (rgparams.size() > 1) // what is function to check if double
            edgecnt = std::stof(rgparams[1]);
        if (rgparams.size() > 2 && is_number(rgparams[2]))
            cnt = stoi(rgparams[2]);
        for (auto p : parsedargs) {
            if (p.first == "p") {
                double f = std::stof(p.second);
                if (f >= 0 && f <= 1)
                    edgecnt = (long int)(std::stof(p.second) * nchoosek(dim,2));
            }
        }

        //if (legacyrandomgraph<abstractrandomgraph>* larg = dynamic_cast<legacyrandomgraph<abstractrandomgraph>*>(rgs[rgsidx])) {
        //    std::cout << "LEGACY ARG\n";
        if (rgparams.empty()) {
            rgparams.clear();
            rgparams.resize(3);
            rgparams[0] = std::to_string(dim);
            rgparams[1] = std::to_string(edgecnt);
            rgparams[2] = std::to_string(cnt);
        }
        rgs[rgsidx]->setparams(rgparams);




        std::vector<graphtype*> gv {};
        gv.resize(cnt);
#ifdef THREADED7
        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        // Note: MUST play safe in the abstractrandomgraph class so it won't contend

        std::vector<std::future<std::vector<graphtype>>> t {};
        t.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            t[j] = std::async(&randomgraphs,rgs[rgidx],dim,edgecnt,int(double(cnt)/double(thread_count)));
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

        rgs[rgsidx]->thread_count = thread_count;
        gv = randomgraphs(rgs[rgsidx],dim,edgecnt,cnt);
        auto wi = new randomgraphsitem(rgs[rgsidx]);
        for (auto p : rgparams) {
            wi->ps.push_back(p);
        }
        _ws->items.push_back(wi);

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

#ifdef CUDAFORCOMPUTENEIGHBORSLISTENMASSE

        std::vector<neighborstype*> nsv;
        nsv.resize(cnt);
        CUDAcomputeneighborslistenmassewrapper(gv,nsv);

        for (int i = 0; i < cnt; ++i)
        {
            auto wi = new graphitem;
            wi->ns = nsv[i];
            wi->g = gv[i];
            wi->name = _ws->getuniquename(wi->classname);
            gv[i]->vertexlabels = vertexlabels;
            _ws->items.push_back( wi );
        }

#else
        for (int i = 0; i < cnt; ++i) {
            auto wi = new graphitem;
            wi->ns = new neighbors(gv[i]);
            wi->g = gv[i];
            wi->name = _ws->getuniquename(wi->classname);
            gv[i]->vertexlabels = vertexlabels;
            _ws->items.push_back( wi );
        }
#endif
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
        *_os << "\t" << "\"o=<filename>\": \t output filename, or \"std::cout\"\n";
        *_os << "\t" << "\"i=<filename>\": \t input filename (use to prepackage verbosity commands)\n";
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
                } else
                {
                    std::time_t result = std::time(nullptr);
                    ofs << "\nAPPEND BEGINS: " << std::asctime(std::localtime(&result)) << "\n";
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
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stoptime - starttime);

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
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTED << "\":" << "\t\t Outputs one graph per fingerprint-equivalence class\n";
        *_os << "\t" << "\"passed\":" << "\t\t Outputs only graphs that pass the last (or parameterized) criterion\n";
    }


    writegraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {}

    void execute(std::vector<std::string> args) override {
        std::ofstream ofs;
        bool ofsrequiresclose = false;
        bool append = true;
        bool overwrite = false;
        bool sortedbool = false;
        bool passed = false;
        std::vector<std::string> passedargs {};
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
            if (cmdlineoptions[n].first == "default") {
                auto parsedargs2 = cmdlineparseiterationthree(cmdlineoptions[n].second);
                bool found = false;
                for (auto p : parsedargs2)
                {
                    if (p.first == "passed")
                    {
                        passed = true;
                        passedargs = p.second;
                        found = true;
                        break;
                    }
                }
                if (found)
                    continue;
            }
            if (cmdlineoptions[n].first == "o") {
                ofname = cmdlineoptions[n].second;
                continue;
            }
            if (cmdlineoptions[n].first == "default" && cmdlineoptions[n].second == CMDLINE_ENUMISOSSORTED) {
                sortedbool = true;
                continue;
            }
            if (cmdlineoptions[n].first == "default" && cmdlineoptions[n].second == CMDLINE_ALL) {
                sortedbool = false;
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
                {
                    ofs.open(ofname,  std::ios::app);
                    std::time_t result = std::time(nullptr);
                    ofs << "\n/* APPEND BEGINS: " << std::asctime(std::localtime(&result)) << " */\n";
                }
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

        std::vector<int> items {};
        for (int i = 0; i < _ws->items.size(); ++i) {
            auto wi = _ws->items[i];
            if (wi->classname == "GRAPH") {
                if (graphitem* gi = dynamic_cast<graphitem*>(wi))
                    items.push_back(i); // default to working with all graphitems
            }
        }


        int sortedcnt = 0;
        std::vector<int> eqclass {};
        if (sortedbool) {
            if (items.size()==0) {
                std::cout << "No graphs to enumerate isomorphisms over\n";
                return;
            }

            eqclass.push_back(0);
            for (int m = 0; m < items.size()-1; ++m) {
                if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[m]])) {
                    bool found = false;
                    for (int r = 0; !found && (r < gi->intitems.size()); ++r) {
                        if (gi->intitems[r]->name() == "FP") {
                            if (fpoutcome* fpo = dynamic_cast<fpoutcome*>(gi->intitems[r])) {
                                if (fpo->value == 1) {
                                    eqclass.push_back(m+1);
                                }
                                found = true;
                                sortedcnt++;
                            } else {
                                std::cout << "Bad cast to fpoutcome (checked by dynamic_cast)\n";
                            }
                        }
                    }
                } else {
                    std::cout << "Bad cast to graphitem (checked by dynamic_cast)\n";
                }
            }
        } else {
            for (int m = 0; m < items.size(); ++m) {
                eqclass.push_back(m);
            }
        }


        for (int i = 0; i < eqclass.size(); ++i) {
            if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[i]]])) {
                if (sortedbool)
                {
                    int eqclasssize;
                    if (i == eqclass.size()-1)
                        eqclasssize = sortedcnt -eqclass[i];
                    else
                        eqclasssize = eqclass[i+1]-eqclass[i];
                    gi->intitems.push_back(new genericgraphoutcome<int>("eqclasssize","Equivalence class size",gi,eqclasssize));
                }
                if (passed)
                {
                    bool accept = true; // default to logical AND
                    if (passedargs.size() == 0)
                        accept = (gi->boolitems.size()>0) && gi->boolitems[gi->boolitems.size()-1]->value;
                    for (auto a : passedargs)
                    {
                        if (is_number(a))
                        {
                            int j = stoi(a);
                            if (j >= 0 && j < gi->boolitems.size())
                                accept = accept && gi->boolitems[j]->value;
                            else
                                if (j < 0 && gi->boolitems.size() + j >= 0)
                                    accept = accept && gi->boolitems[gi->boolitems.size() + j]->value;
                        } else
                        {
                            for (auto bi : gi->boolitems)
                            {
                                if (bi->name() == a)
                                    accept = accept && bi->value;
                            }
                        }

                    }
                    if (accept)
                    {
                        if (!first)
                            *_os << "\n";
                        gi->osmachinereadablegraph(*_os);
                    }
                } else
                {
                    if (!first)
                        *_os << "\n";
                    gi->osmachinereadablegraph(*_os);
                }
                first = false;
            } else {
                std::cout << "Bad cast to graphitem* \n";
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
        *_os << "\t" << "f=\"<graph>\": \t reads the graph from the command line that is enclosed in quotes\n";

    }


    void execute(std::vector<std::string> args) override {

        auto parsedargs = cmdlineparseiterationtwo(args);
        for (int i = 0; i < parsedargs.size(); ++i)
        {
            if (parsedargs[i].first == "f")
            {
                graphitem* gi = new graphitem();
                gi->g = igraphstyle({parsedargs[i].second});
                gi->ns = new neighbors(gi->g);
                gi->name = _ws->getuniquename(gi->classname);
                _ws->items.push_back(gi);

                /*
                int dim = gi->ns->g->dim;
                FP* fp = (FP*)malloc(dim*sizeof(FP));
                for (int k = 0; k < dim; ++k) {
                    fp[k].v=k;
                    fp[k].ns = nullptr;
                    fp[k].nscnt = dim;
                    fp[k].parent = nullptr;
                    fp[k].invert = gi->ns->degrees[j] >= (dim+1)/2;
                }
                takefingerprint(gi->ns,fp,dim);

                fps.push_back(fp);
                nss.push_back(gi->ns);
                dims.push_back(dim);*/
                continue;

            }
        }
        int filenameidx = 0;
        bool oncethrough = true;
        while ((args.size() <= 1 && oncethrough) || filenameidx < args.size()-1)
        {
            if (parsedargs.size() > filenameidx)
                if (parsedargs[filenameidx].first == "f") {
                    filenameidx++;
                    continue;
                }
            filenameidx++;
            oncethrough = false;
            std::ifstream ifs;
            std::istream* is = &std::cin;
            std::ostream* os = _os;
            if ((args.size() > filenameidx) && (args[filenameidx] != "std::cin")) {
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

            // unsigned const thread_count = std::thread::hardware_concurrency();
            std::vector<std::future<bool>> t;
            t.resize(thread_count);

            std::vector<std::vector<std::string>> items {};
            items.resize(thread_count);

            std::string item {};
            int delimetercount = 0;

            std::vector<bool> res {};
            res.resize(thread_count);

            std::vector<graphitem*> giv {};
            giv.resize(thread_count);

            std::vector<bool> foundname {};
            foundname.resize(thread_count);

            std::vector<std::string> name {};
            name.resize(thread_count);

            int threadidx = 0;
            bool done = false;
            while (!done) {
                delimetercount = 0;

                threadidx = 0;
                while ((threadidx < thread_count) && !done) {
                    foundname[threadidx] = false;
                    done = !(*is >> item);

                    while (!done) {
                        if (item == "END" || item == "###") {
                            if( ++delimetercount >= 2 ) {
                                delimetercount = 0;
                                done = done || (is == &std::cin);
                                break;
                            }
                        } else
                            items[threadidx].push_back(item);
                        if (!foundname[threadidx])
                            if (int pos = item.find("#name=") != std::string::npos) {
                                foundname[threadidx] = true;
                                std::string initial = item.substr(pos+5,item.size()-pos-5);
                                if (int pos2 = initial.find(" ") != std::string::npos) {
                                    name[threadidx] =  item.substr(pos+5,pos2);
                                } else
                                    name[threadidx] = item.substr(pos+5,item.size()-pos-5);
                            }

                        done = !(*is >> item);
                    }
                    giv[threadidx] = new graphitem();
                    if (foundname[threadidx])
                        giv[threadidx]->name = name[threadidx];
                    t[threadidx] = std::async(&graphitem::isitemstr,giv[threadidx],items[threadidx]);
                    ++threadidx;
                }
                for (int m = 0; m < threadidx; ++m) {
                    res[m] = t[m].get();
                    items[m].clear();
                    items[m].resize(0);
                }

                for (int m = 0; m < threadidx; ++m) {
                    if (res[m]) {
                        if (giv[m]->name == "") {
                            giv[m]->name = _ws->getuniquename(giv[m]->classname);
                        }
                        _ws->items.push_back(giv[m]);
                    } else
                        delete giv[m];
                }
            }

/*

            graphitem* gi = new graphitem();
            while (gi->isitem(*is)) {
                if (gi->name == "") {
                    gi->name = _ws->getuniquename(gi->classname);
                }
                _ws->items.push_back(gi);
                gi = new graphitem();
            }
            delete gi; */
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


        // unsigned const thread_count = std::thread::hardware_concurrency();
        // unsigned const thread_count = 1;

        std::vector<std::future<void>> t {};
        t.resize(items.size());
#endif

        if (items.empty()) {
            std::cout << "No graphs to fingerprint\n";
            return;
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
            egi->intitems.push_back(new fpoutcome(wi->fpslist[wi->sorted[i]]->ns,egi,wi->res[i]));
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
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f all\"\n";
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTEDVERIFY << "\":  computes automorphisms within each fingerprint-equivalence class\n";
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f all\"\n";
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
            std::vector<const FP*> fpslist {};
            if (sortedbool) {
                if (items.size()==0) {
                    std::cout << "No graphs to enumerate isomorphisms over\n";
                    return;
                }

                eqclass.push_back(0);
                for (int m = 0; m < items.size(); ++m) {
                    auto gi = (graphitem*)_ws->items[items[m]];
                    bool found = false;
                    for (int r = 0; !found && (r < gi->intitems.size()); ++r) {
                        if (gi->intitems[r]->name() == "FP") {
                            auto fpo = (fpoutcome*)gi->intitems[r];
                            if (eqclass[eqclass.size()-1] == m)
                                fpslist.push_back(fpo->fp);
                            if ((fpo->value == 1) && (m < items.size()-1))
                            {
                                eqclass.push_back(m+1);
                            }
                            found = true;
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
                if (fpslist.size() == eqclass.size())
                    t[m] = std::async(&enumisomorphismscore,nslist[eqclass[m]],nslist[eqclass[m]],fpslist[m],fpslist[m]);
                else
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
                std::vector<const FP*> fpslist {};
                fpslist.resize(items.size());
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
                            }
                            fpslist[m] = fpo->fp;
                            found = true;
                        }
                    }
                }
                auto gi = (graphitem*)_ws->items[items[items.size()-1]];
                bool found = false;
                for (int r = 0; !found && (r < gi->intitems.size()); ++r) {
                    if (gi->intitems[r]->name() == "FP") {
                        auto fpo = (fpoutcome*)gi->intitems[r];
                        fpslist[items.size()-1] = fpo->fp;
                        found = true;
                    }
                }

                std::vector<std::future<std::vector<graphmorphism>*>> t {};
                t.resize(items.size());
                int eqclassidx = 1;
                for (int m = 0; m < items.size()-1; ++m) {
                    if ((m+1) != eqclass[eqclassidx]) {
                        t[m] = std::async(&enumisomorphismscore,nslist[m],nslist[m+1],fpslist[m],fpslist[m+1]);
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






class abstractcheckcriterionfeature : public feature {
protected:

    mrecords rec {};
    std::vector<crit*> crs {};
    std::vector<crit*(*)(mrecords*)> crsfactory {};
    std::vector<meas*> mss {};
    std::vector<meas*(*)(mrecords*)> mssfactory {};
    std::vector<tally*> tys {};
    std::vector<tally*(*)(mrecords*)> tysfactory {};
    std::vector<set*> sts {};
    std::vector<set*(*)(mrecords*)> stsfactory {};
    std::vector<set*> oss {};
    std::vector<set*(*)(mrecords*)> ossfactory {};
    std::vector<strmeas*> rms {};
    std::vector<strmeas*(*)(mrecords*)> rmsfactory {};
    std::vector<gmeas*> gms {};
    std::vector<gmeas*(*)(mrecords*)> gmsfactory {};
    std::vector<uncastmeas*> ucs {};
    std::vector<uncastmeas*(*)(mrecords*)> ucsfactory {};

public:
    void setthread_count(const unsigned thread_countin) override {
        feature::setthread_count(thread_countin);
        rec.thread_count = thread_count;
        rec.boolrecs.thread_count = thread_count;
        rec.intrecs.thread_count = thread_count;
        rec.doublerecs.thread_count = thread_count;
        rec.setrecs.thread_count = thread_count;
        rec.tuplerecs.thread_count = thread_count;
        rec.uncastrecs.thread_count = thread_count;
        rec.stringrecs.thread_count = thread_count;
        rec.graphrecs.thread_count = thread_count;
    }

    virtual void listoptions() override {
        feature::listoptions();
    }
    abstractcheckcriterionfeature( std::istream* is, std::ostream* os, workspace* ws ) : feature( is, os, ws) {

        // add any new criterion types to the list here...

        auto (c1) = critfactory<truecrit>;
        auto (cr1) = critfactory<trianglefreecrit>;
        auto (fc) = critfactory<forestcrit>;
        auto (tc) = critfactory<treecrit>;
        auto (kn) = critfactory<knpcrit>;
        auto (connc) = critfactory<connectedcrit>;
        auto (radiusc) = critfactory<radiuscrit>;
        auto (circc) = critfactory<circumferencecrit>;
        auto (diamc) = critfactory<diametercrit>;
        auto (conn1c) = critfactory<connected1crit>;
        auto (kconnc) = critfactory<kconnectedcrit>;
        auto (ledgeconnc) = critfactory<ledgeconnectedcrit>;
        auto (ac) = critfactory<acrit>;
        auto (ec) = critfactory<ecrit>;
        auto (eadjc) = critfactory<eadjcrit>;
        auto (bipc) = critfactory<bipcrit>;
        auto (Nssc) = critfactory<Nsscrit>;
        auto (Separatesc) = critfactory<Separatescrit>;
        auto (connvssc) = critfactory<connvsscrit>;
        auto (connvc) = critfactory<connvcrit>;
        auto (connvsc) = critfactory<connvscrit>;
        auto (indnpc) = critfactory<indnpcrit>;
        auto (nwisec) = critfactory<nwisecrit>;
        auto (toBoolc) = critfactory<toBoolcrit>;
        auto (embedsc) = critfactory<embedscrit>;
        auto (connvusingsetc) = critfactory<connvusingsetcrit>;

        crsfactory.push_back(c1);
        crsfactory.push_back(cr1);
        crsfactory.push_back(fc);
        crsfactory.push_back(tc);
        crsfactory.push_back(kn);
        crsfactory.push_back(connc);
        crsfactory.push_back(radiusc);
        crsfactory.push_back(circc);
        crsfactory.push_back(diamc);
        crsfactory.push_back(conn1c);
        crsfactory.push_back(kconnc);
        crsfactory.push_back(ledgeconnc);
        crsfactory.push_back(ac);
        crsfactory.push_back(ec);
        crsfactory.push_back(eadjc);
        crsfactory.push_back(bipc);
        crsfactory.push_back(Nssc);
        crsfactory.push_back(Separatesc);
        crsfactory.push_back(connvssc);
        crsfactory.push_back(connvc);
        crsfactory.push_back(connvsc);
        crsfactory.push_back(indnpc);
        crsfactory.push_back(nwisec);
        crsfactory.push_back(toBoolc);
        crsfactory.push_back(embedsc);
        crsfactory.push_back(connvusingsetc);

        // ...



        for (int n = 0; n < crsfactory.size(); ++n) {
            crs.push_back((*crsfactory[n])(&rec));
        }


        // add any new measure types to the list here...

        auto (ms4) = measfactory<avgdegreemeas>;
        auto (ms7) = measfactory<legacygirthmeas>;
        auto (mc) = measfactory<maxcliquemeas>;
        auto (cnm) = measfactory<connectedmeas>;
        auto (rm) = measfactory<radiusmeas>;
        auto (circm) = measfactory<circumferencemeas>;
        auto (lcircm) = measfactory<legacycircumferencemeas>;
        auto (diamm) = measfactory<diametermeas>;
        auto (gm) = measfactory<girthmeas>;
        auto (toRealm) = measfactory<toRealmeas>;

        mssfactory.push_back(ms4);
        mssfactory.push_back(ms7);
        mssfactory.push_back(mc);
        mssfactory.push_back(cnm);
        mssfactory.push_back(rm);
        mssfactory.push_back(circm);
        mssfactory.push_back(lcircm);
        mssfactory.push_back(diamm);
        mssfactory.push_back(gm);
        mssfactory.push_back(toRealm);

        // ,,,

        for (int n = 0; n < mssfactory.size(); ++n) {
            mss.push_back((*mssfactory[n])(&rec));
        }


        // add any new tally types to the list here...

        auto (ms2) = tallyfactory<dimtally>;
        auto (ms3) = tallyfactory<edgecnttally>;
        auto (ms5) = tallyfactory<mindegreetally>;
        auto (ms6) = tallyfactory<maxdegreetally>;
        auto (Knt) = tallyfactory<Kntally>;
        auto (indnt) = tallyfactory<indntally>;
        auto (cyclet) = tallyfactory<cycletally>;
        auto (kappat) = tallyfactory<kappatally>;
        auto (lambdat) = tallyfactory<lambdatally>;
        auto (vdt) = tallyfactory<vdtally>;
        auto (st) = tallyfactory<sizetally>;
        auto (lt) = tallyfactory<lengthtally>;
        auto (pct) = tallyfactory<pctally>;
        auto (idxt) = tallyfactory<idxtally>;
        auto (Nt) = tallyfactory<Ntally>;
        auto (cyclesvt) = tallyfactory<cyclesvtally>;
        auto (Chit) = tallyfactory<Chitally>;
        auto (Chigreedyt) = tallyfactory<Chigreedytally>;
        auto (Chiprimet) = tallyfactory<Chiprimetally>;
        auto (Chiprimegreedyt) = tallyfactory<Chiprimegreedytally>;
        auto (Nsst) = tallyfactory<Nsstally>;
        auto (cyclest) = tallyfactory<cyclestally>;
        auto (toIntt) = tallyfactory<toInttally>;
        auto (connt) = tallyfactory<conntally>;

        tysfactory.push_back(ms2);
        tysfactory.push_back(ms3);
        tysfactory.push_back(ms5);
        tysfactory.push_back(ms6);
        tysfactory.push_back(Knt);
        tysfactory.push_back(indnt);
        tysfactory.push_back(cyclet);
        tysfactory.push_back(kappat);
        tysfactory.push_back(lambdat);
        tysfactory.push_back(vdt);
        tysfactory.push_back(st);
        tysfactory.push_back(lt);
        tysfactory.push_back(pct);
        tysfactory.push_back(idxt);
        tysfactory.push_back(Nt);
        tysfactory.push_back(cyclesvt);
        tysfactory.push_back(Chit);
        tysfactory.push_back(Chigreedyt);
        tysfactory.push_back(Chiprimet);
        tysfactory.push_back(Chiprimegreedyt);
        tysfactory.push_back(Nsst);
        tysfactory.push_back(cyclest);
        tysfactory.push_back(toIntt);
        tysfactory.push_back(connt);

        // ...

        for (int n = 0; n < tysfactory.size(); ++n) {
            tys.push_back((*tysfactory[n])(&rec));
        }

        auto (Vs) = setfactory<Vset>;
        auto (Ps) = setfactory<Pset>;
        auto (Sizedsubs) = setfactory<Sizedsubset>;
        auto (NNs) = setfactory<NNset>;
        auto (Nulls) = setfactory<Nullset>;
        auto (Es) = setfactory<Eset>;
        auto (idxs) = setfactory<idxset>;
        auto (TtoS) = setfactory<TupletoSet>;
        auto (Paths) = setfactory<Pathsset>;
        auto (Pathsusingvsets) = setfactory<Pathsusingvsetset>;
        auto (Cyclesvs) = setfactory<Cyclesvset>;
        auto (Setpartitions) = setfactory<Setpartition>;
        auto (nEs) = setfactory<nEset>;
        auto (Cycless) = setfactory<Cyclesset>;
        auto (Perms) = setfactory<Permset>;
        auto (Subgraphss) = setfactory<Subgraphsset>;
        auto (InducedSubgraphss) = setfactory<InducedSubgraphsset>;
        auto (Componentss) = setfactory<Componentsset>;
        auto (Edgess) = setfactory<Edgesset>;
        auto (Automs) = setfactory<Automset>;
        auto (Conncs) = setfactory<Connc>;
        auto (Ns) = setfactory<Nset>;
        auto (Choices) = setfactory<Choiceset>;
        auto (Choice2s) = setfactory<Choice2set>;
        auto (as) = setfactory<aset>;
        auto (eadjs) = setfactory<eadjset>;
        auto (e2eadjs) = setfactory<e2eadjset>;
        auto (Epathss) = setfactory<Epathsset>;
        auto (Mapss) = setfactory<Mapsset>;
        auto (Gpathss) = setfactory<Gpathsset>;

        stsfactory.push_back(Vs);
        stsfactory.push_back(Ps);
        stsfactory.push_back(Sizedsubs);
        stsfactory.push_back(NNs);
        stsfactory.push_back(Nulls);
        stsfactory.push_back(Es);
        stsfactory.push_back(idxs);
        stsfactory.push_back(TtoS);
        stsfactory.push_back(Paths);
        stsfactory.push_back(Cyclesvs);
        stsfactory.push_back(Setpartitions);
        stsfactory.push_back(nEs);
        stsfactory.push_back(Cycless);
        stsfactory.push_back(Perms);
        // stsfactory.push_back(Subgraphss);
        // stsfactory.push_back(InducedSubgraphss);
        stsfactory.push_back(Componentss);
        stsfactory.push_back(Edgess);
        stsfactory.push_back(Automs);
        stsfactory.push_back(Conncs);
        stsfactory.push_back(Ns);
        stsfactory.push_back(Choices);
        stsfactory.push_back(Choice2s);
        stsfactory.push_back(as);
        stsfactory.push_back(eadjs);
        stsfactory.push_back(e2eadjs);
        stsfactory.push_back(Epathss);
        stsfactory.push_back(Pathsusingvsets);
        stsfactory.push_back(Mapss);
        stsfactory.push_back(Gpathss);

        for (int n = 0; n < stsfactory.size(); ++n) {
            sts.push_back((*stsfactory[n])(&rec));
        }

        // ...

        auto (Chip) = tuplefactory<Chituple>;
        auto (Chigreedyp) = tuplefactory<Chigreedytuple>;
        auto (Sp) = tuplefactory<Stuple>;
        auto (nwalksbetweenp) = tuplefactory<nwalksbetweentuple>;
        auto (Connvp) = tuplefactory<Connvtuple>;
        auto (Connmatrixp) = tuplefactory<Connmatrix>;
        auto (Subp) = tuplefactory<Subtuple>;
#ifdef FLAGCALC_CUDA
        auto (CUDAnwalksbetweenp) = tuplefactory<CUDAnwalksbetweentuple>;
        auto (CUDAConnvp) = tuplefactory<CUDAConnvtuple>;
#endif

        ossfactory.push_back(Chip);
        ossfactory.push_back(Chigreedyp);
        ossfactory.push_back(Sp);
        ossfactory.push_back(nwalksbetweenp);
        ossfactory.push_back(Connvp);
        ossfactory.push_back(Connmatrixp);
        ossfactory.push_back(Subp);
#ifdef FLAGCALC_CUDA
        ossfactory.push_back(CUDAnwalksbetweenp);
        ossfactory.push_back(CUDAConnvp);
#endif

        for (int n = 0; n < ossfactory.size(); ++n) {
            oss.push_back((*ossfactory[n])(&rec));
        }

        // auto (Gg) = stringfactory<Ggmeas>;

        // rmsfactory.push_back(Chip);

        // for (int n = 0; n < rmsfactory.size(); ++n) {
        //    oss.push_back((*rmsfactory[n])(&rec));
        // }

        auto (GraphonVEg) = graphfactory<GraphonVEgmeas>;
        auto (SubgraphonUg) = graphfactory<SubgraphonUgmeas>;
        auto (Gg) = graphfactory<Ggmeas>;

        gmsfactory.push_back(GraphonVEg);
        gmsfactory.push_back(SubgraphonUg);
        gmsfactory.push_back(Gg);

        for (int n = 0; n < gmsfactory.size(); ++n) {
            gms.push_back((*gmsfactory[n])(&rec));
        }


        // ...



    }

    ~abstractcheckcriterionfeature() {

        for (int i = 0; i < crs.size();++i) {
            delete crs[i];
        }
        for (int i = 0; i < mss.size(); ++i) {
            delete mss[i];
        }
        for (int i = 0; i < tys.size(); ++i) {
            delete tys[i];
        }
        for (int i = 0; i < sts.size(); ++i) {
            delete sts[i];
        }
        for (int i = 0; i < oss.size(); ++i) {
            delete oss[i];
        }
        for (int i = 0; i < rms.size(); ++i) {
            delete rms[i];
        }
        for (int i = 0; i < gms.size(); ++i) {
            delete gms[i];
        }
        for (int i = 0; i < ucs.size(); ++i) {
            delete ucs[i];
        }
    }


};

inline void readfromfile( std::string ifname, std::vector<std::string>& out )
{
    out.clear();
    std::cout << "Opening file " << ifname << "\n";
    std::ifstream infile(ifname);
    if (infile.good()) {
        std::ifstream ifs;
        ifs.open(ifname, std::fstream::in );
        std::string dat = "";
        std::string tmp {};
        while (!ifs.eof()) {
            ifs >> tmp;
            bool changed = false;
            if (tmp == "#include") {
                ifs >> tmp;
                if (tmp.size() >= 2 && ((tmp[0] == '\"' && tmp[tmp.size()-1] == '\"') || (tmp[0] == '<' && tmp[tmp.size()-1] == '>'))) {
                    tmp = tmp.substr(1, tmp.size()-2);
                }
                std::vector<std::string> includedout {};
                readfromfile(tmp, includedout);
                for (auto s : includedout) {
                    out.push_back(s);
                }
                continue;
            }
            while (!ifs.eof() && tmp != "END" && tmp != "###")
            {
                dat += " " + tmp + " ";
                ifs >> tmp;
                changed = true;
            }
            if (changed)
                out.push_back(dat);
            dat = "";
        }
        ifs.close();
    } else {
        std::cout << "Couldn't open file for reading " << ifname << "\n";
    }
}

inline void removecomments( std::vector<std::string> streamstr, std::vector<std::string>& streamout )
{
    bool incomment = false;
    bool entirelinecomment = false;
    int i = 0;
    streamout.clear();
    std::string workingstr;
    if (streamstr.size() > 0)
        workingstr = streamstr[0];
    std::string outstr {};
    while (i < streamstr.size())
    {
        if (!incomment)
        {
            auto commentstartpos = workingstr.find(COMMENTSTARTDELIMITER);
            if (commentstartpos != std::string::npos)
            {
                incomment = true;
                entirelinecomment = false;
                outstr = workingstr.substr(0, commentstartpos);
                workingstr = workingstr.substr(commentstartpos+strlen(COMMENTSTARTDELIMITER),workingstr.size()-commentstartpos-strlen(COMMENTSTARTDELIMITER));
            }
            else
            {
                commentstartpos = workingstr.find(COMMENTENTIRELINEDELIMITER);
                if (commentstartpos != std::string::npos)
                {
                    incomment = true;
                    entirelinecomment = true;
                    outstr = workingstr.substr(0, commentstartpos);
                    workingstr = workingstr.substr(commentstartpos+strlen(COMMENTENTIRELINEDELIMITER),workingstr.size()-commentstartpos-strlen(COMMENTENTIRELINEDELIMITER));
                } else
                {
                    outstr = workingstr;
                    streamout.push_back(outstr);
                    outstr.clear();
                    ++i;
                    if (i < streamstr.size())
                        workingstr = streamstr[i];
                    continue;
                }
            }
        }
        if (incomment)
        {
            int commentendpos;
            int delimsize;
            if (!entirelinecomment) {
                commentendpos = workingstr.find(COMMENTENDDELIMITER);
                delimsize = strlen(COMMENTENDDELIMITER);
            } else {
                commentendpos = workingstr.find('\n');
                if (commentendpos == std::string::npos) {
                    commentendpos = workingstr.size()-1;
                }
                delimsize = 1;
            }
            if (commentendpos != std::string::npos)
            {
                incomment = false;
                workingstr = workingstr.substr(commentendpos + delimsize, workingstr.size()-commentendpos - delimsize);
                continue;
            }
            else
            {
                streamout.push_back(outstr);
                outstr.clear();
                ++i;
                if (i < streamstr.size())
                    workingstr = streamstr[i];
                continue;
            }
        }

    }

}



inline void readfromfileandremovecomments(std::string ifname, std::vector<std::string>& out )
{
    std::vector<std::string> streamstr;
    readfromfile(ifname, streamstr);
    removecomments(streamstr, out);
}



struct compactcmdline
{
    std::string t;
    int i;
    bool n;
};

inline compactcmdline parsecompactcmdline( std::string& s )
{
    compactcmdline res;
    res.t = "";
    int j = 0;
    if (!s.empty() && s[0] == 'n')
    {
        res.n = true;
        j = 1;
    } else
    {
        res.n = false;
        j = 0;
    }
    while (s.size() > j && isalpha(s[j]))
        res.t.push_back(s[j++]);
    if (j >= s.size())
    {
        res.i = 0;
        return res;
    }
    if (is_number(s.substr(j,s.size()-j)))
        res.i = std::stoi( s.substr(j,s.size()-j));
    else
        res.i = 0;
    return res;

}




// unsigned const thread_count = 1;



template<typename T>
void runthreads(const int iidx, const params& ps, thrrecords<T>& r )
{
    const double section = double(r.sz) / double(r.thread_count);
    std::vector<std::future<void>> t;
    t.resize(r.thread_count);
    for (int m = 0; m < r.thread_count; ++m) {
        const int startidx = int(m*section);
        const int stopidx = int((m+1.0)*section);
        t[m] = std::async(&thrrecords<T>::threadfetch,r,startidx,stopidx,iidx,ps);
    }
    for (int m = 0; m < r.thread_count; ++m)
    {
        t[m].get();
    }
}

template<typename T>
void runthreadspartial(const int iidx, const params& ps, thrrecords<T>& r, std::vector<bool>* todo )
{
    const double section = double(r.sz) / double(r.thread_count);
    std::vector<std::future<void>> t;
    t.resize(r.thread_count);
    for (int m = 0; m < r.thread_count; ++m) {
        const int startidx = int(m*section);
        const int stopidx = int((m+1.0)*section);
        t[m] = std::async(&thrrecords<T>::threadfetchpartial,r,startidx,stopidx,iidx,ps,todo);
    }
    for (int m = 0; m < r.thread_count; ++m)
    {
        t[m].get();
    }
}



template<typename T>
class threaddata
{
public:
  std::vector<T> d {};
};

/* THIS SHOWS NO SPEED IMPROVEMENT
template<typename T>
void populatewithreaded(chkmeasaitem<T>* wi, std::vector<T>& threaddata,
    std::vector<graphtype*>& glist, std::vector<neighborstype*>& nslist,
    std::vector<bool>& todo,
    const int startidx,
    const int stopidx)
{
    for (int m=startidx; m < stopidx; ++m)
    {
        wi->parentbool[m] = todo[m];
        wi->parentboolcnt += (todo[m] ? 1 : 0);
        if (todo[m])
        {
            wi->res[m] = threaddata[m];
            wi->glist[m] = glist[m];
            wi->sorted[m] = m;
            wi->nslist[m] = nslist[m];
            wi->meas[m] = threaddata[m];
            // if (k < res.size()) {
            // recall all the sentence-level criteria were added after malloc
            // if (l == 0)
            // (res[k])[m] = wi->res[m];
            // }
            // if (l < resm.size())
            // {
            // if (threadbool[m])
            // resm[l][m] = threaddouble[m];
            // }
            // wi->meas.resize(eqclass.size());
            // if (wi->res[m]) {
            // wi->meas[m] = threaddouble[m]; //ms[crmspairs[k][l]]->takemeasureidxed(eqclass[m]);
            // if (!(done[m][l])) {
            // gi->doubleitems.push_back( new abstractmeasureoutcome<double>(ms[l],gi,wi->meas[m]));
            // done[m][l] = true;
            // } // default for wi->meas[m] ?
            // }
        }
    }
}
*/

template<typename T>
void populatewi(workspace* _ws, chkmeasaitem<T>* wi, std::vector<T>& threaddata, std::vector<int> items, std::vector<int>& eqclass,
    std::vector<graphtype*>& glist, std::vector<neighborstype*>& nslist,
    std::vector<bool>& todo ) {

    wi->res.resize(eqclass.size());
    wi->meas.resize(eqclass.size());
    wi->parentbool.resize(eqclass.size());
    wi->parentboolcnt = 0;
    wi->fpslist = {};
    wi->glist.resize(eqclass.size());
    wi->sorted.resize(eqclass.size());
    wi->gnames.resize(eqclass.size());
    wi->nslist.resize(eqclass.size());

    for (int m=0; m < eqclass.size(); ++m)
    {
        wi->parentbool[m] = todo[m];
        wi->parentboolcnt += (todo[m] ? 1 : 0);
        if (todo[m])
        {
            wi->res[m] = threaddata[m];
            wi->glist[m] = glist[m];
            wi->sorted[m] = m;
            wi->nslist[m] = nslist[m];
            wi->meas[m] = threaddata[m];
            // if (k < res.size()) {
            // recall all the sentence-level criteria were added after malloc
            // if (l == 0)
            // (res[k])[m] = wi->res[m];
            // }
            // if (l < resm.size())
            // {
            // if (threadbool[m])
            // resm[l][m] = threaddouble[m];
            // }
            // wi->meas.resize(eqclass.size());
            // if (wi->res[m]) {
            // wi->meas[m] = threaddouble[m]; //ms[crmspairs[k][l]]->takemeasureidxed(eqclass[m]);
            // if (!(done[m][l])) {
            // gi->doubleitems.push_back( new abstractmeasureoutcome<double>(ms[l],gi,wi->meas[m]));
            // done[m][l] = true;
            // } // default for wi->meas[m] ?
            // }
        }
    }


   /* The following shows no speed improvement
    const double section = double(eqclass.size()) / double(thread_count);
    std::vector<std::future<void>> t;
    t.resize(thread_count);
    for (int m = 0; m < thread_count; ++m) {
        const int startidx = int(m*section);
        const int stopidx = int((m+1.0)*section);
        t[m] = std::async(std::bind(&populatewithreaded<T>,wi,threaddata,glist,nslist,todo,startidx,stopidx));
    }
    for (int m = 0; m < thread_count; ++m)
    {
        t[m].get();
    }*/

    wi->name = _ws->getuniquename(wi->classname);
    _ws->items.push_back(wi);
}


struct storedprocstruct
{
    std::string name;
    ams a;
    namedparams nps;
    std::string body;
    int iidx;
};

#ifdef FLAGCALCWITHPYTHON
struct pythonmethodstruct {
    std::string name;
    py::object m;
    namedparams nps;
    namedparams ips; // e.g. dim, adjmatrix, etc.
    ams a;
    int iidx;
    std::string filename;
};

#endif


class checkcriterionfeature : public abstractcheckcriterionfeature {
protected:

    std::vector<itn*> iter {};
    std::vector<int> litnumps {};
    std::vector<measuretype> littypes {};
    std::vector<std::string> litnames {};
    std::vector<storedprocstruct> storedprocedures {};

#ifdef FLAGCALCWITHPYTHON
    std::vector<pythonmethodstruct> pythonmethods {};
    bool is_pyinterpreterstarted = false;
#endif

    itn* newiteration( measuretype mtin, int roundin, const ams ain, const bool hiddenin = false )
    {
        int j;
        auto resi = new itn;
        switch (mtin)
        {
        case measuretype::mtbool:
            j = rec.boolrecs.pmsv->size();
            rec.boolrecs.pmsv->push_back(ain.a.cs);
            resi->nps = ain.a.cs->nps;
            break;
        case measuretype::mtdiscrete:
            j = rec.intrecs.pmsv->size();
            rec.intrecs.pmsv->push_back(ain.a.ts);
            resi->nps = ain.a.ts->nps;
            break;
        case measuretype::mtcontinuous:
            j = rec.doublerecs.pmsv->size();
            rec.doublerecs.pmsv->push_back(ain.a.ms);
            resi->nps = ain.a.ms->nps;
            break;
        case measuretype::mtset:
            j = rec.setrecs.pmsv->size();
            rec.setrecs.pmsv->push_back(ain.a.ss);
            resi->nps = ain.a.ss->nps;
            break;
        case measuretype::mttuple:
            j = rec.tuplerecs.pmsv->size();
            rec.tuplerecs.pmsv->push_back(ain.a.os);
            resi->nps = ain.a.os->nps;
            break;
        case measuretype::mtstring:
            j = rec.stringrecs.pmsv->size();
            rec.stringrecs.pmsv->push_back(ain.a.rs);
            resi->nps = ain.a.rs->nps;
            break;
        case measuretype::mtgraph:
            j = rec.graphrecs.pmsv->size();
            rec.graphrecs.pmsv->push_back(ain.a.gs);
            resi->nps = ain.a.gs->nps;
            break;
        case measuretype::mtuncast:
            j = rec.uncastrecs.pmsv->size();
            rec.uncastrecs.pmsv->push_back(ain.a.uc);
            resi->nps = ain.a.uc->nps;
            break;

        }

        resi->t = mtin;
        resi->round = roundin;
        resi->iidx = rec.maxm()+1;
        rec.addm(resi->iidx,resi->t,j);
        rec.thread_count = thread_count;
        resi->hidden = hiddenin;
        //resi->ps.clear();
        return resi;

    }


    int lookupstoredprocedure( const std::string sin, const int roundin )
    {
        for (auto i = 0; i < storedprocedures.size(); ++i)
        {
            auto sps = storedprocedures[i];
            if (sin == sps.name) {
                if (sps.iidx < 0)
                {
                    ams a = sps.a;
                    namedparams nps = sps.nps;
                    std::string s = bindformula(sps.body, sps.a.t, roundin );

//                    params ps2 {};
//                    for (auto p : ps)
//                        ps2.push_back(p.second);

                    switch (a.t)
                    {
                    case mtbool:
                        {
                            sentofcrit* cs = new sentofcrit(&rec,litnumps,littypes,litnames, nps , s, sps.name);
                            a.a.cs = cs;
                            a.a.cs->negated = false;
                            crs.push_back(a.a.cs);
                            break;
                        }
                    case mtdiscrete:
                        {
                            formtally* ts = new formtally(&rec,litnumps,littypes,litnames, nps, s, sps.name);
                            a.a.ts = ts;
                            tys.push_back(a.a.ts);
                            break;
                        }
                    case mtcontinuous:
                        {
                            formmeas* ms = new formmeas(&rec,litnumps,littypes,litnames, nps,s, sps.name);
                            a.a.ms = ms;
                            mss.push_back(a.a.ms);
                            break;
                        }
                    case mtset:
                        {
                            formset* ss = new formset(&rec,litnumps,littypes,litnames, nps, s, sps.name);
                            a.a.ss = ss;
                            sts.push_back(a.a.ss);
                            break;
                        }
                    case mttuple:
                        {
                            formtuple* os = new formtuple(&rec,litnumps,littypes,litnames,nps, s, sps.name);
                            a.a.os = os;
                            oss.push_back(a.a.os);
                            break;
                        }
                    case mtstring:
                        {
                            formstring* rs = new formstring(&rec,litnumps,littypes,litnames,nps, s, sps.name);
                            a.a.rs = rs;
                            rms.push_back(a.a.rs);
                            break;
                        }
                    case mtgraph:
                        {
                            formgraph* gs = new formgraph(&rec,litnumps,littypes,litnames,nps, s, sps.name);
                            a.a.gs = gs;
                            gms.push_back(a.a.gs);
                            break;
                        }
                    }

                    auto it = newiteration(a.t,roundin,a,true);

                    // int j = addmeas( sps.name, sps.a.t, roundin);
                    it->nps = nps;
                    iter.push_back(it);

                    int j = iter.size()-1;
                    litnumps.push_back(nps.size());
                    littypes.push_back(a.t);
                    litnames.push_back(sps.name);
//                    litnumps.resize(j+1);
//                    litnumps[j] = ps.size();
//                    littypes.resize(j+1);
//                    littypes[j] = a.t;
//                    litnames.resize(j+1);
//                    litnames[j] = sps.name;
                    sps.iidx = j;
                    storedprocedures[i] = sps;
                    return sps.iidx;
                } else
                    return sps.iidx;
            }
        }
        return -1;
    }

#ifdef FLAGCALCWITHPYTHON
    int lookuppythonmethod( const std::string sin, const int roundin )
    {
        for (auto i = 0; i < pythonmethods.size(); ++i)
        {
            auto pym = pythonmethods[i];
            if (sin == pym.name) {
                if (pym.iidx < 0)
                {
                    ams a = pym.a;
                    namedparams nps = pym.nps;

                    std::string s {};

                    pythonuncastmeas* pu = new pythonuncastmeas(&rec,pym.m,pym.nps,pym.ips,pym.name);
                    a.a.uc = pu;


                    auto it = newiteration(a.t,roundin,a,true);

                    it->nps = nps;
                    iter.push_back(it);

                    int j = iter.size()-1;
                    litnumps.push_back(nps.size());
                    littypes.push_back(a.t);
                    litnames.push_back(pym.name);
                    pym.iidx = j;
                    pythonmethods[i] = pym;
                    return pym.iidx;
                } else
                    return pym.iidx;
            }
        }
        return -1;
    }

#endif

    void addstoredproc( const std::string sin )
    {

        std::vector<std::string> parsedstring = parsecomponents( sin );
        if (parsedstring.size() > 1)
        {
            ams a {};
            a.t = mtbool;
            bool found = false;
            for (auto tname : measuretypenames)
                if (parsedstring[0] == tname.second)
                {
                    a.t = tname.first;
                    found = true;
                }
            if (!found)
            {
                std::cout << "Unknown stored procedure type " << parsedstring[0]
                    << "; using " << measuretypenames[a.t] << std::endl;
            }

            std::string name = parsedstring[1];
            if (lookupiter(name) >= 0)
            {
                std::cout << "Stored procedure name already exists (" << name << ")" << std::endl;
                return;
            }


            params ps {};
            // std::vector<qclass*> variables;
            namedparams variablenames {};

            int i = 2;
            if (parsedstring.size() >= 4 && parsedstring[i] == "(")
            {
                ++i;
                std::vector<std::string> psstring {};
                while (i < parsedstring.size() && parsedstring[i] != ")")
                    psstring.push_back(parsedstring[i++]);
                ++i;

                for (int j = 0; j < psstring.size(); j += 3)
                {
                    std::string ptype = psstring[j];
                    std::string pname = psstring[j+1];
                    valms v;
                    v.t = mtdiscrete;
                    bool found = false;
                    for (auto tname : measuretypenames)
                        if (ptype == tname.second)
                        {
                            v.t = tname.first;
                            found = true;
                        }
                    if (!found)
                    {
                        std::cout << "Unknown stored procedure variable type " << ptype
                            << "; using " << measuretypenames[v.t] << std::endl;
                    }
                    ps.push_back(v);
                    // auto qc = new qclass();
                    // qc->name = pname;
                    // variables.push_back(qc);
                    variablenames.push_back({pname,v});
                }
            }

            std::string form {};
            for ( ; i < parsedstring.size(); ++i)
                form += parsedstring[i] + " ";

            // std::cout << name << ": " << form << std::endl;

            storedprocstruct sps;

            sps.name = name;
            sps.body = form;
            sps.nps = variablenames;
            sps.a = a;
            sps.iidx = -1;
            storedprocedures.push_back(sps);

        } else
        {
            std::cout << "Empty stored procedure" << std::endl;
        }
    }


    int lookupiter( const std::string sin )
    {
        std::string sn;
        valms res;
        for (auto i = 0; i < iter.size(); ++i)
        {
            auto a = rec.lookup(iter[i]->iidx);
            switch (a.t)
            {
            case mtbool: sn = a.a.cs->shortname;
                break;
            case mtdiscrete: sn = a.a.ts->shortname;
                break;
            case mtcontinuous: sn = a.a.ms->shortname;
                break;
            case mtset: sn = a.a.ss->shortname;
                break;
            case mttuple: sn = a.a.os->shortname;
                break;
            case mtstring: sn = a.a.rs->shortname;
                break;
            case mtgraph: sn = a.a.gs->shortname;
                break;
            }
            if (sn == sin)
                return iter[i]->iidx;

        }
        return -1;

    }


    int addmeas(const std::string sin, const measuretype mtin, const int roundin )
    {
        int li = lookupiter(sin);
        if (li >= 0)
            return li;
        for (int i = 0; i < stsfactory.size(); ++i)
        {
            if (sin == sts[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtset;
                a.a.ss = (*stsfactory[i])(&rec);
                iter.push_back(newiteration(mtset,roundin,a,true));  // hide also those of pssz == 0
                int j = iter.size()-1;

                litnumps.push_back(sts[i]->npssz);
                littypes.push_back(a.t);
                litnames.push_back(sin);
                return j;
            }
        }
        for (int i = 0; i < crsfactory.size(); ++i)
        {
            if (sin == crs[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtbool;
                a.a.cs = (*crsfactory[i])(&rec);
                iter.push_back(newiteration(mtbool,roundin,a,true));
                int j = iter.size()-1;

                litnumps.push_back(crs[i]->npssz);
                littypes.push_back(a.t);
                litnames.push_back(sin);
                return j;
            }
        }
        for (int i = 0; i < mssfactory.size(); ++i)
        {
            if (sin == mss[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtcontinuous;
                a.a.ms = (*mssfactory[i])(&rec);
                iter.push_back(newiteration(mtcontinuous,roundin,a, true));
                int j = iter.size()-1;

                litnumps.push_back(mss[i]->npssz);
                littypes.push_back(a.t);
                litnames.push_back(sin);
                return j;
            }
        }
        for (int i = 0; i < tysfactory.size(); ++i)
        {
            if (sin == tys[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtdiscrete;
                a.a.ts = (*tysfactory[i])(&rec);
                iter.push_back(newiteration(mtdiscrete,roundin,a,true));
                int j = iter.size()-1;

                litnumps.push_back(tys[i]->npssz);
                littypes.push_back(a.t);
                litnames.push_back(sin);
                return j;
            }
        }
        for (int i = 0; i < ossfactory.size(); ++i)
        {
            if (sin == oss[i]->shortname)
            {
                ams a;
                a.t = measuretype::mttuple;
                a.a.os = (*ossfactory[i])(&rec);
                iter.push_back(newiteration(mttuple,roundin,a,true));
                int j = iter.size()-1;

                litnumps.push_back(oss[i]->npssz);
                littypes.push_back(a.t);
                litnames.push_back(sin);
                return j;
            }
        }

        for (int i = 0; i < rmsfactory.size(); ++i)
        {
            if (sin == rms[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtstring;
                a.a.rs = (*rmsfactory[i])(&rec);
                iter.push_back(newiteration(mtstring,roundin,a,true));  // hide also those of pssz == 0
                int j = iter.size()-1;

                litnumps.push_back(rms[i]->npssz);
                littypes.push_back(a.t);
                litnames.push_back(sin);
                return j;
            }
        }

        for (int i = 0; i < gmsfactory.size(); ++i)
        {
            if (sin == gms[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtgraph;
                a.a.gs = (*gmsfactory[i])(&rec);
                iter.push_back(newiteration(mtgraph,roundin,a,true));  // hide also those of pssz == 0
                int j = iter.size()-1;

                litnumps.push_back(gms[i]->npssz);
                littypes.push_back(a.t);
                litnames.push_back(sin);
                return j;
            }
        }

        return -1;
    }


    std::string bindformula( std::string sin, const measuretype mtin, const int roundin )
    {

        /* const std::regex r {"\\[([[:alpha:]]\\w*)\\]"};
        std::vector<std::string> out {}; */
        std::vector<std::string> out2 {};

        /* for (std::sregex_iterator p(sin.begin(),sin.end(),r); p!=std::sregex_iterator{}; ++p)
        {
            out.push_back((*p)[1]);
        } */
        const std::regex r2 {"([[:alpha:]]\\w*)"};
        for (std::sregex_iterator p( sin.begin(), sin.end(),r2); p != std::sregex_iterator{}; ++p)
        {
            out2.push_back((*p)[1]);
        }
        /*
        for (int i = 0; i < out.size(); ++i)
        {
            if (is_number(out[i]))
                continue;
            int idx = lookupiter(out[i]);
            if (idx < 0)
            {
                idx = lookupstoredprocedure(out[i], roundin);
                if (idx < 0)
                    idx = addmeas( out[i],mtin, roundin );
            }

            if (idx < 0)
                std::cout << "Unknown bracketed literal " << out[i] << std::endl;

            // std::string replacement = "[" + std::to_string(idx) + "]";
            std::string replacement = "[" + out[i] + "]";

            std::string pattern = "\\[" + out[i] + "\\]";
            std::regex reg(pattern);
            sin = std::regex_replace(sin,reg,replacement);
        }
        // std::cout << sin << "\n";
*/
        for (int i = 0; i < out2.size(); ++i)
        {
            if (is_operator(out2[i]))
                continue;

            if (is_number(out2[i]))
                continue;
            int idx = lookupiter(out2[i]);
            if (idx < 0)
            {
                idx = lookupstoredprocedure(out2[i], roundin);
                if (idx < 0)
                {
#ifdef FLAGCALCWITHPYTHON
                    idx = lookuppythonmethod(out2[i], roundin);
                    if (idx < 0) {
                        idx = addmeas( out2[i],mtin, roundin );
                    }
#else
                    idx = addmeas( out2[i],mtin, roundin );
#endif
                    // if (idx < 0)
                    // std::cout << " presumed variable " << out2[i] << std::endl;
                }
                // if (idx < 0)
                    // std::cout << " presumed variable " << out2[i] << std::endl;
            }
        }


        return sin;
    }



public:

    // std::vector<criterion*> cs {};
    // std::vector<measure*> ms {};
    // std::vector<std::pair<int,std::string>> mscombinations {};
    std::vector<std::string> sentences {};
    std::vector<std::string> formulae {};

    void clear()
    {
        iter.clear();
        litnumps.clear();
        littypes.clear();
        litnames.clear();
        storedprocedures.clear();
#ifdef FLAGCALCWITHPYTHON
        pythonmethods.clear();
#endif
        sentences.clear();
        formulae.clear();
    }


    std::string cmdlineoption() override { return "a"; }
    std::string cmdlineoptionlong() { return "checkcriteria"; }
    checkcriterionfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractcheckcriterionfeature( is, os, ws) {
    }

    ~checkcriterionfeature() {
        // if (is_pyinterpreterstarted)
            // py::finalize_interpreter();
        // is_pyinterpreterstarted = false; // crashes when using multiple s's
    }
    void listoptions() override {
        abstractcheckcriterionfeature::listoptions();
        *_os << "\t" << "\"" << CMDLINE_ALL << "\": \t\t\t checks criteria for ALL graphs found on the workspace\n";
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTED << "\": \t\t checks criteria for each fingerprint-equivalent class\n";
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f\"\n";
        *_os << "\t" << "\"j=n\": uses n threads (set to 1 for any Python integration)\n";
        *_os << "\t" << "\"ipy=<filename>\": uses Python script ../python/<filename>.py\n";
        *_os << "\t" << "\"not=<n>\": \t\t applies the logical NOT to the criteria numbered n, prior to AND or OR\n";
        *_os << "\t" << "\"l=AND\": \t\t applies the logical AND to the criteria (\"m=\" is optional)\n";
        *_os << "\t" << "\"l=OR\": \t\t applies the logical OR to the criteria (\"m=\" is optional)\n";
        *_os << "\t" << "\"s=<sentence>\": applies the logical sentence inside the quotes to the criteria\n";
        *_os << "\t" << "\"is=<filename>\": applies the logical sentence in <filename> to the criteria\n";
        *_os << "\t" << "\"f=<graph>\": \t checks the criterion of <graph> embedding\n";
        *_os << "\t" << "\"nf=<graph>\": \t checks the criterion of NOT <graph> embedding\n";
        *_os << "\t" << "\"ft=<graph>\": \t counts the tally of <graph> embedding\n";
        *_os << "\t" << "\"if=<filename>\": applies the criteria of flag(s) in <filename> embedding\n";
        *_os << "\t" << "\"a=<expression>\": uses mathematical expression to serve as a measure\n";
        *_os << "\t" << "\"z=<expression>\": uses mathematical expression to serve as an integer\n";
        *_os << "\t" << "\"e=<expression>\": uses mathematical expression to serve as a set\n";
        *_os << "\t" << "\"p=<expression>\": uses mathematical expression to serve as a tuple\n";
        *_os << "\t" << "\"ia=<filename>\": uses the mathematical expression(s) in <filename> embedding\n";
        *_os << "\t" << "<expression>: using these built-in mtcontinuous-type mathematical functions:\n";
        for (const auto& pair : global_fnptrs) {
            *_os << "\t\t\"" << pair.first << "\": takes " << pair.second.second << " input(s)\n";
        }
        *_os << "\t" << "<criterion>:\t which criterion to use, standard options are:\n";
        for (int n = 0; n < crs.size(); ++n) {
            *_os << "\t\t\"" << crs[n]->shortname << "\": " << crs[n]->name << "\n";
        }
        *_os << "\t" << "m=<measure>:\t which measure to use, standard options are:\n";
        for (int n = 0; n < mss.size(); ++n) {
            *_os << "\t\t\"" << mss[n]->shortname << "\": " << mss[n]->name << "\n";
        }
        *_os << "\t" << "z=<tally>:\t which tally to use, standard options are:\n";
        for (int n = 0; n < tys.size(); ++n) {
            *_os << "\t\t\"" << tys[n]->shortname << "\": " << tys[n]->name << "\n";
        }
        *_os << "\t" << "e=<set>:\t which set to use, standard options are:\n";
        for (int n = 0; n < sts.size(); ++n) {
            *_os << "\t\t\"" << sts[n]->shortname << "\": " << sts[n]->name << "\n";
        }
        *_os << "\t" << "p=<tuple>:\t which tuple to use, standard options are:\n";
        for (int n = 0; n < oss.size(); ++n) {
            *_os << "\t\t\"" << oss[n]->shortname << "\": " << oss[n]->name << "\n";
        }
        *_os << "\t" << "gm=<tuple>:\t which graph to use, standard options are:\n";
        for (int n = 0; n < gms.size(); ++n) {
            *_os << "\t\t\"" << gms[n]->shortname << "\": " << gms[n]->name << "\n";
        }
    }

    void execute(std::vector<std::string> args) override
    {
        std::vector<int> items {}; // a list of indices within workspace of the graph items to FP and sort

        // clear();

        bool takeallgraphitems = false;
        int numofitemstotake = 1;
        bool sortedbool = false;
        bool sortedverify = false;
        bool andmode = false;
        bool ormode = false;
        bool takeallsubitems = false;

        sentences.clear();
        formulae.clear();

        std::vector<std::pair<std::string,std::string>> parsedargs = cmdlineparseiterationtwo(args);

        std::vector<graphitem*> flaggraphitems {};

        std::vector<FP*> fps {};
        std::vector<int> dims {};
        std::vector<neighbors*> nss {};
        std::vector<FP*> fpsc {};
        std::vector<int> dimsc {};
        std::vector<neighbors*> nssc {};


        for (int i = 0; i < parsedargs.size(); ++i) {
            if (parsedargs[i].first == "default" && parsedargs[i].second  == CMDLINE_ALL) {
                takeallgraphitems = true;
                sortedbool = false;
                continue;
            }
            if ((parsedargs[i].first == "default") && (parsedargs[i].second == CMDLINE_ENUMISOSSORTED)) {
                sortedbool = true;
                takeallgraphitems = true;
                continue;
            }
            if ((parsedargs[i].first == "default") && (parsedargs[i].second == CMDLINE_SUBOBJECTS)) {
                sortedbool = false;
                takeallgraphitems = false;
                takeallsubitems = true;
                continue;
            }


            compactcmdline ccl;
            namedparams paramnames {};
            if (parsedargs[i].first == "default")
            {
                ccl.t = "c";
                ccl.n = false;
                ccl.i = 0;
            } else
            {
                ccl = parsecompactcmdline(parsedargs[i].first);
            }

            if (ccl.t == "j") // j is a letter borrowed from cmake
            {
                int n = stoi(parsedargs[i].second);
                setthread_count( n < std::thread::hardware_concurrency() ? n : std::thread::hardware_concurrency() );
                std::cout << "Criterion thread count: " << thread_count << "\n";
            }
            if (ccl.t == "s")
            {
                std::string s = bindformula(parsedargs[i].second,mtbool,ccl.i);
                ams a;
                a.t = measuretype::mtbool;
                a.a.cs = new sentofcrit(&rec,litnumps,littypes,litnames, paramnames,s);
                a.a.cs->negated = ccl.n;
                auto it = newiteration(mtbool,ccl.i,a);
                iter.push_back(it);
                litnames.push_back(s);
                litnumps.push_back(0);
                littypes.push_back(mtbool);
                continue;
            }
            if (ccl.t == "a")
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";

                std::string s = bindformula(parsedargs[i].second,mtcontinuous,ccl.i);
                ams a;
                a.t = measuretype::mtcontinuous;
                a.a.ms = new formmeas(&rec,litnumps,littypes,litnames, paramnames, s);
                auto it = newiteration(mtcontinuous,ccl.i,a);
                iter.push_back(it);
                litnames.push_back(s);
                litnumps.push_back(0);
                littypes.push_back(mtcontinuous);
                continue;
            }

            if (ccl.t == "z" || ccl.t == "i") // demoting the "i" option in favor of "z"
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";

                std::string s = bindformula(parsedargs[i].second,mtdiscrete,ccl.i);
                ams a;
                a.t = measuretype::mtdiscrete;
                a.a.ts = new formtally(&rec,litnumps,littypes,litnames, paramnames, s);
                auto it = newiteration(mtdiscrete,ccl.i,a);
                iter.push_back(it);
                litnames.push_back(s);
                litnumps.push_back(0);
                littypes.push_back(mtdiscrete);
                continue;
            }

            if (ccl.t == "gm") // graph measure
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";

                std::string s = bindformula(parsedargs[i].second,mtgraph,ccl.i);
                ams a;
                a.t = measuretype::mtgraph;
                a.a.gs = new formgraph(&rec,litnumps,littypes,litnames, paramnames, s);
                auto it = newiteration(mtgraph,ccl.i,a);
                iter.push_back(it);
                litnames.push_back(s);
                litnumps.push_back(0);
                littypes.push_back(mtgraph);
                continue;
            }

            if (ccl.t == "e")
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";

                std::string s = bindformula(parsedargs[i].second,mtset,ccl.i);
                ams a;
                a.t = measuretype::mtset;
                a.a.ss = new formset(&rec,litnumps,littypes,litnames, paramnames, s);
                auto it = newiteration(mtset,ccl.i,a);
                iter.push_back(it);
                litnames.push_back(s);
                litnumps.push_back(0);
                littypes.push_back(mtset);
                continue;
            }

            if (ccl.t == "p")
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";

                std::string s = bindformula(parsedargs[i].second,mttuple,ccl.i);
                ams a;
                a.t = measuretype::mttuple;
                a.a.os = new formtuple(&rec,litnumps,littypes,litnames,paramnames,s);
                auto it = newiteration(mttuple,ccl.i,a);
                iter.push_back(it);
                litnames.push_back(s);
                litnumps.push_back(0);
                littypes.push_back(mttuple);
                continue;
            }


            if (ccl.t == "sp") // Stored procedure
            {
                addstoredproc(parsedargs[i].second);
                continue;
            }

            if (ccl.t == "isp") // Stored procedures from a file
            {
                std::vector<std::string> filedata;
                readfromfileandremovecomments(parsedargs[i].second, filedata );

                std::vector<std::vector<std::string>> tmp {};
                for (auto d : filedata)
                {
                    std::vector<std::string> tmp2;
                    tmp2.clear();
                    tmp2.push_back(d);
                    tmp.push_back(tmp2);
                }

                for (auto t : tmp)
                    for (auto s : t)
                        addstoredproc(s);

                // for (auto t : tmp) {
                //    for (auto s : t)
                //        std::cout << s << ", ";
                //    std::cout << std::endl;
                // }
                // std::cout << std::endl;

//                exit(1);


                continue;
            }

#ifdef FLAGCALCWITHPYTHON
            if (ccl.t == "ipy") // Python methods from a file
            {
                std::string filename = parsedargs[i].second;

                // py::scoped_interpreter guard{}; // won't outlast the scope


                try {

                    if (!is_pyinterpreterstarted)
                        py::initialize_interpreter();
                    is_pyinterpreterstarted = true;
                    py::module_ sys = py::module_::import("sys");
                    sys.attr("path").attr("append")("../python");


                    auto pymodule = py::module_::import(filename.c_str());

                    std::cout << "Opening Python file " << filename << "...\n";

                    py::list method_list = py::cast<py::list>(py::eval("dir()", pymodule.attr("__dict__")));

                    std::vector<std::string> methods = py::cast<std::vector<std::string>>(method_list);

                    for (auto d : method_list) {

                        std::regex pattern(R"(__\w+__)");
                        std::smatch matches;

                        std::string s = py::str(d);
                        std::regex_search(s, matches, pattern);
                        if (matches.size() > 0)
                            continue;
                        if (py::isinstance<py::module_>(d))
                            continue; // not working...

                        // std::cout << " - " << std::string(py::str(d)) << "(";

                        try {
                            py::function callback_ = pymodule.attr(py::str(d));
                            py::module_ inspect_module = py::module::import("inspect");
                            py::object signature_obj = inspect_module.attr("signature")(callback_);
                            py::object parameters_dict = signature_obj.attr("parameters");
                            // Get the length of the parameters dictionary
                            auto num_params = py::len(parameters_dict); // num_params is a Py_ssize_t/long

                            pythonmethodstruct pym;

                            pym.name = py::str(d);
                            pym.m = pymodule;
                            pym.filename = filename;
                            // pym.a.t = mtcontinuous;
                            pym.a.t = mtuncast;
                            namedparams nps {};
                            namedparams ips {};

                            // std::cout << "The Python function has " << num_params << " arguments: ";
                            int i = 0;
                            int specialargumentcount = 0;
                            valms u;
                            u.t = mtuncast;
                            for (auto v : parameters_dict) {
                                std::string s = py::str(v);
                                // std::cout << s << ",";
                                if (s == "adjmatrix" || s == "dim"
                                    || s == "Edges" || s == "Nonedges"
                                    || s == "Neighborslist" || s == "Nonneighborslist"
                                    || s == "degrees"
                                    || s == "maxdegree") {
                                    valms w;
                                    if (s == "dim" || s == "maxdegree")
                                        w.t = mtdiscrete;
                                    else if (s == "Neighborslist" || s == "Nonneighborslist"
                                        || s == "degrees")
                                        w.t = mttuple;
                                    else
                                        w.t = mtset;
                                    ++specialargumentcount;
                                    ips.push_back({s,w});
                                }
                                if (i >= specialargumentcount)
                                    nps.push_back({s,u});
                                i++;
                            }
                            // std::cout << "\b)\n";

                            pym.nps = nps;
                            pym.ips = ips;
                            pym.iidx = -1;
                            pythonmethods.push_back(pym);

                        } catch ( std::exception& e) { // need to figure out how simply to test for being a function or not
                            // also need to replace with the specific exception thrown
                            // std::cout << "\b: Ignoring " << py::str(d) << " (probably a module or otherwise not a function in Python)\n";
                        }
                    }
                    // py::object out = pymodule.attr("pytest")(3);
                    // double res = out.cast<double>();
                    // std::cout << " Hi : " << res << std::endl;

                    py::gil_scoped_release release;

                } catch (const py::error_already_set& e) {
                    std::cout << "Error with Python 'ipy' feature:" << e.what() << std::endl;
                    exit(1);
                }

            }

#endif



            if (ccl.t == "c")
            {
                bool found = false;
                std::vector<std::pair<std::string,std::vector<std::string>>> parsedargs2
                            = cmdlineparseiterationthree(parsedargs[i].second);
                for (int n = 0; !found && (n < crs.size()); ++n) {
                    for (int m = 0; !found && (m < parsedargs2.size()); ++m)
                    {
                        if (parsedargs2[m].first == crs[n]->shortname)
                        {
                            ams a;
                            a.t = mtbool;
                            if (lookupiter(parsedargs2[m].first) < 0)
                            {
                                a.a.cs = (*crsfactory[n])(&rec);
                                a.a.cs->negated = ccl.n;
                                iter.push_back( newiteration(mtbool,ccl.i,a));

                                litnumps.push_back(a.a.cs->npssz);
                                littypes.push_back(mtbool);
                                litnames.push_back(parsedargs2[m].first);

                                // litnumps.resize(iter.size());
                                // litnumps[iter.size()-1] = a.a.cs->pssz;
                                // littypes.resize(iter.size());
                                // littypes[iter.size()-1] = mtbool;
                                // litnames.resize(iter.size());
                                // litnames[iter.size()-1] = parsedargs2[m].first;

                                if (!parsedargs2[m].second.empty())
                                {
                                    // cs[cs.size()-1]->setparams(parsedargs2[m].second);
                                    // found = true;
                                    for (auto k = 0; k < parsedargs2[m].second.size(); ++k) {
                                        switch (a.a.cs->nps[k].second.t)
                                        {
                                        case measuretype::mtbool: a.a.cs->nps[k].second.v.bv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtdiscrete: a.a.cs->nps[k].second.v.iv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtcontinuous: a.a.cs->nps[k].second.v.dv = stof(parsedargs2[m].second[k]);
                                            break;
                                        }
                                    }
                                    iter[iter.size()-1]->nps = a.a.cs->nps;
                                }
                            }
                            found = true;
                        }
                    }
                }
                if (found)
                    continue;
            }

            if (ccl.t == "m")
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";
                bool found = false;
                std::vector<std::pair<std::string,std::vector<std::string>>> parsedargs2
                            = cmdlineparseiterationthree(parsedargs[i].second);
                for (int n = 0; !found && (n < mss.size()); ++n) {
                    for (int m = 0; !found && (m < parsedargs2.size()); ++m)
                    {
                        if (parsedargs2[m].first == mss[n]->shortname)
                        {
                            ams a;
                            a.t = mtcontinuous;
                            if (lookupiter(parsedargs2[m].first) < 0)
                            {
                                a.a.ms = (*mssfactory[n])(&rec);
                                iter.push_back( newiteration(mtcontinuous,ccl.i,a));


                                litnumps.push_back(a.a.ms->npssz);
                                littypes.push_back(mtcontinuous);
                                litnames.push_back(parsedargs2[m].first);



                                // litnumps.resize(iter.size());
                                // litnumps[iter.size()-1] = a.a.ms->pssz;
                                // littypes.resize(iter.size());
                                // littypes[iter.size()-1] = mtcontinuous;
                                // litnames.resize(iter.size());
                                // litnames[iter.size()-1] = parsedargs2[m].first;

                                if (!parsedargs2[m].second.empty())
                                {
                                    for (auto k = 0; k < parsedargs2[m].second.size(); ++k) {
                                        switch (a.a.ms->nps[k].second.t)
                                        {
                                        case measuretype::mtbool: a.a.ms->nps[k].second.v.bv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtdiscrete: a.a.ms->nps[k].second.v.iv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtcontinuous: a.a.ms->nps[k].second.v.dv = stof(parsedargs2[m].second[k]);
                                            break;
                                        }
                                    }
                                    iter[iter.size()-1]->nps = a.a.ms->nps;
                                }
                            }
                            found = true;
                        }
                    }
                }
                if (found)
                    continue;
            }

            if (ccl.t == "t")
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";
                bool found = false;
                std::vector<std::pair<std::string,std::vector<std::string>>> parsedargs2
                            = cmdlineparseiterationthree(parsedargs[i].second);
                for (int n = 0; !found && (n < tys.size()); ++n) {
                    for (int m = 0; !found && (m < parsedargs2.size()); ++m)
                    {
                        if (parsedargs2[m].first == tys[n]->shortname)
                        {
                            ams a;
                            a.t = mtdiscrete;
                            if (lookupiter(parsedargs2[m].first) < 0)
                            {
                                a.a.ts = (*tysfactory[n])(&rec);
                                iter.push_back( newiteration(mtdiscrete,ccl.i,a));


                                litnumps.push_back(a.a.ts->npssz);
                                littypes.push_back(mtdiscrete);
                                litnames.push_back(parsedargs2[m].first);

                                // litnumps.resize(iter.size());
                                // litnumps[iter.size()-1] = a.a.ts->pssz;
                                // littypes.resize(iter.size());
                                // littypes[iter.size()-1] = mtdiscrete;
                                // litnames.resize(iter.size());
                                // litnames[iter.size()-1] = parsedargs2[m].first;


                                if (!parsedargs2[m].second.empty())
                                {
                                    for (auto k = 0; k < parsedargs2[m].second.size(); ++k) {
                                        switch (a.a.ts->nps[k].second.t)
                                        {
                                        case measuretype::mtbool: a.a.ts->nps[k].second.v.bv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtdiscrete: a.a.ts->nps[k].second.v.iv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtcontinuous: a.a.ts->nps[k].second.v.dv = stof(parsedargs2[m].second[k]);
                                            break;
                                        }
                                    }
                                    iter[iter.size()-1]->nps = a.a.ts->nps;
                                }
                            }
                            found = true;
                        }
                    }
                }
                if (found)
                    continue;
            }

            if (ccl.t == "f")
            {

                std::vector<std::string> flagv {};
                flagv.push_back(parsedargs[i].second);



                graphitem* gi = new graphitem();
                gi->g = igraphstyle({parsedargs[i].second});
                gi->ns = new neighbors(gi->g);
                gi->name = _ws->getuniquename(gi->classname) + "FLAG";
                flaggraphitems.push_back(gi);
                //_ws->items.push_back(gi);
                int dim = gi->ns->g->dim;
                FP* fp = (FP*)malloc(dim*sizeof(FP));
                for (int j = 0; j < dim; ++j) {
                    fp[j].v=j;
                    fp[j].ns = nullptr;
                    fp[j].nscnt = dim;
                    fp[j].parent = nullptr;
                    fp[j].invert = gi->ns->degrees[j] >= int((dim+1)/2);
                }
                takefingerprint(gi->ns,fp,dim);

                fpsc.push_back(fp);
                nssc.push_back(gi->ns);
                dimsc.push_back(dim);

                ams a;
                a.t = mtbool;
                a.a.cs = new legacyembedscrit(&rec,gi->ns,fp);
                a.a.cs->negated = ccl.n;
                auto it = newiteration(mtbool,ccl.i,a);
                iter.push_back(it);

                litnumps.push_back(a.a.cs->npssz);
                littypes.push_back(mtbool);
                litnames.push_back(gi->name);


                // litnumps.resize(iter.size());
                // litnumps[iter.size()-1] = a.a.cs->pssz;
                // littypes.resize(iter.size());
                // littypes[iter.size()-1] = mtbool;
                // litnames.resize(iter.size());
                // litnames[iter.size()-1] = gi->name;

                continue;

            }


            if (ccl.t == "ft")
            {

                if (ccl.n)
                    std::cout << "No feature to negate here\n";


                std::vector<std::string> flagv {};
                flagv.push_back(parsedargs[i].second);



                graphitem* gi = new graphitem();
                gi->g = igraphstyle({parsedargs[i].second});
                gi->ns = new neighbors(gi->g);
                gi->name = _ws->getuniquename(gi->classname) + "FLAG";
                flaggraphitems.push_back(gi);
                //_ws->items.push_back(gi);
                int dim = gi->ns->g->dim;
                FP* fp = (FP*)malloc(dim*sizeof(FP));
                for (int j = 0; j < dim; ++j) {
                    fp[j].v=j;
                    fp[j].ns = nullptr;
                    fp[j].nscnt = dim;
                    fp[j].parent = nullptr;
                    fp[j].invert = gi->ns->degrees[j] >= int((dim+1)/2);
                }
                takefingerprint(gi->ns,fp,dim);

                fpsc.push_back(fp);
                nssc.push_back(gi->ns);
                dimsc.push_back(dim);

                ams a;
                a.t = mtdiscrete;
                a.a.ts = new embedstally(&rec,gi->ns,fp);
                auto it = newiteration(mtdiscrete,ccl.i,a);
                iter.push_back(it);


                litnumps.push_back(a.a.ts->npssz);
                littypes.push_back(mtdiscrete);
                litnames.push_back(gi->name);



                // litnumps.resize(iter.size());
                // litnumps[iter.size()-1] = a.a.ts->pssz;
                // littypes.resize(iter.size());
                // littypes[iter.size()-1] = mtdiscrete;
                // litnames.resize(iter.size());
                // litnames[iter.size()-1] = gi->name;

                continue;

            }


            if (ccl.t == "is")
            {

                std::vector<std::string> filedata;

                readfromfile( parsedargs[i].second, filedata);

                for (auto q : filedata)
                {
                    std::string s = bindformula(q,mtbool,ccl.i);
                    ams a;
                    a.t = measuretype::mtbool;
                    a.a.cs = new sentofcrit(&rec,litnumps,littypes,litnames, paramnames, s);
                    a.a.cs->negated = ccl.n;
                    auto it = newiteration(mtbool,ccl.i,a);
                    iter.push_back(it);

                    litnumps.push_back(a.a.cs->npssz);
                    littypes.push_back(mtbool);
                    litnames.push_back(parsedargs[i].first);


                    // litnumps.resize(iter.size());
                    // litnumps[iter.size()-1] = a.a.cs->pssz;
                    // littypes.resize(iter.size());
                    // littypes[iter.size()-1] = mtbool;
                    // litnames.resize(iter.size());
                    // litnames[iter.size()-1] = parsedargs[i].second;

                }
                continue;
            }

            if (ccl.t == "ia")
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";

                std::vector<std::string> filedata;

                readfromfile( parsedargs[i].second, filedata);

                for (auto q : filedata)
                {
                    std::string s = bindformula(q,mtcontinuous,ccl.i);
                    ams a;
                    a.t = measuretype::mtcontinuous;
                    a.a.ms = new formmeas(&rec,litnumps,littypes,litnames,paramnames,s);
                    auto it = newiteration(mtcontinuous,ccl.i,a);
                    iter.push_back(it);

                    litnumps.push_back(a.a.ms->npssz);
                    littypes.push_back(mtcontinuous);
                    litnames.push_back(parsedargs[i].second);

                    // litnumps.resize(iter.size());
                    // litnumps[iter.size()-1] = a.a.ms->pssz;
                    // littypes.resize(iter.size());
                    // littypes[iter.size()-1] = mtcontinuous;
                    // litnames.resize(iter.size());
                    // litnames[iter.size()-1] = parsedargs[i].second;

                }
                continue;

            }
            if (ccl.t == "if")
            {

                std::vector<std::string> filedata;
                readfromfile(parsedargs[i].second, filedata );

                std::vector<std::vector<std::string>> tmp {};
                for (auto d : filedata)
                {
                    std::vector<std::string> tmp2;
                    tmp2.clear();
                    tmp2.push_back(d);
                    tmp.push_back(tmp2);
                }

                int n = 0;
                graphitem* gi = new graphitem();
                while (n < tmp.size() && gi->isitemstr(tmp[n++])) {
                    gi->name = _ws->getuniquename(gi->classname) + "FLAG";
                    flaggraphitems.push_back(gi);
                    //_ws->items.push_back(gi);
                    int dim = gi->ns->g->dim;
                    FP* fp = (FP*)malloc(dim*sizeof(FP));
                    for (int j = 0; j < dim; ++j) {
                        fp[j].v=j;
                        fp[j].ns = nullptr;
                        fp[j].nscnt = dim;
                        fp[j].parent = nullptr;
                        fp[j].invert = gi->ns->degrees[j] >= (dim+1)/2;
                    }
                    takefingerprint(gi->ns,fp,dim);

                    fps.push_back(fp);
                    nss.push_back(gi->ns);
                    dims.push_back(dim);
                    gi = new graphitem();
                }
                delete gi;

                for (int i = 0; i < fps.size(); ++i)
                {
                    ams a;
                    a.t = mtbool;
                    a.a.cs = new legacyembedscrit(&rec,nss[i],fps[i]);
                    a.a.cs->negated = ccl.n;
                    auto it = newiteration(mtbool,ccl.i,a);
                    iter.push_back(it);

                    litnumps.push_back(a.a.cs->npssz);
                    littypes.push_back(mtbool);
                    litnames.push_back(gi->name);


                    // litnumps.resize(iter.size());
                    // litnumps[iter.size()-1] = a.a.cs->pssz;
                    // littypes.resize(iter.size());
                    // littypes[iter.size()-1] = mtbool;
                    // litnames.resize(iter.size());
                    // litnames[iter.size()-1] = gi->name;

                }
                continue;
           }
        }

        if (takeallsubitems)
        {
            int idx = _ws->items.size();
            while (idx > 0) {
                idx--;
                if (_ws->items[idx]->classname == "SUBOBJECT")  {
                    items.push_back(idx);
                    --numofitemstotake;

                } else {
                    //std::cout << idx << ": " << _ws->items[idx]->classname << "\n";
                }
            }
        }

        if (!takeallgraphitems && !takeallsubitems) {
            int idx = _ws->items.size();
            while (idx > 0 && (numofitemstotake > 0)) {
                idx--;
                if (_ws->items[idx]->classname == "GRAPH")  {
                    items.push_back(idx);
                    --numofitemstotake;

                } else {
                    //std::cout << idx << ": " << _ws->items[idx]->classname << "\n";
                }
            }
        } else {
            if (!takeallsubitems) {
                for (int n = 0; n < _ws->items.size(); ++n)
                {
                    if (_ws->items[n]->classname == "GRAPH") {
                        items.push_back(n);
                    }
                }
            }
        }

        if (!sortedbool && !sortedverify)
            if (items.size() == 0) {
                std::cout << "No graphs available to check criterion\n";
                return;
            }

        std::vector<neighborstype*> nslist {};
        nslist.resize(items.size());
        std::vector<graphtype*> glist {};
        glist.resize(items.size());
        // code this to run takefingerprint only once if graph 1 = graph 2


        //std::vector<bool*> res {};


        std::vector<int> eqclass {};
        if (sortedbool && !takeallsubitems) {
            if (items.size()==0) {
                std::cout << "No graphs to check criterion over\n";
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
            eqclass.clear();
            eqclass.resize(items.size());
            for (int m = 0; m < items.size(); ++m) {
                eqclass[m] = m;
            }
        }

        for (int i = 0; i < eqclass.size(); ++i)
        {
            if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[i]]])) {
                nslist[i] = gi->ns;
                glist[i] = gi->g;
            } else
                std::cout << "Error dynamically casting to graphitem*\n";
        }

        if (iter.empty())
        {
            ams a;
            a.t = measuretype::mtbool;
            a.a.cs = (*crsfactory[0])(&rec);
            iter.push_back(newiteration(mtbool,0,a));
            litnames.push_back(a.a.cs->shortname);
            littypes.push_back(mtbool);
            litnumps.push_back(a.a.cs->npssz);
        }

        rec.gptrs = &glist;
        rec.nsptrs = &nslist;
        rec.setsize(eqclass.size());



        if (items.size() < 1)
        {
            std::cout << "No items to check criteria and measures on\n";
            return;
        }


        bool found = false;

        std::vector<bool> todo;
        bool alltodo;
        todo.resize(eqclass.size());
        for (int m = 0; m < todo.size();++m)
            todo[m] = true;
        alltodo = true;

        std::vector<bool> threadbool {};
        std::vector<int> threadint {};
        std::vector<double> threaddouble {};
        std::vector<setitr*> threadset {};
        std::vector<setitr*> threadtuple {};
        std::vector<std::string*> threadstring {};
        std::vector<neighborstype*> threadgraph {};
        std::vector<valms> threaduncast {};
        threadbool.resize(eqclass.size());
        threadint.resize(eqclass.size());
        threaddouble.resize(eqclass.size());
        threadset.resize(eqclass.size());
        threadtuple.resize(eqclass.size());
        threadstring.resize(eqclass.size());
        threadgraph.resize(eqclass.size());
        threaduncast.resize(eqclass.size());
        for (int k = 0; k < iter.size(); ++k)
        {
            int ilookup = rec.intlookup(iter[k]->iidx);
            ams alookup = rec.lookup(iter[k]->iidx);

            if (!litnumps.empty())
                if (litnumps[k] > 0 && iter[k]->hidden)
                    continue;

            if (iter[k]->nps.size() > 0)
                continue;

            params ps;
            ps.resize(0);

//            ps.resize(iter[k]->nps.size());
//            int i = 0;
//            for (auto np : iter[k]->nps)
//                ps[i++] = np.second;

            if (alltodo)
            {
                switch (iter[k]->t) {
                    case mtbool:
                    runthreads<bool>(ilookup,ps,rec.boolrecs);
                    break;
                    case mtdiscrete:
                    runthreads<int>(ilookup,ps,rec.intrecs);
                    break;
                    case mtcontinuous:
                    runthreads<double>(ilookup,ps,rec.doublerecs);
                    break;
                    case mtset:
                    runthreads<setitr*>(ilookup,ps,rec.setrecs);
                    break;
                    case mttuple:
                    runthreads<setitr*>(ilookup,ps,rec.tuplerecs);
                    break;
                    case mtstring:
                    runthreads<std::string*>(ilookup,ps,rec.stringrecs);
                    break;
                    case mtgraph:
                    runthreads<neighborstype*>(ilookup,ps,rec.graphrecs);
                    break;
                    case mtuncast:
                    runthreads<valms>(ilookup,ps,rec.uncastrecs);
                    break;
                }
            } else
            {
                switch (iter[k]->t) {
                    case mtbool:
                    runthreadspartial<bool>(ilookup,ps,rec.boolrecs,&todo);
                    break;
                    case mtdiscrete:
                    runthreadspartial<int>(ilookup,ps,rec.intrecs,&todo);
                    break;
                    case mtcontinuous:
                    runthreadspartial<double>(ilookup,ps,rec.doublerecs,&todo);
                    break;
                    case mtset:
                    runthreadspartial<setitr*>(ilookup,ps,rec.setrecs,&todo);
                    break;
                    case mttuple:
                    runthreadspartial<setitr*>(ilookup,ps,rec.tuplerecs,&todo);
                    break;
                    case mtstring:
                    runthreadspartial<std::string*>(ilookup,ps,rec.stringrecs,&todo);
                    break;
                    case mtgraph:
                    runthreadspartial<neighborstype*>(ilookup,ps,rec.graphrecs,&todo);
                    break;
                    case mtuncast:
                    runthreadspartial<valms>(ilookup,ps,rec.uncastrecs,&todo);
                    break;
                }
            }

            switch (iter[k]->t) {
                case mtbool:
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (alltodo || todo[m])
                        threadbool[m] = rec.boolrecs.fetch(m,ilookup, ps);
                }
                for (int m = 0; m < eqclass.size(); ++m)
                    rec.addliteralvalueb( iter[k]->iidx, m, threadbool[m]);
                break;
                case mtdiscrete:
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (alltodo || todo[m])
                        threadint[m] = rec.intrecs.fetch(m,ilookup, ps);
                }
                for (int m = 0; m < eqclass.size(); ++m)
                    rec.addliteralvaluei( iter[k]->iidx, m, threadint[m]);
                break;
                case mtcontinuous:
                if (iter[k]->t == mtcontinuous)
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (alltodo || todo[m])
                        threaddouble[m] = rec.doublerecs.fetch(m,ilookup, ps);
                }
                for (int m = 0; m < eqclass.size(); ++m)
                    rec.addliteralvalued( iter[k]->iidx, m, threaddouble[m]);
                break;
                case mtset:
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (alltodo || todo[m])
                        threadset[m] = rec.setrecs.fetch(m,ilookup, ps);
                }
                for (int m = 0; m < eqclass.size(); ++m)
                    rec.addliteralvalues( iter[k]->iidx, m, threadset[m]);
                break;
                case mttuple:
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (alltodo || todo[m])
                        threadtuple[m] = rec.tuplerecs.fetch(m,ilookup, ps);
                }
                for (int m = 0; m < eqclass.size(); ++m)
                    rec.addliteralvaluet( iter[k]->iidx, m, threadtuple[m]);
                break;
                case mtstring:
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (alltodo || todo[m])
                        threadstring[m] = rec.stringrecs.fetch(m,ilookup, ps);
                }
                for (int m = 0; m < eqclass.size(); ++m)
                    rec.addliteralvaluer( iter[k]->iidx, m, threadstring[m]);
                break;
                case mtgraph:
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (alltodo || todo[m])
                        threadgraph[m] = rec.graphrecs.fetch(m,ilookup, ps);
                }
                for (int m = 0; m < eqclass.size(); ++m)
                    rec.addliteralvalueg( iter[k]->iidx, m, threadgraph[m]);
                break;
                case mtuncast:
                    for (int m = 0; m < eqclass.size(); ++m)
                    {
                        if (alltodo || todo[m])
                            threaduncast[m] = rec.uncastrecs.fetch(m,ilookup, ps);
                    }
                    for (int m = 0; m < eqclass.size(); ++m)
                        rec.addliteralvalueu( iter[k]->iidx, m, threaduncast[m]);
                    break;
            }

            if (iter[k]->hidden)
                continue;

            switch (iter[k]->t)
            {
            case mtbool: {
                auto wi = new checkbooleanitem(*alookup.a.cs);
                populatewi<bool>(_ws, wi, threadbool,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        if (takeallsubitems)
                        {
                            // auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
                            if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                            {
                                gi->boolitems.push_back(new ameasoutcome<bool>(alookup.a.cs,gi,threadbool[m]));
                                wi->gnames[m] = gi->name;
                            }
                            else
                            {
                                std::cout << "Dynamic cast error to graphitem*\n";
                            }



                        } else
                        {
                            // auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
                            if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                            {
                                gi->boolitems.push_back(new ameasoutcome<bool>(alookup.a.cs,gi,threadbool[m]));
                                wi->gnames[m] = gi->name;
                            }
                            else
                            {
                                std::cout << "Dynamic cast error to graphitem*\n";
                            }
                        }
                    }
                }
                break;
            }
            case mtdiscrete:
            {
                auto wi = new checkdiscreteitem<int>(*alookup.a.ts);
                populatewi<int>(_ws, wi, threadint,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        // auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
                        if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                        {
                            gi->intitems.push_back(new ameasoutcome<int>(alookup.a.ts,gi,threadint[m]));
                            wi->gnames[m] = gi->name;
                        }
                        else
                        {
                            std::cout << "Dynamic cast error to graphitem*\n";
                        }
                    }
                }
                break;
            }
            case mtcontinuous:
            {
                auto wi = new checkcontinuousitem<double>(*alookup.a.ms);
                populatewi<double>(_ws, wi, threaddouble,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        // auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
                        if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                        {
                            gi->doubleitems.push_back(new ameasoutcome<double>(alookup.a.ms,gi,threaddouble[m]));
                            wi->gnames[m] = gi->name;
                        }
                        else
                        {
                            std::cout << "Dynamic cast error to graphitem*\n";
                        }
                    }
                }
                break;
            }
            case mtset:
            {
                auto wi = new checksetitem<setitr*>(*alookup.a.ss);
                populatewi<setitr*>(_ws, wi, threadset,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        // auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
                        if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                        {
                            gi->setitems.push_back(new setoutcome<setitr*>(alookup.a.ss,gi,threadset[m]));
                            wi->gnames[m] = gi->name;
                        }
                        else
                        {
                            std::cout << "Dynamic cast error to graphitem*\n";
                        }
                    }
                }
                break;
            }
            case mttuple:
            {
                auto wi = new checktupleitem<setitr*>(*alookup.a.os);
                populatewi<setitr*>(_ws, wi, threadtuple,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        // auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
                        if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                        {
                            gi->tupleitems.push_back(new tupleoutcome<setitr*>(alookup.a.os,gi,threadset[m]));
                            wi->gnames[m] = gi->name;
                        }
                        else
                        {
                            std::cout << "Dynamic cast error to graphitem*\n";
                        }
                    }
                }
                break;
            }
            case mtstring:
            {
                auto wi = new checkstringitem<std::string*>(*alookup.a.rs);
                populatewi<std::string*>(_ws, wi, threadstring,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                        {
                            gi->stringitems.push_back(new stringoutcome<std::string*>(alookup.a.rs,gi,threadstring[m]));
                            wi->gnames[m] = gi->name;
                        }
                        else
                        {
                            std::cout << "Dynamic cast error to graphitem*\n";
                        }
                    }
                }
                break;
            }
            case mtgraph:
            {
                auto wi = new checkgraphitem<neighborstype*>(*alookup.a.gs);
                populatewi<neighborstype*>(_ws, wi, threadgraph,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[items[eqclass[m]]]))
                        {
                            gi->graphitems.push_back(new gmeasoutcome<neighborstype*>(alookup.a.gs,gi,threadgraph[m]));
                            wi->gnames[m] = gi->name;
                        }
                        else
                        {
                            std::cout << "Dynamic cast error to graphitem*\n";
                        }
                    }
                }
                break;
            }
            }

            if (k+1 < iter.size()) {
                int nextnonhidden = k+1;
                while (nextnonhidden < iter.size() && iter[nextnonhidden]->hidden)
                    ++nextnonhidden;
                if (nextnonhidden < iter.size() && iter[nextnonhidden]->round > iter[k]->round)
                {
                    switch (iter[k]->t) {
                    case mtbool:
                        for (int m = 0; m < threadbool.size();++m)
                            todo[m] = todo[m] && threadbool[m];
                         break;
                    case mtdiscrete:
                        for (int m = 0; m < threadint.size();++m)
                            todo[m] = todo[m] && (threadint[m] != 0);
                        break;
                    case mtcontinuous:
                        for (int m = 0; m < threaddouble.size();++m)
                            todo[m] = todo[m] && (abs(threaddouble[m]) > ABSCUTOFF);
                        break;
                    case mtset:
                        for (int m = 0; m < threadset.size(); ++m)
                            todo[m] = todo[m] && threadset[m]->getsize()>0;
                        break;
                    case mttuple:
                        for (int m = 0; m < threadtuple.size(); ++m)
                            todo[m] = todo[m] && threadtuple[m]->getsize()>0;
                        break;
                    case mtstring:
                        for (int m = 0; m < threadstring.size(); ++m)
                            todo[m] = todo[m] && threadstring[m]->size() > 0;
                        break;
                    case mtgraph:
                        for (int m = 0; m < threadgraph.size(); ++m)
                            todo[m] = todo[m] && threadgraph[m]->g->dim>0;
                        break;
                    }
                    alltodo = false;
                }
            }
        }


        for (int i = 0; i < fps.size(); ++i) {
            freefps(fps[i],dims[i]);
            free(fps[i]);
        }
        for (int i = 0; i < fpsc.size(); ++i) {
            freefps(fpsc[i],dimsc[i]);
            free(fpsc[i]);
        }

        for (auto gi : flaggraphitems)
            _ws->items.push_back(gi);

        for (auto i : iter)
            delete i;
        iter.clear();

    }
};


/*
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
        edgecntmeasure* ms = new edgecntmeasure();;
        double max = as->computeasymptotic(cr,ms,outof,limitdim, *_os, _ws);

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
*/

class pairwisedisjointrandomfeature : public feature {
protected:
    std::vector<pairwisedisjointrandomgraph*> rgs {};
    public:
    virtual void listoptions() override {
        feature::listoptions();
        *_os << "\t" << "<name>: \t\t\t first give the name of the graph to take subobjects from\n";
        *_os << "\t" << "t=\"vertices\": \t\t use the subgraph induced by the vertices \"type\" listed\n";
        *_os << "\t" << "fn=\"graph\": \t\t where fn is f1, f2, etc.: specify the given \"flags\"\n";
        *_os << "\t" << "r=<rgs>(<params>): \t randomly extend the subobjects up to the parameters provided\n";
        *_os << "\t\t\t\t\t where <rgs> is one of:\n";
        for (int n = 0; n < rgs.size(); ++n)
        {
            *_os << "\t\t\"" << rgs[n]->shortname() << "\": \t" << rgs[n]->name << "\n";
        }

    }


    std::string cmdlineoption() override {return "p";}
    std::string cmdlineoptionlong() override { return "pairwisedisjoint"; }
    pairwisedisjointrandomfeature( std::istream* is, std::ostream* os, workspace* ws )
        : feature( is, os, ws )
    {

        // add any new abstractrandomgraph types to the list here...

        auto rs1 = new uniformpairwisedisjointrandomgraph;
        rgs.push_back(rs1);

        // add any new abstract subobject types to the list here...

    }

    ~pairwisedisjointrandomfeature()
    {}


    std::vector<std::vector<graphtype*>> threadrandomgraphs( pairwisedisjointrandomgraph* r,
        std::vector<int> dims, graphtype* parentgi, std::vector<int>* subg, const int cnt)
    {
        std::vector<std::vector<graphtype*>> res {};
        if (cnt <= 0)
            return res;

        auto g = r->randomgraphs(dims,parentgi,subg, cnt, thread_count);
        for (auto r : g)
        {
            res.push_back(r);
        }
        return res;
    }





    void execute(std::vector<std::string> args) override {
        std::vector<std::pair<std::string,std::string>> parsedargs = cmdlineparseiterationtwo(args);


        int rgsidx = -1;
        std::vector<std::string> rgparams {};
        std::vector<graphitem*> flags {};
        std::vector<std::string> workingv {};
        std::vector<int> subg {};

        std::string giname;
        if (parsedargs.size() > 0)
            if (parsedargs[0].first == "default" || parsedargs[0].first == "g")
                giname = parsedargs[0].second;
            else
            {
                std::cout << "Error: use -p along with a graph name\n";
                return;
            }
        else
        {
            std::cout << "Error: use -p along with a graph name\n";
            return;
        }

        bool found = false;
        int i;
        for (i = 0; !found && (i < _ws->items.size()); ++i)
        {
            found = found || _ws->items[i]->name == giname;

        }
        graphitem* gi;
        if (found)
        {
            if (gi = dynamic_cast<graphitem*>(_ws->items[--i]))
            {
                for (int j = 1; j < parsedargs.size(); ++j)
                {
                    found = false;

                    if (parsedargs[j].first == "t") {
                        auto vgs = new verticesforgraphstyle;
                        workingv = vgs->getvertices(parsedargs[j].second); // allow unsorted and allow repeats


                        // "getintvertices"
                        subg.clear();
                        for (auto s : workingv)
                        {
                            bool found = false;
                            for (auto i = 0; !found && (i < gi->g->vertexlabels.size()); ++i)
                            {
                                if (s==gi->g->vertexlabels[i])
                                {
                                    found = true;
                                    subg.push_back(i);
                                }
                            }
                        }


                        delete vgs;
                    }


                    if (parsedargs[j].first.size() > 1 && parsedargs[j].first[0] == 'f' && is_number(parsedargs[j].first.substr(1, parsedargs[j].first.size()-1))) {
                        int fn = stoi(parsedargs[j].first.substr(1, parsedargs[j].first.size()-1));
                        graphitem* gi = new graphitem();
                        gi->g = igraphstyle({parsedargs[j].second});
                        gi->ns = new neighbors(gi->g);
                        gi->name = _ws->getuniquename(gi->classname);
                        _ws->items.push_back(gi);
                        flags.resize(fn < flags.size() ? flags.size() : fn);
                        flags[fn-1] = gi;
                        continue;
                    }

                    if (parsedargs[j].first == "r")
                    {
                        auto parsedargs2 = cmdlineparseiterationthree(parsedargs[j].second);

                        for (int r = 0; r < parsedargs2.size(); ++r)
                        {
                            for (int l = 0; l < rgs.size(); ++l)
                            {
                                if (parsedargs2[r].first == rgs[l]->shortname())
                                {
                                    rgsidx = l;
                                    rgparams = parsedargs2[r].second;
                                    //for (int k = 0; k < rgparams.size(); ++k)
                                    //    std::cout << "rgparam " << rgparams[k] << ", ";
                                    //std::cout << "\n";
                                }
                            }
                        }
                    }
                }
                //
            } else
                std::cout << "Error dynamically casting to graphitem*\n";
        } else
        {
            std::cout << "Unknown graph item named " << giname << "\n";
            return;
        }
        if (rgsidx >= 0)
        {
            std::vector<std::vector<graphtype*>> pdrg {};
            std::vector<std::vector<neighborstype*>> pdrns {};
            std::vector<int> dims;
            dims.resize(flags.size());
            for (int i = 0; i < flags.size(); ++i) {
                dims[i] = flags[i]->g->dim;
            }

            int outof = 0;
            if (rgparams.size() > 0)
                outof = stoi(rgparams[0]);

            // unsigned const thread_count = std::thread::hardware_concurrency();
            // unsigned const thread_count = 1;

            int cnt2 = 0;
            const double section = double(outof) / double(thread_count);
            std::vector<std::future<std::vector<std::vector<graphtype*>>>> t;
            t.resize(thread_count);
            for (int m = 0; m < thread_count; ++m) {
                const int startidx = int(m*section);
                const int stopidx = int((m+1.0)*section);

                t[m] = std::async(&pairwisedisjointrandomfeature::threadrandomgraphs,this,
                    rgs[rgsidx], dims, gi->g, &subg, stopidx-startidx);

            }
            for (int m = 0; m < thread_count; ++m)
            {
                auto w = t[m].get();
                for (auto r : w)
                {
                    pdrg.push_back(r);
                }
            }

            int cnt = 0;
            std::vector<int*> m1s;
            std::vector<int*> m2s;
            m1s.resize(flags.size());
            for (int j = 0; j < flags.size(); ++j) {
                int dim = flags[j]->g->dim;
                m1s[j] = (int*)malloc(dim*sizeof(int));
                memset(m1s[j],0,dim*sizeof(int));
                for (int i = 0; i < subg.size(); ++i) {
                    for (int k = 0; k < dim; ++k) {
                        if (subg[i] == k)
                            m1s[j][k] = subg[i] + 1;
                    }
                }
            }
            m2s.resize(flags.size());
            for (int j = 0; j < flags.size(); ++j) {
                int dim = flags[j]->g->dim;
                m2s[j] = (int*)malloc(dim*sizeof(int));
                memset(m2s[j],0,dim*sizeof(int));
                for (int i = 0; i < subg.size(); ++i) {
                    for (int k = 0; k < dim; ++k) {
                        if (subg[i] == k)
                            m2s[j][k] = subg[i] + 1;
                    }
                }
            }
            pdrns.resize(pdrg.size());
            for (int i = 0; i < pdrg.size(); ++i) {
                pdrns[i].resize(pdrg[i].size());
                for (int k = 0; k < pdrg[i].size(); ++k) {
                    pdrns[i][k] = new neighborstype(pdrg[i][k]);
                }
                bool match = true;
                for (int j = 0; match && (j < flags.size()); ++j) {
                    match = match && existsiso2(m1s[j],m2s[j],flags[j]->g,flags[j]->ns,pdrg[i][j],pdrns[i][j]);
                }
                cnt += match ? 1 : 0;
            }

            auto wi = new pairwisedisjointitem();
            wi->cnt = cnt;
            wi->total = pdrg.size();
            _ws->items.push_back(wi);


            for (auto p : pdrg)
                for (auto g : p)
                    delete g;
            for (auto n : pdrns )
                for (auto ns : n)
                    delete ns;
            pdrg.clear();
            pdrns.clear();
        }
    }


};


class populatesubobjectfeature : public feature
{
protected:
    std::vector<abstractsubobjectitem*> asois {};
    std::vector<abstractsubobjectitem*(*)(graphitem*,std::string)> asoifactory {};
    std::vector<abstractparameterizedsubrandomgraph*> rgs {};
    std::vector<abstractsubobjectitem*> sois {};
    std::vector<int> soistypes {};

public:
    virtual void listoptions() override {
        feature::listoptions();

        *_os << "\t" << "<name>: \t\t\t first give the name of the graph to take subobjects from\n";
        *_os << "\t" << "n=\"vertices\": \t\t use the subgraph induced by the vertices listed\n";
        *_os << "\t" << "r=<rgs>(<params>): \t randomly extend the subobjects up to the parameters provided\n";
        *_os << "\t\t\t\t\t where <rgs> is one of:\n";
        for (int n = 0; n < rgs.size(); ++n)
        {
            *_os << "\t\t\"" << rgs[n]->shortname() << "\": \t" << rgs[n]->name << "\n";
        }
    }

    std::vector<abstractsubobjectitem*> subobjs {};
    std::string cmdlineoption() override {return "u";}
    std::string cmdlineoptionlong() override { return "subobject"; }
    populatesubobjectfeature( std::istream* is, std::ostream* os, workspace* ws )
        : feature( is, os, ws )
    {


        // add any new abstractrandomgraph types to the list here...

        auto rs1 = new legacyrandomsubgraph<legacystdrandomsubgraph>;
        rgs.push_back(rs1);

        // add any new abstract subobject types to the list here...

        auto (asoi1) = abstractsubobjectitemfactory<inducedsubgraphitem>;

        asoifactory.push_back(asoi1);

        for (int n = 0; n < asoifactory.size(); ++n)
        {
            asois.push_back(asoifactory[n](nullptr,""));
        }
    }

    ~populatesubobjectfeature()
    {
        for (auto a : asois)
            delete a;
        for (int i = 0; i < rgs.size();++i) {
            delete rgs[i];
        }
    }

    std::vector<workitems*> threadrandomgraphs( abstractparameterizedsubrandomgraph* r, const int i, graphitem* gi, const int dim, graphtype* parentgi, std::vector<int>* subg, const int cnt)
    {
        std::vector<workitems*> res {};
        if (cnt <= 0)
            return res;

        auto g = r->randomgraphs(dim,parentgi,subg, cnt);
        for (auto r : g)
        {
            auto s = (*asoifactory[soistypes[i]])(gi,sois[i]->str);
            s->intvertices = *subg;
            s->g = r;
            s->ns = new neighborstype(r);
            res.push_back(s);
        }
        return res;
    }


    void execute(std::vector<std::string> args) override
    {
        std::vector<std::pair<std::string,std::string>> parsedargs = cmdlineparseiterationtwo(args);


        int rgsidx = -1;
        std::vector<std::string> rgparams {};


        std::string giname;
        if (parsedargs.size() > 0)
            if (parsedargs[0].first == "default" || parsedargs[0].first == "g")
                giname = parsedargs[0].second;
            else
            {
                std::cout << "Error: use -u along with a graph name\n";
                return;
            }
        else
        {
            std::cout << "Error: use -u along with a graph name\n";
            return;
        }

        bool found = false;
        int i;
        for (i = 0; !found && (i < _ws->items.size()); ++i)
        {
            found = found || _ws->items[i]->name == giname;

        }
        graphitem* gi;
        if (found)
        {
            if (gi = dynamic_cast<graphitem*>(_ws->items[--i]))
            {
                for (int j = 1; j < parsedargs.size(); ++j)
                {
                    found = false;
                    for (int k = 0; k < asois.size(); ++k)
                    {
                        if (parsedargs[j].first == asois[k]->shortname)
                        {
                            auto soi = (*asoifactory[k])(gi,parsedargs[j].second);
                            sois.push_back(soi);
                            soistypes.resize(sois.size());
                            soistypes[sois.size()-1] = k;
                            found = true;
                            break;
                        }
                    }
                    if (found)
                        continue;

                    if (parsedargs[j].first == "r")
                    {
                        auto parsedargs2 = cmdlineparseiterationthree(parsedargs[j].second);

                        for (int r = 0; r < parsedargs2.size(); ++r)
                        {
                            for (int l = 0; l < rgs.size(); ++l)
                            {
                                if (parsedargs2[r].first == rgs[l]->shortname())
                                {
                                    rgsidx = l;
                                    rgparams = parsedargs2[r].second;
                                    //for (int k = 0; k < rgparams.size(); ++k)
                                    //    std::cout << "rgparam " << rgparams[k] << ", ";
                                    //std::cout << "\n";
                                }
                            }
                        }
                    }
                }
                //
            } else
                std::cout << "Error dynamically casting to graphitem*\n";
        } else
        {
            std::cout << "Unknown graph item named " << giname << "\n";
            return;
        }
        if (rgsidx >= 0)
        {

            for (int i = 0; i < sois.size(); ++i) {
                rgs[rgsidx]->setparams(rgparams);
                int outof = stoi(rgparams[1]);

                // unsigned const thread_count = std::thread::hardware_concurrency();
                //unsigned const thread_count = 1;

                int cnt2 = 0;
                const double section = double(outof) / double(thread_count);
                std::vector<std::future<std::vector<workitems*>>> t;
                t.resize(thread_count);
                for (int m = 0; m < thread_count; ++m) {
                    const int startidx = int(m*section);
                    const int stopidx = int((m+1.0)*section);

                    t[m] = std::async(&populatesubobjectfeature::threadrandomgraphs,this,
                        rgs[rgsidx],i, gi,stoi(rgparams[0]),
                        sois[i]->parentgi->g,&sois[i]->intvertices,stopidx-startidx);
                }
                for (int m = 0; m < thread_count; ++m)
                {
                    auto w = t[m].get();
                    for (auto r : w)
                    {
                        _ws->items.push_back(r);
                    }
                }
            }
        } else
        {
            for (int i = 0; i < sois.size(); ++i)
                _ws->items.push_back(sois[i]);
        }
        for (auto s : sois)
            delete s;
        sois.clear();
        soistypes.clear();
    }
};





class samplerandomgraphsfeature : public abstractrandomgraphsfeature {
public:
    double percent = -1;
    std::string cmdlineoption() {
        return "R";
    }
    std::string cmdlineoptionlong() {
        return "samplerandomgraphsforpairwiseequiv";
    }
    void listoptions() override {
        abstractrandomgraphsfeature::listoptions();

        *_os << "\t" << "all: \t\t\t\t sample the graphs on the workspace pairwise for isomorphism\n";
        *_os << "\t" << "sub: \t\t\t\t sample the subgraphs on the workspace pairwise for isomporphism\n";
        *_os << "\t" << "cnt=<n>: \t\t\t sample the last n graphs on the workspace pairwise for isomporphism\n";
        *_os << "\t" << "<outof>: \t\t\t how many samples to take\n";
    }

    samplerandomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractrandomgraphsfeature( is, os, ws) {}

    enum samplerandomgraphsmodes {small, smsub, smcnt };

    int sampleobjectsrandom( std::vector<int>* items, samplerandomgraphsmodes sm, const int cnt)
    {
        int res = 0;
        int sz = items->size();

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);

        if (sm == smsub)
        {
            for (int j = 0; j < cnt; ++j)
            {
                int i1 = int((double)dist10000(rng)*((double)sz)/(RANDOMRANGEdouble));
                int i2 = int((double)dist10000(rng)*((double)sz)/(RANDOMRANGEdouble));

                auto w1 = _ws->items[(*items)[i1]];
                auto w2 = _ws->items[(*items)[i2]];
                if (abstractsubobjectitem* a1 = dynamic_cast<abstractsubobjectitem*>(w1))
                    if (abstractsubobjectitem* a2 = dynamic_cast<abstractsubobjectitem*>(w2)) {
                        if (a1->intvertices.size() == a2->intvertices.size()) {
                            if (!a1->m) {
                                int dim = a1->g->dim;
                                a1->m = (int*)malloc(dim*sizeof(int));
                                memset(a1->m,0,dim*sizeof(int));
                                for (int i = 0; i < a1->intvertices.size(); ++i) {
                                    for (int j = 0; j < dim; ++j) {
                                        if (a1->intvertices[i] == j)
                                            a1->m[j] = a2->intvertices[i] + 1;
                                    }
                                }
                            }
                            if (!a2->m) {
                                int dim = a2->g->dim;
                                a2->m = (int*)malloc(dim*sizeof(int));
                                memset(a2->m,0,dim*sizeof(int));
                                for (int i = 0; i < a2->intvertices.size(); ++i) {
                                    for (int j = 0; j < dim; ++j) {
                                        if (a2->intvertices[i] == j)
                                            a2->m[j] = a1->intvertices[i] + 1;
                                    }
                                }
                            }
                            // if (a1->intvertices.size() == a2->intvertices.size()) {
                            // for (int i = 0; i < a1->intvertices.size(); ++i)
                            // m.push_back( {a1->intvertices[i],a2->intvertices[i]});
                            if (existsiso2( a1->m, a2->m, a1->g, a1->ns, a2->g, a2->ns ))
                                res++;

                        }
                    }
            }
        }
        if (sm == small || sm == smcnt)
        {
            for (int j = 0; j < cnt; ++j)
            {

                int i1 = int((double)dist10000(rng)*((double)sz)/(RANDOMRANGEdouble));
                int i2 = int((double)dist10000(rng)*((double)sz)/(RANDOMRANGEdouble));

                auto w1 = _ws->items[(*items)[i1]];
                auto w2 = _ws->items[(*items)[i2]];
                if (graphitem* a1 = dynamic_cast<graphitem*>(w1))
                    if (graphitem* a2 = dynamic_cast<graphitem*>(w2))
                    {
                        if (existsiso2( nullptr, nullptr, a1->g, a1->ns, a2->g, a2->ns ))
                            res++;
                    } else
                    {
                        std::cout << "Dynamic cast error \n";
                    }
                else
                {
                    std::cout << "Dynamic cast error \n";
                }
            }
        }
        return res;
    };


    void execute(std::vector<std::string> args) override {

        samplerandomgraphsmodes sm = small;
        int cnt = 1;
        int outof = 0;

        std::vector<std::pair<std::string,std::string>> parsedargs = cmdlineparseiterationtwo(args);
        for (int i = 0; i < parsedargs.size(); ++i)
        {
            if (parsedargs[i].first == "all" || (parsedargs[i].first == "default" && parsedargs[i].second == "all")) {
                sm = small;
                continue;
            }
            if (parsedargs[i].first == "sub" || (parsedargs[i].first == "default" && parsedargs[i].second == "sub")) {
                sm = smsub;
                continue;
            }

            if (parsedargs[i].first == "cnt" && is_number(parsedargs[i].second)) {
                sm = smcnt;
                cnt = stoi(parsedargs[i].second);
                continue;
            }

            if (is_number(parsedargs[i].first))
            {
                outof = stoi(parsedargs[i].first);
                continue;
            }

            if (parsedargs[i].first == "c" && is_number(parsedargs[i].second))
            {
                outof = stoi(parsedargs[i].second);
                continue;
            }
        }

        std::vector<int> items {};
        if (sm == small)
        {
            for (int i = 0; i < _ws->items.size(); ++i)
            {
                if (_ws->items[i]->classname == "GRAPH")
                    if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[i]))
                        items.push_back(i); // default to working with all graphitems
            }
        }
        if (sm == smsub) {
            for (int i = 0; i < _ws->items.size(); ++i)
            {
                if (_ws->items[i]->classname == "SUBOBJECT")
                    if (abstractsubobjectitem* asoi = dynamic_cast<abstractsubobjectitem*>(_ws->items[i]))
                        items.push_back(i); // default to working with all graphitems
            }
        }
        if (sm == smcnt)
        {
            int j = _ws->items.size();
            int i = j-1;
            int c = 0;
            while (c < cnt && i >= 0) {
                if (_ws->items[i]->classname == "GRAPH")
                {
                    if (graphitem* gi = dynamic_cast<graphitem*>(_ws->items[i]))
                        items.push_back(i); // default to working with all graphitems
                    ++c;
                }
                --i;
            }
        }

        if (outof == 0)
            outof = items.size();

        // unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        int cnt2 = 0;
        const double section = double(outof) / double(thread_count);
        std::vector<std::future<int>> t;
        t.resize(thread_count);
        for (int m = 0; m < thread_count; ++m) {
            const int startidx = int(m*section);
            const int stopidx = int((m+1.0)*section);

            t[m] = std::async(&samplerandomgraphsfeature::sampleobjectsrandom,this,&items,sm,stopidx-startidx);
        }
        for (int m = 0; m < thread_count; ++m)
        {
            auto i = t[m].get();
            cnt2 = cnt2 + i;
        }



        // int cnt = samplematchingrandomgraphs(rgs[rgsidx],dim,edgecnt, outof);
            // --- yet a third functionality: randomly range over connected graphs (however, the algorithm should be checked for the right sense of "randomness"
            // note the simple check of starting with a vertex, recursively obtaining sets of neighbors, then checking that all
            // vertices are obtained, is rather efficient too.
            // Note also this definition of "randomness" is not correct: for instance, on a graph on three vertices, it doesn't run all the way
            //  up to and including three edges; it stops as soon as the graph is connected, i.e. at two vertices.




        auto wi = new samplerandommatchinggraphsitem;
        wi->cnt = cnt2;
        wi->outof = outof;
        wi->name = _ws->getuniquename(wi->classname);

        wi-> percent = ((double)wi->cnt / (double)wi->outof);
        _ws->items.push_back(wi);

    }

};






#endif //FEATURE_H
