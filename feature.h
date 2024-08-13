//
// Created by peterglenn on 6/12/24.
//

#ifndef FEATURE_H
#define FEATURE_H
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <wchar.h>

#include "asymp.h"
// #include "measure.cpp"
#include "graphio.h"
#include "graphoutcome.h"
#include "graphs.h"
#include "prob.h"
#include "workspace.h"
#include "mantel.h"
#include "thread_pool.cpp"
#include "ameas.h"
#include "meas.cpp"
#include "probsub.cpp"

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
        std::cout << "\n";

        std::string replacement = "1";
        std::string result = std::regex_replace(f,r,replacement);
        std::cout << result << "\n";




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


        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

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
        *_os << "\t" << "<randomalgorithm>:\t which algorithm to use, standard options are r1,...,r5:\n";
        for (int n = 0; n < rgs.size(); ++n) {
            *_os << "\t\t\"" << rgs[n]->shortname() << "\": " << rgs[n]->name << "\n";
        }
    }

    randomgraphsfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractrandomgraphsfeature( is, os, ws) {
    }

    void execute(std::vector<std::string> args) override {








        abstractrandomgraphsfeature::execute(args);





        int cnt = 100;
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
        gv = randomgraphs(rgs[rgsidx],dim,edgecnt,cnt);

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

            unsigned const thread_count = std::thread::hardware_concurrency();
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


        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

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

public:
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
        auto (ac) = critfactory<acrit>;
        auto (ec) = critfactory<ecrit>;
        auto (eadjc) = critfactory<eadjcrit>;

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
        crsfactory.push_back(ac);
        crsfactory.push_back(ec);
        crsfactory.push_back(eadjc);

        // ...



        for (int n = 0; n < crsfactory.size(); ++n) {
            crs.push_back((*crsfactory[n])(&rec));
        }


        // add any new measure types to the list here...

        auto (ms1) = measfactory<boolmeas>;
        auto (ms2) = measfactory<dimmeas>;
        auto (ms3) = measfactory<edgecntmeas>;
        auto (ms4) = measfactory<avgdegreemeas>;
        auto (ms5) = measfactory<mindegreemeas>;
        auto (ms6) = measfactory<maxdegreemeas>;
        auto (ms7) = measfactory<legacygirthmeas>;
        auto (mc) = measfactory<maxcliquemeas>;
        auto (cnm) = measfactory<connectedmeas>;
        auto (rm) = measfactory<radiusmeas>;
        auto (circm) = measfactory<circumferencemeas>;
        auto (lcircm) = measfactory<legacycircumferencemeas>;
        auto (diamm) = measfactory<diametermeas>;
        auto (gm) = measfactory<girthmeas>;

        mssfactory.push_back(ms1);
        mssfactory.push_back(ms2);
        mssfactory.push_back(ms3);
        mssfactory.push_back(ms4);
        mssfactory.push_back(ms5);
        mssfactory.push_back(ms6);
        mssfactory.push_back(ms7);
        mssfactory.push_back(mc);
        mssfactory.push_back(cnm);
        mssfactory.push_back(rm);
        mssfactory.push_back(circm);
        mssfactory.push_back(lcircm);
        mssfactory.push_back(diamm);
        mssfactory.push_back(gm);

        // ,,,

        for (int n = 0; n < mssfactory.size(); ++n) {
            mss.push_back((*mssfactory[n])(&rec));
        }


        // add any new tally types to the list here...

        auto (Knt) = tallyfactory<Kntally>;
        auto (cyclet) = tallyfactory<cycletally>;
        auto (kappat) = tallyfactory<kappatally>;
        auto (vdt) = tallyfactory<vdtally>;
        auto (st) = tallyfactory<sizetally>;
        auto (pct) = tallyfactory<pctally>;
        auto (firstt) = tallyfactory<pairfirsttally>;
        auto (secondt) = tallyfactory<pairsecondtally>;
        auto (pst) = tallyfactory<psizetally>;

        tysfactory.push_back(Knt);
        tysfactory.push_back(cyclet);
        tysfactory.push_back(kappat);
        tysfactory.push_back(vdt);
        tysfactory.push_back(st);
        tysfactory.push_back(pct);
        tysfactory.push_back(firstt);
        tysfactory.push_back(secondt);
        tysfactory.push_back(pst);



        // ,,,

        for (int n = 0; n < tysfactory.size(); ++n) {
            tys.push_back((*tysfactory[n])(&rec));
        }




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




unsigned const thread_count = std::thread::hardware_concurrency();
//unsigned const thread_count = 1;


template<typename T>
void runthreads(const int iidx, const params& ps, thrrecords<T>& r )
{
    const double section = double(r.sz) / double(thread_count);
    std::vector<std::future<void>> t;
    t.resize(thread_count);
    for (int m = 0; m < thread_count; ++m) {
        const int startidx = int(m*section);
        const int stopidx = int((m+1.0)*section);
        t[m] = std::async(&thrrecords<T>::threadfetch,r,startidx,stopidx,iidx,ps);
    }
    for (int m = 0; m < thread_count; ++m)
    {
        t[m].get();
    }
}

template<typename T>
void runthreadspartial(const int iidx, const params& ps, thrrecords<T>& r, std::vector<bool>* todo )
{
    const double section = double(r.sz) / double(thread_count);
    std::vector<std::future<void>> t;
    t.resize(thread_count);
    for (int m = 0; m < thread_count; ++m) {
        const int startidx = int(m*section);
        const int stopidx = int((m+1.0)*section);
        t[m] = std::async(&thrrecords<T>::threadfetchpartial,r,startidx,stopidx,iidx,ps,todo);
    }
    for (int m = 0; m < thread_count; ++m)
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



class checkcriterionfeature : public abstractcheckcriterionfeature {
protected:





//    std::vector<iteration*> iter {};
    std::vector<itn*> iter {};
    std::vector<int> litnumps {};
    std::vector<measuretype> littypes {};

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
            }
            if (sn == sin)
                return iter[i]->iidx;

        }
        return -1;

    }

    itn* newiteration( measuretype mtin, int roundin, const ams ain, const bool hiddenin = false )
    {
        int j;
        auto resi = new itn;
        switch (mtin)
        {
        case measuretype::mtbool:
            j = rec.boolrecs.pmsv->size();
            rec.boolrecs.pmsv->push_back(ain.a.cs);
            break;
        case measuretype::mtdiscrete:
            j = rec.intrecs.pmsv->size();
            rec.intrecs.pmsv->push_back(ain.a.ts);
            break;
        case measuretype::mtcontinuous:
            j = rec.doublerecs.pmsv->size();
            rec.doublerecs.pmsv->push_back(ain.a.ms);
            break;
        }

        resi->t = mtin;
        resi->round = roundin;
        resi->iidx = rec.maxm()+1;
        rec.addm(resi->iidx,resi->t,j);
        resi->hidden = hiddenin;
        resi->ps.clear();
        return resi;

    }

    int addmeas(const std::string sin, const measuretype mtin, const int roundin )
    {
        int li = lookupiter(sin);
        if (li >= 0)
            return li;
        for (int i = 0; i < crs.size(); ++i)
        {
            if (sin == crs[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtbool;
                a.a.cs = (*crsfactory[i])(&rec);
                iter.push_back(newiteration(mtbool,roundin,a,crs[i]->pssz > 0));
                int j = iter.size()-1;
                litnumps.resize(j+1);
                litnumps[j] = crs[i]->pssz;
                littypes.resize(j+1);
                littypes[j] = a.t;
                return j;
            }
        }
        for (int i = 0; i < mss.size(); ++i)
        {
            if (sin == mss[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtcontinuous;
                a.a.ms = (*mssfactory[i])(&rec);
                iter.push_back(newiteration(mtcontinuous,roundin,a,mss[i]->pssz > 0));
                int j = iter.size()-1;
                litnumps.resize(j+1);
                littypes.resize(j+1);
                litnumps[j] = mss[i]->pssz;
                littypes[j] = a.t;
                return j;
            }
        }
        for (int i = 0; i < tys.size(); ++i)
        {
            if (sin == tys[i]->shortname)
            {
                ams a;
                a.t = measuretype::mtdiscrete;
                a.a.ts = (*tysfactory[i])(&rec);
                iter.push_back(newiteration(mtdiscrete,roundin,a,tys[i]->pssz > 0));
                int j = iter.size()-1;
                litnumps.resize(j+1);
                littypes.resize(j+1);
                litnumps[j] = tys[i]->pssz;
                littypes[j] = a.t;
                return j;
            }
        }

        return -1;
    }


    std::string bindformula( std::string sin, const measuretype mtin, const int roundin )
    {
        const std::regex r {"\\[([[:alpha:]]\\w*)\\]"};
        std::vector<std::string> out {};
        for (std::sregex_iterator p(sin.begin(),sin.end(),r); p!=std::sregex_iterator{}; ++p)
        {
            out.push_back((*p)[1]);
        }
        for (int i = 0; i < out.size(); ++i)
        {
            if (is_number(out[i]))
                continue;
            int idx = lookupiter(out[i]);
            if (idx < 0)
            {

                idx = addmeas( out[i],mtin, roundin );
            }
            std::string replacement = "[" + std::to_string(idx) + "]";
            std::string pattern = "\\[" + out[i] + "\\]";
            std::regex reg(pattern);
            sin = std::regex_replace(sin,reg,replacement);
        }
        // std::cout << sin << "\n";
        return sin;
    }



public:

    // std::vector<criterion*> cs {};
    // std::vector<measure*> ms {};
    // std::vector<std::pair<int,std::string>> mscombinations {};
    std::vector<std::string> sentences {};
    std::vector<std::string> formulae {};

    std::string cmdlineoption() override { return "a"; }
    std::string cmdlineoptionlong() { return "checkcriteria"; }
    checkcriterionfeature( std::istream* is, std::ostream* os, workspace* ws ) : abstractcheckcriterionfeature( is, os, ws) {}

    ~checkcriterionfeature() {
        /*
        for (int i = 0; i < cs.size(); ++i) {
           bool found = false;
           for (int j = 0; !found && j < crs.size(); ++j) {
               found = found || crs[j] == cs[i];
           }
           if (!found)
                delete cs[i];
        }

        for (int i = 0; i < ms.size(); ++i) {
            bool found = false;
            for (int j = 0; !found && j < mss.size(); ++j) {
                found = found || mss[j] == ms[i];
            }
            if (!found)
                delete ms[i];
        }
*/
    }
    void listoptions() override {
        abstractcheckcriterionfeature::listoptions();
        *_os << "\t" << "\"" << CMDLINE_ALL << "\": \t\t\t checks criteria for ALL graphs found on the workspace\n";
        *_os << "\t" << "\"" << CMDLINE_ENUMISOSSORTED << "\": \t\t checks criteria for each fingerprint-equivalent class\n";
        *_os << "\t" << "\t\t\t\t obtained by previous calls to \"-f\"\n";
        *_os << "\t" << "\"not=<n>\": \t\t applies the logical NOT to the criteria numbered n, prior to AND or OR\n";
        *_os << "\t" << "\"l=AND\": \t\t applies the logical AND to the criteria (\"m=\" is optional)\n";
        *_os << "\t" << "\"l=OR\": \t\t applies the logical OR to the criteria (\"m=\" is optional)\n";
        *_os << "\t" << "\"s=<sentence>\": applies the logical sentence inside the quotes to the criteria\n";
        *_os << "\t" << "\"is=<filename>\": applies the logical sentence in <filename> to the criteria\n";
        *_os << "\t" << "\"f=<graph>\": \t checks the criterion of <graph> embedding\n";
        *_os << "\t" << "\"if=<filename>\": applies the criteria of flag(s) in <filename> embedding\n";
        *_os << "\t" << "\"a=<expression>\": uses mathematical expression to serve as a measure\n";
        *_os << "\t" << "\"ia=<filename>\": uses the mathematical expression(s) in <filename> embedding\n";

        *_os << "\t" << "<criterion>:\t which criterion to use, standard options are:\n";
        for (int n = 0; n < crs.size(); ++n) {
            *_os << "\t\t\"" << crs[n]->shortname << "\": " << crs[n]->name << "\n";
        }
        *_os << "\t" << "m=<measure>:\t which measure to use, standard options are:\n";
        for (int n = 0; n < mss.size(); ++n) {
            *_os << "\t\t\"" << mss[n]->shortname << "\": " << mss[n]->name << "\n";
        }
        *_os << "\t" << "<tally>:\t which tally to use, standard options are:\n";
        for (int n = 0; n < tys.size(); ++n) {
            *_os << "\t\t\"" << tys[n]->shortname << "\": " << tys[n]->name << "\n";
        }
    }

    void execute(std::vector<std::string> args) override
    {
        std::vector<int> items {}; // a list of indices within workspace of the graph items to FP and sort


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

        std::vector<double*> variables;

        //rec = new mrecords;


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
            if (parsedargs[i].first == "default")
            {
                ccl.t = "c";
                ccl.n = false;
                ccl.i = 0;
            } else
            {
                ccl = parsecompactcmdline(parsedargs[i].first);
            }
            if (ccl.t == "s")
            {
                std::string s = bindformula(parsedargs[i].second,mtbool,ccl.i);
                ams a;
                a.t = measuretype::mtbool;
                a.a.cs = new sentofcrit(&rec,litnumps,littypes,s);
                a.a.cs->negated = ccl.n;
                auto it = newiteration(mtbool,ccl.i,a);
                iter.push_back(it);
                continue;
            }
            if (ccl.t == "a")
            {
                if (ccl.n)
                    std::cout << "No feature to negate here\n";

                std::string s = bindformula(parsedargs[i].second,mtcontinuous,ccl.i);
                ams a;
                a.t = measuretype::mtcontinuous;
                a.a.ms = new formmeas(&rec,litnumps,littypes,s);
                auto it = newiteration(mtcontinuous,ccl.i,a);
                iter.push_back(it);
                continue;
            }
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
                                litnumps.resize(iter.size());
                                litnumps[iter.size()-1] = a.a.cs->pssz;
                                littypes.resize(iter.size());
                                littypes[iter.size()-1] = mtbool;

                                if (!parsedargs2[m].second.empty())
                                {
                                    // cs[cs.size()-1]->setparams(parsedargs2[m].second);
                                    // found = true;
                                    for (auto k = 0; k < parsedargs2[m].second.size(); ++k) {
                                        switch (a.a.cs->ps[k].t)
                                        {
                                        case measuretype::mtbool: a.a.cs->ps[k].v.bv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtdiscrete: a.a.cs->ps[k].v.iv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtcontinuous: a.a.cs->ps[k].v.dv = stof(parsedargs2[m].second[k]);
                                            break;
                                        }
                                    }
                                    iter[iter.size()-1]->ps = a.a.cs->ps;
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
                                litnumps.resize(iter.size());
                                litnumps[iter.size()-1] = a.a.ms->pssz;
                                littypes.resize(iter.size());
                                littypes[iter.size()-1] = mtcontinuous;

                                if (!parsedargs2[m].second.empty())
                                {
                                    for (auto k = 0; k < parsedargs2[m].second.size(); ++k) {
                                        switch (a.a.ms->ps[k].t)
                                        {
                                        case measuretype::mtbool: a.a.ms->ps[k].v.bv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtdiscrete: a.a.ms->ps[k].v.iv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtcontinuous: a.a.ms->ps[k].v.dv = stof(parsedargs2[m].second[k]);
                                            break;
                                        }
                                    }
                                    iter[iter.size()-1]->ps = a.a.ms->ps;
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
                                litnumps.resize(iter.size());
                                litnumps[iter.size()-1] = a.a.ts->pssz;
                                littypes.resize(iter.size());
                                littypes[iter.size()-1] = mtdiscrete;

                                if (!parsedargs2[m].second.empty())
                                {
                                    for (auto k = 0; k < parsedargs2[m].second.size(); ++k) {
                                        switch (a.a.ts->ps[k].t)
                                        {
                                        case measuretype::mtbool: a.a.ts->ps[k].v.bv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtdiscrete: a.a.ts->ps[k].v.iv = stoi(parsedargs2[m].second[k]);
                                            break;
                                        case measuretype::mtcontinuous: a.a.ts->ps[k].v.dv = stof(parsedargs2[m].second[k]);
                                            break;
                                        }
                                    }
                                    iter[iter.size()-1]->ps = a.a.ts->ps;
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
                a.a.cs = new embedscrit(&rec,gi->ns,fp);
                a.a.cs->negated = ccl.n;
                auto it = newiteration(mtbool,ccl.i,a);
                iter.push_back(it);
                litnumps.resize(iter.size());
                litnumps[iter.size()-1] = a.a.cs->pssz;
                littypes.resize(iter.size());
                littypes[iter.size()-1] = mtbool;
                continue;

            }


            if (ccl.t == "ft")
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
                a.t = mtdiscrete;
                a.a.ts = new embedstally(&rec,gi->ns,fp);
                auto it = newiteration(mtdiscrete,ccl.i,a);
                iter.push_back(it);
                litnumps.resize(iter.size());
                litnumps[iter.size()-1] = a.a.ts->pssz;
                littypes.resize(iter.size());
                littypes[iter.size()-1] = mtdiscrete;

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
                    a.a.cs = new sentofcrit(&rec,litnumps,littypes,s);
                    a.a.cs->negated = ccl.n;
                    auto it = newiteration(mtbool,ccl.i,a);
                    iter.push_back(it);
                    litnumps.resize(iter.size());
                    litnumps[iter.size()-1] = a.a.cs->pssz;
                    littypes.resize(iter.size());
                    littypes[iter.size()-1] = mtbool;

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
                    a.a.ms = new formmeas(&rec,litnumps,littypes,s);
                    auto it = newiteration(mtcontinuous,ccl.i,a);
                    iter.push_back(it);
                    litnumps.resize(iter.size());
                    litnumps[iter.size()-1] = a.a.ms->pssz;
                    littypes.resize(iter.size());
                    littypes[iter.size()-1] = mtcontinuous;

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
                    a.a.cs = new embedscrit(&rec,nss[i],fps[i]);
                    a.a.cs->negated = ccl.n;
                    auto it = newiteration(mtbool,ccl.i,a);
                    iter.push_back(it);
                    litnumps.resize(iter.size());
                    litnumps[iter.size()-1] = a.a.cs->pssz;
                    littypes.resize(iter.size());
                    littypes[iter.size()-1] = mtbool;

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
        threadbool.resize(eqclass.size());
        threadint.resize(eqclass.size());
        threaddouble.resize(eqclass.size());
        for (int k = 0; k < iter.size(); ++k)
        {
            int ilookup = rec.intlookup(iter[k]->iidx);
            ams alookup = rec.lookup(iter[k]->iidx);
            if (k > 0)
            {
                if (iter[k-1]->t == mtbool)
                    for (int m = 0; m < eqclass.size(); ++m)
                        rec.addliteralvalueb( iter[k-1]->iidx, m, threadbool[m]);
                if (iter[k-1]->t == mtdiscrete)
                    for (int m = 0; m < eqclass.size(); ++m)
                        rec.addliteralvaluei( iter[k-1]->iidx, m, threadint[m]);
                if (iter[k-1]->t == mtcontinuous)
                    for (int m = 0; m < eqclass.size(); ++m)
                        rec.addliteralvalued( iter[k-1]->iidx, m, threaddouble[m]);
            }

            if (k > 0 && iter[k]->round > iter[k-1]->round)
            {
                // ...add here support for andmode and ormode
                if (iter[k-1]->t == mtbool)
                    for (int m = 0; m < threadbool.size();++m)
                        todo[m] = todo[m] && threadbool[m];
                if (iter[k-1]->t == mtdiscrete)
                    for (int m = 0; m < threadint.size();++m)
                        todo[m] = todo[m] && (threadint[m] != 0);
                if (iter[k-1]->t == mtcontinuous)
                    for (int m = 0; m < threaddouble.size();++m)
                        todo[m] = todo[m] && (abs(threaddouble[m]) > 0.00000001);

                alltodo = false;

            }

            if (iter[k]->hidden)
                continue;



            if (alltodo)
            {

                if (iter[k]->t == mtbool)
                    runthreads<bool>(ilookup,iter[k]->ps,rec.boolrecs);
                if (iter[k]->t == mtdiscrete)
                    runthreads<int>(ilookup,iter[k]->ps,rec.intrecs);
                if (iter[k]->t == mtcontinuous)
                    runthreads<double>(ilookup,iter[k]->ps,rec.doublerecs);

            } else
            {
                if (iter[k]->t == mtbool)
                    runthreadspartial<bool>(ilookup,iter[k]->ps,rec.boolrecs,&todo);
                if (iter[k]->t == mtdiscrete)
                    runthreadspartial<int>(ilookup,iter[k]->ps,rec.intrecs,&todo);
                if (iter[k]->t == mtcontinuous)
                    runthreadspartial<double>(ilookup,iter[k]->ps,rec.doublerecs,&todo);

            }

            if (iter[k]->t == mtbool)
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    threadbool[m] = rec.boolrecs.fetch(m,ilookup, iter[k]->ps);
                }

            if (iter[k]->t == mtdiscrete)
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    threadint[m] = rec.intrecs.fetch(m,ilookup, iter[k]->ps);
                }

            if (iter[k]->t == mtcontinuous)
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    threaddouble[m] = rec.doublerecs.fetch(m,ilookup, iter[k]->ps);
                }

            if (iter[k]->t == mtbool)
            {
                auto wi = new checkdiscreteitem<bool>(*alookup.a.cs);
                populatewi<bool>(_ws, wi, threadbool,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        if (takeallsubitems)
                        {
                            auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
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
                            auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
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
            }
            if (iter[k]->t == mtdiscrete)
            {
                auto wi = new checkdiscreteitem<int>(*alookup.a.ts);
                populatewi<int>(_ws, wi, threadint,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
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
            }
            if (iter[k]->t == mtcontinuous)
            {
                auto wi = new checkcontinuousitem<double>(*alookup.a.ms);
                populatewi<double>(_ws, wi, threaddouble,  items, eqclass,
                    glist, nslist, todo );
                for (int m = 0; m < eqclass.size(); ++m)
                {
                    if (todo[m])
                    {
                        auto gi = (graphitem*)_ws->items[items[eqclass[m]]];
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
    {

    }


    std::vector<std::vector<graphtype*>> threadrandomgraphs( pairwisedisjointrandomgraph* r,
        std::vector<int> dims, graphtype* parentgi, std::vector<int>* subg, const int cnt)
    {
        std::vector<std::vector<graphtype*>> res {};
        if (cnt <= 0)
            return res;

        auto g = r->randomgraphs(dims,parentgi,subg, cnt);
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

            unsigned const thread_count = std::thread::hardware_concurrency();
            //unsigned const thread_count = 1;

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

                unsigned const thread_count = std::thread::hardware_concurrency();
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

        unsigned const thread_count = std::thread::hardware_concurrency();
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
