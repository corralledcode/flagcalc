#include <chrono>
#include <iostream>

#include "graphs.h"
#include <fstream>

#include "asymp.h"
#include "prob.h"
#include "verboseio.h"
#include "feature.h"
#include "workspace.h"
#include <cstring>

int main(int argc, char* argv[]) {
    workspace* ws = new workspace();

    auto ug = new userguidefeature(&std::cin, &std::cout,ws);
    auto rg = new readgraphsfeature(&std::cin, &std::cout,ws);
    auto ei = new enumisomorphismsfeature(&std::cin, &std::cout,ws);
    auto cf = new cmpfingerprintsfeature(&std::cin, &std::cout,ws);
    auto lsr = new legacysamplerandomgraphsfeature(&std::cin, &std::cout,ws);
    auto sr = new samplerandomgraphsfeature(&std::cin, &std::cout,ws);
    auto mt = new mantelstheoremfeature(&std::cin,&std::cout,ws);
    auto mv = new mantelsverifyfeature(&std::cin,&std::cout, ws);
    auto vb = new verbosityfeature(&std::cin,&std::cout, ws);
    auto _sb = new _sandboxfeature(&std::cin,&std::cout, ws);
    auto pr = new randomgraphsfeature(&std::cin,&std::cout, ws);
    auto cw = new clearworkspacefeature(&std::cin, &std::cout,ws);
    auto cc = new checkcriterionfeature(&std::cin, &std::cout,ws);
    auto wg = new writegraphsfeature(&std::cin, &std::cout,ws);
    auto sbg = new populatesubobjectfeature(&std::cin, &std::cout,ws);

    std::vector<feature*> featureslist {};
    featureslist.push_back(ug);
    featureslist.push_back(rg);
    featureslist.push_back(ei);
    featureslist.push_back(cf);
    featureslist.push_back(lsr);
    featureslist.push_back(sr);
    featureslist.push_back(mt);
    featureslist.push_back(mv);
    featureslist.push_back(vb);
    featureslist.push_back(_sb);
    featureslist.push_back(pr);
    featureslist.push_back(cw);
    featureslist.push_back(cc);
    featureslist.push_back(wg);
    featureslist.push_back(sbg);

    ug->featureslist = featureslist;

    std::vector<std::vector<std::string>> args {};

    int i = 1;
    std::vector<std::string> tmpv {};

    // add a few lines to implement DEFAULTCMDLINE

    while (i < argc) {
        std::vector<std::string> newargs {};
        if (std::strlen(argv[i])>1 && argv[i][0] == '-'&& argv[i][1] == '-') {
            if (tmpv.size() > 0)
                args.push_back(tmpv);
            tmpv.clear();
            char* longname = new char[std::strlen(argv[i])];
            std::strcpy( longname, &argv[i][2]);
            tmpv.push_back(longname);
            ++i;
            delete longname;
        } else {
            if (std::strlen(argv[i])>0 && argv[i][0] == '-') {
                if (tmpv.size() > 0)
                    args.push_back(tmpv);
                tmpv.clear();
                std::string shortname;
                shortname = argv[i][1];
                tmpv.push_back(shortname);
                ++i;
            } else {
                if (tmpv.size() > 0)
                    tmpv.push_back(argv[i]);
                else {
                    tmpv.push_back( DEFAULTCMDLINESWITCH );
                    tmpv.push_back(argv[i]);
                }
                ++i;
            }
        }

    }

    if (tmpv.size()>0)
        args.push_back(tmpv);

    /*  test code to ensure right parsing of the command line switches
    for (int i = 0; i < args.size(); ++i) {
        for (int j = 0; j < args[i].size(); ++j) {
            std::cout << "args " << i<<", "<<j<<": "<<args[i][j]  << "\n";
        }
    }
    return 0;
    */


    for (int i = 0; i < args.size(); ++i) {
        int f = 0; // note that this setup defaults to the most recently added feature in the list above
        for (int n = 0; n < featureslist.size(); ++n) {
            if (featureslist[n]->cmdlineoption()[0] == args[i][0][0] || featureslist[n]->cmdlineoptionlong() == args[i][0]) {
                f = n;
            }
        }

        auto starttime = std::chrono::high_resolution_clock::now();

        featureslist[f]->execute(args[i]);

        auto stoptime = std::chrono::high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(stoptime - starttime);

        timedrunitem* tr = new timedrunitem();
        tr->duration = duration.count();
        tr->name = ws->getuniquename(tr->classname);
//        if (ws->items.size()>0) {
//            tr->name = "TimedRun" +  ws->items[ws->items.size()-1]->name;
//        }
        ws->items.push_back(tr);
    }


    for (int n = 0; n < featureslist.size(); ++n) {
        delete featureslist[n];
    }


    for (int n = 0; n < ws->items.size(); ++n) {
        ws->items[n]->freemem();  // figure out if this is a memory leak
        delete ws->items[n];
    }
    delete ws;

    return 0;

}
