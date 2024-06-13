#include <chrono>
#include <iostream>

#include "graphs.h"
#include <fstream>

#include "asymp.h"
#include "prob.h"
#include "verboseio.h"
#include "feature.h"
#include "workspace.h"


int main(int argc, char* argv[]) {
    workspace* ws = new workspace();

    enumisomorphismsfeature* ei = new enumisomorphismsfeature(&std::cin, &std::cout,ws);
    cmpfingerprintsfeature* cf = new cmpfingerprintsfeature(&std::cin, &std::cout,ws);
    samplerandomgraphsfeature* sr = new samplerandomgraphsfeature(&std::cin, &std::cout,ws);
    mantelstheoremfeature* mt = new mantelstheoremfeature(&std::cin,&std::cout,ws);
    mantelsverifyfeature* mv = new mantelsverifyfeature(&std::cin,&std::cout, ws);

    std::vector<feature*> featureslist {};
    featureslist.push_back(ei);
    featureslist.push_back(cf);
    featureslist.push_back(sr);
    featureslist.push_back(mt);
    featureslist.push_back(mv);

    int cnt; // count of how many args are consumed by executing the feature
    int idx = 1;
    if (argc <= 1) // that is, only the executable's name
        ei->execute(argc,argv,&cnt);
    else {
        while (idx < argc) {
            char shortname;
            std::string longname = "";
            if (std::strlen(argv[idx]) > 1) {
                if (argv[idx][0] == '-') {
                    if (argv[idx][1] == '-') {
                        if (std::strlen(argv[idx]) > 2) {
                            char longnametmp[std::strlen(argv[idx]-2)];
                            std::strcpy( longnametmp,&argv[idx][2]);
                            longname = longnametmp;
                        }
                    } else {
                        shortname = argv[idx][1];
                    }

                } else {
                    shortname = 'i'; // default to enumisomorphisms feature
                    idx--;
                }
                int f = 0;
                for (int n = 0; n < featureslist.size(); ++n) {
                    if (featureslist[n]->cmdlineoption()[0] == shortname || featureslist[n]->cmdlineoptionlong() == longname) {
                        f = n;
                        //std::cout << shortname << ", " << featureslist[n]->cmdlineoption() << ", " << featureslist[n]->cmdlineoptionlong() << ", " << longname << "\n";
                    }
                }
                //std::cout << argv[idx] << "\n";
                auto starttime = std::chrono::high_resolution_clock::now();

                featureslist[f]->execute(argc - idx, &(argv[idx]), &cnt);

                auto stoptime = std::chrono::high_resolution_clock::now();
                auto duration = duration_cast<std::chrono::microseconds>(stoptime - starttime);
                std::cout << "Time elapsed: " << float(duration.count())/1000000 << "\n";

            }
            idx += (cnt+1);
        }
    }
    for (int n = 0; n < featureslist.size(); ++n) {
        delete featureslist[n];
    }
        // --- separate functionality below... aiming to have a main menu feature, or addl command line options

    for (int n = 0; n < ws->items.size(); ++n) {
        ws->items[n]->freemem();
        delete ws->items[n];
    }
    delete ws;



    return 0;
}
