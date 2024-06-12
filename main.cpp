#include <iostream>

#include "graphs.h"
#include <fstream>
#include "Graph.cpp"
#include "Formatgraph.cpp"
#include "EdgesforHelly.cpp"
#include "prob.h"
#include "verboseio.h"


int main(int argc, char* argv[]) {

    std::ifstream ifs;
    std::istream* s = &std::cin;
    if (argc > 1) {
        std::string filename = argv[1];
        std::cout << "Opening file " << filename << "\n";
        ifs.open(filename);
        if (!ifs)
            std::cout << "Couldn't open file for reading \n";
        s = &ifs;
    } else {
        std::cout << "Enter a filename or enter T for terminal mode: ";
        std::string filename;
        std::cin >> filename;
        if (filename != "T") {
            ifs.open(filename);
            if (!ifs)
                std::cout << "Couldn't open file for reading \n";
            s = &ifs;
        }
    }

    auto V = new Vertices();
    auto EV = new Batchprocesseddata<strtype>();
    auto FV = new Formatvertices(V,EV);
    FV->pause();
    FV->readdata(*s);
    FV->resume();
    int sV = FV->size();
    for( int n = 0; n<sV; ++n)
        std::cout << n << "{" << FV->idata->getdata(n) << ":" << FV->edata->getdata(n) << "}, ";
    std::cout << "\b\b  \n";

    auto E = new EdgesforHelly();
    auto EE = new Batchprocesseddata<Edgestr>();
    auto FE = new Formatedges(E,EE);
    FE->pause();
    FE->readdata(*s);
    FE->setvertices(FV);
    FE->resume();
    int sE = FE->size();

    for (int m = 0; m < sE; ++m) {
        std::cout << "{" << FE->idata->getdata(m).first << ", " << FE->idata->getdata(m).second;
        std::cout << ":" << FE->edata->getdata(m).first << ", " << FE->edata->getdata(m).second << "}, ";
    }
    std::cout << "\b\b  \n";

    // repeat the code above for a second graph...

    auto V2 = new Vertices();
    auto EV2 = new Batchprocesseddata<strtype>();
    auto FV2 = new Formatvertices(V2,EV2);
    FV2->pause();
    FV2->readdata(*s);
    FV2->resume();
    int sV2 = FV2->size();
    for( int n = 0; n<sV2; ++n)
        std::cout << n << "{" << FV2->idata->getdata(n) << ":" << FV2->edata->getdata(n) << "}, ";
    std::cout << "\b\b  \n";

    auto E2 = new EdgesforHelly();
    auto EE2 = new Batchprocesseddata<Edgestr>();
    auto FE2 = new Formatedges(E2,EE2);
    FE2->pause();
    FE2->readdata(*s);
    FE2->setvertices(FV2);
    FE2->resume();
    int sE2 = FE2->size();

    for (int m = 0; m < sE2; ++m) {
        std::cout << "{" << FE2->idata->getdata(m).first << ", " << FE2->idata->getdata(m).second;
        std::cout << ":" << FE2->edata->getdata(m).first << ", " << FE2->edata->getdata(m).second << "}, ";
    }
    std::cout << "\b\b  \n";


    graph g1;
    g1.dim = V->size();

    graph g2;
    g2.dim = V2->size();

    // below is rather embarrassing hack around some omissions in the general purpose intent of the HellyTool code...
    // (recall that HellyTool does not make use of the vertexadjacency matrix, hence the lack of debugging around its use...)
    E->maxvertex = g1.dim-1;
    E->vertexadjacency = new bool[(g1.dim) * g1.dim];
    E->computevertexadjacency();
    g1.adjacencymatrix = E->vertexadjacency;

    E2->maxvertex = g2.dim-1;
    E2->vertexadjacency = new bool[(g2.dim) * g2.dim];
    E2->computevertexadjacency();
    g2.adjacencymatrix = E2->vertexadjacency;

    osadjacencymatrix( std::cout, g1 );
    osadjacencymatrix( std::cout, g2 );

    neighbors ns1;
    ns1 = computeneighborslist(g1);
    osneighbors(std::cout,ns1);

    neighbors ns2;
    ns2 = computeneighborslist(g2);
    osneighbors(std::cout,ns2);

    FP fps1[g1.dim];
    for (vertextype n = 0; n < g1.dim; ++n) {
        fps1[n].v = n;
        fps1[n].ns = nullptr;
        fps1[n].nscnt = 0;
        fps1[n].parent = nullptr;
    }

    takefingerprint(ns1,fps1,g1.dim);

    osfingerprint(std::cout,ns1,fps1,g1.dim);

    FP fps2[g1.dim];
    for (vertextype n = 0; n < g2.dim; ++n) {
        fps2[n].v = n;
        fps2[n].ns = nullptr;
        fps2[n].nscnt = 0;
        fps2[n].parent = nullptr;
    }

    takefingerprint(ns2,fps2,g2.dim);

    FP fpstmp1;
    fpstmp1.parent = nullptr;
    fpstmp1.ns = fps1;
    fpstmp1.nscnt = g1.dim;

    FP fpstmp2;
    fpstmp2.parent = nullptr;
    fpstmp2.ns = fps2;
    fpstmp2.nscnt = g2.dim;

    osfingerprint(std::cout,ns2,fps2,g2.dim);
    if (FPcmp(ns1,ns2,fpstmp1,fpstmp2) == 0) {
        std::cout << "Fingerprints MATCH\n";
    } else {
        std::cout << "Fingerprints DO NOT MATCH\n";
    }

    std::vector<graphmorphism> maps = enumisomorphisms(ns1,ns2);
    osgraphmorphisms(std::cout, maps);

    free(FE);
    free(EE);
    free(E);
    free(FV);
    free(EV);
    free(V);
    free(ns1.neighborslist);
    free(ns1.degrees);
    freefps(fps1, g1.dim);

    free(FE2);
    free(EE2);
    free(E2);
    free(FV2);
    free(EV2);
    free(V2);
    free(ns2.neighborslist);
    free(ns2.degrees);
    freefps(fps2, g2.dim);

    // --- separate functionality below... aiming to have a main menu feature, or addl command line options

    int outof = 1000;
    int dim = 5;
    float edgecnt = 2.0;
    if (argc >= 4) {
        dim = std::stoi(argv[2]);
        edgecnt = std::stof(argv[3]);
        outof = std::stoi(argv[4]);
    }

    stdrandomgraph* rg1 = new stdrandomgraph( edgecnt );
    samplematchingrandomgraphs(rg1,dim,outof,std::cout);
    // --- yet a third functionality: randomly range over connected graphs (however, the algorithm should be checked for the right sense of "randomness"
    // note the simple check of starting with a vertex, recursively obtaining sets of neighbors, then checking that all
    // vertices are obtained, is rather efficient too.
    // Note also this definition of "randomness" is not correct: for instance, on a graph on three vertices, it doesn't run all the way
    //  up to and including three edges; it stops as soon as the graph is connected, i.e. at two vertices.

    delete rg1;

    randomconnectedgraphfixededgecnt* rg2 = new randomconnectedgraphfixededgecnt( 8 );
    samplematchingrandomgraphs(rg2,dim,outof,std::cout);

    delete rg2;

    

    return 0;
}
