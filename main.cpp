#include <iostream>

#include "graphs.h"
#include <fstream>
#include "Graph.cpp"
#include "Formatgraph.cpp"
#include "EdgesforHelly.cpp"
#include "prob.h";



int main(int argc, char* argv[]) {
    std::cout << "Hello, World!" << std::endl;

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

    int cnt = 0;
    int outof = 1000;
    int dim = 7;
    int edgecnt = 16;
    graph g3;
    g3.dim = dim;
    g3.adjacencymatrix = (bool*)malloc(g3.dim * g3.dim * sizeof(bool));

    graph g4;
    g4.dim = dim;
    g4.adjacencymatrix = (bool*)malloc(g4.dim * g4.dim * sizeof(bool));

    for (int i = 0; i < outof; ++i) {
        randomgraph(&g3,edgecnt);
        //osadjacencymatrix(std::cout,g3);
        //std::cout << "\n";
        randomgraph(&g4,edgecnt);
        //osadjacencymatrix(std::cout,g4);

        neighbors ns3;
        ns3 = computeneighborslist(g3);
        //osneighbors(std::cout,ns3);

        neighbors ns4;
        ns4 = computeneighborslist(g4);
        //osneighbors(std::cout,ns4);

        FP fps3[g3.dim];
        for (vertextype n = 0; n < g3.dim; ++n) {
            fps3[n].v = n;
            fps3[n].ns = nullptr;
            fps3[n].nscnt = 0;
            fps3[n].parent = nullptr;
        }

        takefingerprint(ns3,fps3,g3.dim);

        //osfingerprint(std::cout,ns3,fps3,g3.dim);

        FP fps4[g4.dim];
        for (vertextype n = 0; n < g4.dim; ++n) {
            fps4[n].v = n;
            fps4[n].ns = nullptr;
            fps4[n].nscnt = 0;
            fps4[n].parent = nullptr;
        }

        takefingerprint(ns4,fps4,g4.dim);

        FP fpstmp3;
        fpstmp3.parent = nullptr;
        fpstmp3.ns = fps3;
        fpstmp3.nscnt = g3.dim;

        FP fpstmp4;
        fpstmp4.parent = nullptr;
        fpstmp4.ns = fps4;
        fpstmp4.nscnt = g4.dim;

        //osfingerprint(std::cout,ns4,fps4,g4.dim);
        if (FPcmp(ns3,ns4,fpstmp3,fpstmp4) == 0) {
            //std::cout << "Fingerprints MATCH\n";
            cnt++;
        } else {
            //std::cout << "Fingerprints DO NOT MATCH\n";
        }
        freefps(fps3, g3.dim);
        freefps(fps4, g4.dim);
        free(ns3.neighborslist);
        free(ns3.degrees);
        free(ns4.neighborslist);
        free(ns4.degrees);
    }
    std::cout << "Random probability of fingerprints matching is " << cnt << " out of " << outof << " == " << float(cnt)/float(outof) << "\n";

    free(g3.adjacencymatrix);
    free(g4.adjacencymatrix);

    return 0;
}
