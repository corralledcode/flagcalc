//
// Created by peterglenn on 6/6/24.
//

#ifndef GRAPHS_H
#define GRAPHS_H
#include <ostream>
#include <vector>

using vertextype = int;


struct graph {
    vertextype dim;
    bool* adjacencymatrix;
};

struct neighbors {
    graph g;
    vertextype* neighborslist;
    int* degrees;
    int maxdegree;
};

struct FP {
    vertextype v;
    FP* ns; //neighbors
    int nscnt;  // note differs from parent's neighbor count because we are only considering non repeating walks
    FP* parent = nullptr;
};

using graphmorphism = std::vector<std::pair<vertextype,vertextype>>;

int cmpwalk( neighbors ns, FP w1, FP w2 );


int FPcmp( neighbors ns1, neighbors ns2, FP w1, FP w2 );

void sortneighbors( neighbors ns, FP* fps, int fpscnt );

void takefingerprint( neighbors ns, FP* fps, int fpscnt );

void freefps( FP* fps, int fpscnt );

neighbors computeneighborslist( graph g );

void sortneighborslist( neighbors* nsptr );

int seqtoindex( vertextype* seq, const int idx, const int sz );

bool isiso( graph g1, graph g2, graphmorphism map );

std::vector<graphmorphism> enumisomorphisms( neighbors ns1, neighbors ns2 );

int edgecnt( graph g );


//bool areisomorphic( graph g1, graph g2, neighbors ns1, neighbors ns2 );

void osfingerprint( std::ostream &os, neighbors ns, FP* fps, int fpscnt );

void osfingerprintminimal( std::ostream &os, neighbors ns, FP* fps, int fpscnt );


void osadjacencymatrix( std::ostream &os, graph g );

void osneighbors( std::ostream &os, neighbors ns );

void osedges( std::ostream &os, graph g );

void osgraphmorphisms( std::ostream &os, std::vector<graphmorphism> maps );

#endif //GRAPHS_H
