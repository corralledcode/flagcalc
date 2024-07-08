//
// Created by peterglenn on 6/6/24.
//

#ifndef GRAPHS_H
#define GRAPHS_H
#include <iostream>
#include <ostream>
#include <vector>

using vertextype = int;
using vltype = std::string();


class graph {
public:
    int dim;
    bool* adjacencymatrix;
    graph(int dimin) : dim{dimin} {
        adjacencymatrix = (bool*)malloc(dim*dim*sizeof(bool));
    }
    ~graph() {
        delete adjacencymatrix;
    }
};

template< typename T >
class labelledgraph : public graph {
public:
    std::vector<T> vertexlabels;

    labelledgraph( int dim ) : graph( dim ) {
        //vertexlabels = (T*)malloc(dim*sizeof(T));
    }

};

using graphtype = labelledgraph<std::string>;
//using graphtype = labelledgraph<vltype>;

void osadjacencymatrix( std::ostream &os, graphtype* g );


class neighbors {
public:
    graphtype* g;
    int dim;
    int* neighborslist;
    int* degrees;
    int maxdegree;
    int* nonneighborslist;
    bool computeneighborslist() {
        maxdegree = -1;
        int nondegrees[dim];
        for (int n = 0; n < dim; ++n) {
            degrees[n] = 0;
            for (int i = 0; i < dim; ++i) {
                if (g->adjacencymatrix[n*dim + i]) {
                    neighborslist[n * dim + degrees[n]] = i;
                    degrees[n]++;
                }
            }
            maxdegree = (degrees[n] > maxdegree ? degrees[n] : maxdegree );
        }
        for (int n = 0; n < dim; ++n) {
            nondegrees[n] = 0;
            for (int i = 0; i < dim; ++i) {
                if (!g->adjacencymatrix[n*dim + i] && (n != i)) {
                    nonneighborslist[n*dim + nondegrees[n]] = i;
                    nondegrees[n]++;
                }
            }
        }
        for (int n = 0; n < dim; ++n) {
            if (degrees[n] + nondegrees[n] != dim-1) {
                std::cout << "BUG in computeneighborslist\n";
                osadjacencymatrix( std::cout, g);
                return false;
            }
        }

        return true;
    }
    neighbors(graphtype* gin) {
        g = gin;
        if (g == nullptr) {
            std::cout << "Can't create a neighbor object with a null graph\n";
            dim = 0;
            maxdegree=0;
            return;
        }
        dim = g->dim;
        maxdegree = 0;
        neighborslist = (vertextype*)malloc(dim * (dim) * sizeof(int));
        nonneighborslist = (vertextype*)malloc(dim * (dim) * sizeof(int));
        degrees = (int*)malloc(dim * sizeof(int) );
        computeneighborslist();
    }
    ~neighbors() {
        free(neighborslist);
        free(nonneighborslist);
        free(degrees);
    }

};

using neighborstype = neighbors;

struct FP {
    int v;
    FP* ns = nullptr; //neighbors
    int nscnt;  // note differs from parent's neighbor count because we are only considering non repeating walks
    FP* parent = nullptr;
    bool invert; // use non-neighbors if degree is more than half available
};

using graphmorphism = std::vector<std::pair<vertextype,vertextype>>;

int cmpwalk( neighbors ns, FP w1, FP w2 );


int FPcmp( const neighbors* ns1, const neighbors* ns2, const FP* w1, const FP* w2 );

void sortneighbors( neighbors* ns, FP* fps, int fpscnt );

void takefingerprint( const neighbors* ns, FP* fps, int fpscnt );

void freefps( FP* fps, int fpscnt );

/* superceded by class method of class neighbors
neighbors computeneighborslist( graphtypeg );
*/

void sortneighborslist( neighbors* nsptr );

int seqtoindex( vertextype* seq, const int idx, const int sz );

bool isiso( const graphtype* g1, const graphtype* g2, const graphmorphism* map );

std::vector<graphmorphism>* enumisomorphismscore( const neighborstype* ns1, const neighborstype* ns2, const FP* fps1ptr, const FP* fps2ptr );

std::vector<graphmorphism>* enumisomorphisms( neighbors* ns1, neighbors* ns2 );

int edgecnt( const graphtype* g );




bool existsisocore( const neighbors* ns1, const neighbors* ns2, const FP* fp1, const FP* fp2);

bool existsiso( const neighbors* ns1, FP* fps1ptr, const neighbors* ns2);

int nchoosek( const int n, const int k);

void enumsizedsubsets(int sizestart, int sizeend, int* seq, int start, int stop, std::vector<int>* res);

bool embeds( const neighbors* ns1, FP* fps1ptr, const neighbors* ns2, const int mincnt );

bool embedsquick( const neighbors* ns1, FP* fp, const neighbors* ns2, const int mincnt );



//bool areisomorphic( graphtype g1, graphtype g2, neighbors ns1, neighbors ns2 );

void osfingerprint( std::ostream &os, neighbors* ns, FP* fps, int fpscnt );

void osfingerprintminimal( std::ostream &os, neighbors* ns, FP* fps, int fpscnt );


void osadjacencymatrix( std::ostream &os, graphtype* g );

void osneighbors( std::ostream &os, neighbors* ns );

void osedges( std::ostream &os, graphtype* g );

void osgraphmorphisms( std::ostream &os, graphtype* g1, graphtype* g2, std::vector<graphmorphism>* maps );

void osmachinereadablegraph(std::ostream &os, graphtype* g);

void zerograph(graphtype* g);

graphtype* cyclegraph( const int dim );

#endif //GRAPHS_H
