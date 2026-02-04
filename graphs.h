//
// Created by peterglenn on 6/6/24.
//

#ifndef GRAPHS_H
#define GRAPHS_H
#include <iostream>
#include <ostream>
#include <vector>

#define KNMAXCLIQUESIZE 15
#define INDNMAXINDSIZE 15
#define GRAPH_PRECOMPUTECYCLESCNT 15


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

void osadjacencymatrix( std::ostream &os, const graphtype* g );


class neighbors {
public:
    graphtype* g;
    const int dim;
    int* neighborslist;
    int* degrees;
    int maxdegree;
    int* nonneighborslist;
    bool computeneighborslist();
    bool CPUcomputeneighborslist();
    neighbors(graphtype* gin) : dim{ gin->dim } {
        g = gin;
        if (g == nullptr) {
            std::cout << "Can't create a neighbor object with a null graph\n";
            //dim = 0;
            maxdegree=0;
            return;
        }
        //dim = g->dim;
        maxdegree = 0;
        neighborslist = (vertextype*)malloc(dim * dim * sizeof(vertextype));
        nonneighborslist = (vertextype*)malloc(dim * dim * sizeof(vertextype));
        degrees = (int*)malloc(dim * sizeof(int) );
        computeneighborslist();
    }
    neighbors(graphtype* gin, bool compute ) : dim{ gin->dim }
    {
        g = gin;
        if (g == nullptr) {
            std::cout << "Can't create a neighbor object with a null graph\n";
            //dim = 0;
            maxdegree=0;
            return;
        }
        // dim = g->dim;
        maxdegree = 0;
        neighborslist = (vertextype*)malloc(dim * dim * sizeof(vertextype));
        nonneighborslist = (vertextype*)malloc(dim * dim * sizeof(vertextype));
        degrees = (int*)malloc(dim * sizeof(int) );
        if (compute)
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

inline graphtype* edgegraph( neighborstype* ns )
{

    std::vector<std::pair<int, int>> neighbors {};
    for (int i = 0; i+1 < ns->dim; ++i)
        for (int j = i+1; j < ns->dim; ++j)
            if (ns->g->adjacencymatrix[i*ns->dim + j])
                neighbors.push_back(std::make_pair(i, j));
    int dim = neighbors.size();
    auto gout = new graphtype(dim);
    for (int k = 0; k+1 < dim; ++k)
    {
        gout->adjacencymatrix[k*dim + k] = false;
        for (int l = k+1; l < dim; ++l)
        {
            gout->adjacencymatrix[k*dim + l] = neighbors[k].first == neighbors[l].first
                                                || neighbors[k].first == neighbors[l].second
                                                || neighbors[k].second == neighbors[l].first
                                                || neighbors[k].second == neighbors[l].second;
            gout->adjacencymatrix[l*dim + k] = gout->adjacencymatrix[k*dim + l];
        }
    }
    gout->adjacencymatrix[(dim-1)*dim + dim-1] = false;
    return gout;
}

inline std::vector<std::pair<vertextype,vertextype>> graphedges( graphtype* g ) {
    std::vector<std::pair<vertextype,vertextype>> edges {};
    int dim = g->dim;
    for (vertextype i = 0; i+1 < dim; ++i)
        for (vertextype j = i+1; j < dim; ++j)
            if (g->adjacencymatrix[i*dim + j])
                edges.push_back(std::make_pair(i, j));
    return edges;
}

inline std::vector<std::pair<vertextype,vertextype>> graphnonedges( graphtype* g ) {
    std::vector<std::pair<vertextype,vertextype>> nonedges {};
    int dim = g->dim;
    for (vertextype i = 0; i+1 < dim; ++i)
        for (vertextype j = i+1; j < dim; ++j)
            if (!g->adjacencymatrix[i*dim + j])
                nonedges.push_back(std::make_pair(i, j));
    return nonedges;
}

using graphmorphism = std::vector<std::pair<vertextype,vertextype>>;

int cmpwalk( neighbors ns, FP w1, FP w2 );


int FPcmp( const neighbors* ns1, const neighbors* ns2, const FP* w1, const FP* w2 );

void sortneighbors( neighbors* ns, FP* fps, int fpscnt );

void takefingerprint( const neighbors* ns, FP* fps, int fpscnt, const bool useinvert = true );

FP* startfingerprint( const neighborstype& ns, bool useinvert = true );

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

int connectedcount( graphtype* g, neighborstype* ns, const int breaksize);



bool existsisocore( const neighbors* ns1, const neighbors* ns2, const FP* fp1, const FP* fp2);

bool existsiso( const neighbors* ns1, FP* fps1ptr, const neighbors* ns2);

bool existsiso2( const int* m1, const int* m2, const graphtype* g1, const neighbors* ns1, const graphtype* g2, const neighbors* ns2 );
// the only difference being that the latter has a different input set, and is more general

void enumsizedsubsets(int sizestart, int sizeend, int* seq, int start, int stop, std::vector<int>* res);

bool embeds( const neighbors* ns1, FP* fps1ptr, const neighbors* ns2, const int mincnt );

bool embedsquick( const neighbors* ns1, FP* fp, const neighbors* ns2, const int mincnt );

bool embedsgenerousquick( const neighbors* ns1, FP* fp, const neighbors* ns2, const int mincnt );
// i.e. non-induced (preserving adjacency but not necessarily non-adjacency

bool hastopologicalminorquick( const neighbors* ns1, const neighbors* ns2, const int mincnt );

bool hastopologicalminorquick2( const neighbors* childns, const neighbors* parentns, const int mincnt );

bool hastopologicalminorquick3( const neighbors* childns, const neighbors* parentns, const int mincnt );

bool hastopologicalminorquick4( const neighbors* ns1, const neighbors* ns2, const int mincnt );

bool hasminorquick( const neighbors* ns1, const neighbors* ns2, const int mincnt );

int embedscount( const neighbors* ns1, FP* fp, const neighbors* ns2);

int embedsgenerouscount( const neighbors* ns1, FP* fp, const neighbors* ns2);

// int hastopologicalminorcount( const neighbors* ns1, const neighbors* ns2 );

bool graphextendstotopologicalminor( const neighborstype& parentns,
        const vertextype* vertices, const neighborstype& childns, const int& mincnt );

bool kconnectedfn( graphtype* g, neighborstype* ns, int k );

bool ledgeconnectedfn( graphtype* g, neighborstype* ns, const int k );



//bool areisomorphic( graphtype g1, graphtype g2, neighbors ns1, neighbors ns2 );

void osfingerprint( std::ostream &os, neighbors* ns, FP* fps, int fpscnt );

void osfingerprintminimal( std::ostream &os, neighbors* ns, FP* fps, int fpscnt );


void osadjacencymatrix( std::ostream &os, const graphtype* g );

void osneighbors( std::ostream &os, const neighbors* ns );

void osedges( std::ostream &os, const graphtype* g );

void osgraphmorphisms( std::ostream &os, const graphtype* g1, const graphtype* g2, const std::vector<graphmorphism>* maps );

void osmachinereadablegraph(std::ostream &os, graphtype* g);

void zerograph(graphtype* g);

int pathsbetweencount( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2);

void pathsbetweentuples( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2, std::vector<std::vector<vertextype>>& out );

void pathbetweentuple( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2, std::vector<vertextype>& out );

void shortpathbetweentuple( const graphtype* g, const neighborstype* ns, const vertextype& v1, const vertextype& v2, std::vector<vertextype>& out );

int cyclesvcount( graphtype* g, neighborstype* ns, vertextype v1 );

void cyclesvset( graphtype* g, neighborstype* ns, vertextype v, std::vector<std::vector<vertextype>>& out );

graphtype* findedgesgivenvertexset( graphtype* g, std::vector<vertextype> vs);

int pathsbetweenmin( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2, int min);

int connectedsubsetcount(graphtype *g, neighborstype *ns, bool* vertices, const int breaksize);

graphtype* cyclegraph( const int dim );

void cyclesset( graphtype* g, neighborstype* ns, std::vector<std::vector<vertextype>>& out );

int cyclescount( graphtype* g, neighborstype* ns );

void copygraph( graphtype* g1, graphtype* g2 );

std::vector<std::vector<int>> getpermutations( const int i );

void verticesconnectedmatrix( bool* out, const graphtype* g, const neighborstype* ns  );

void verticesconnectedlist( const graphtype* g, const neighborstype* ns, vertextype* partitions, int* pindices  );

void CUDAverticesconnectedmatrix( bool* out, const graphtype* g, const neighborstype* ns );

void connectedpartition(graphtype *g, neighborstype *ns, std::vector<bool*>& outv);


#endif //GRAPHS_H
