//
// Created by peterglenn on 6/6/24.
//

#include "graphs.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>

int cmpwalk( neighbors ns, FP w1, FP w2 ) { // return -1 if w1 > w2; 0 if w1 == w2; 1 if w1 < w2
    int res = -1;
    if (w1.parent == nullptr) {
        if (w2.parent == nullptr) {
            res = 0;
        } else {
            return cmpwalk( ns, w1, *w2.parent);
        }
    }
    if (w2.parent == nullptr) {
        return cmpwalk( ns, *w1.parent, w2 );
    }

    if (res != 0)
        res = cmpwalk( ns, *w1.parent, *w2.parent);

    if (res == 0) {
        if (ns.degrees[w1.v] > ns.degrees[w2.v]) {
            return -1;
        } else {
            if (ns.degrees[w1.v] == ns.degrees[w2.v]) {
                return 0;
            } else {
                return 1;
            }
        }
    }
    return res;
}


int FPcmp( neighbors ns1, neighbors ns2, FP w1, FP w2 ) { // acts without consideration of self or parents; looks only downwards

    //std::cout << "FPcmp checking in: w1.v == " << w1.v << ", w2.v == " << w2.v << "\n";
/* take out part that checks parents
    if (ns.degrees[w1.v] < ns.degrees[w2.v])
        return 1;
    if (ns.degrees[w1.v] > ns.degrees[w2.v])
        return -1;
*/
    // ... and otherwise ...
    if (w1.nscnt == 3) {
        if ((ns1.degrees[w1.ns[0].v] == 1) && (ns1.degrees[w1.ns[1].v] == 2) && (ns1.degrees[w1.ns[2].v] == 3)) {
            if (w2.nscnt == 3) {
                if ((ns2.degrees[w2.ns[0].v] == 1) && (ns2.degrees[w2.ns[1].v] == 2) && (ns2.degrees[w2.ns[2].v] == 3)) {
                    std::cout << "Found\n";
                }
            }
        }
    }
    for (int n = 0; (n < w1.nscnt) && (n < w2.nscnt); ++n) {
        if (ns1.degrees[w1.ns[n].v] < ns2.degrees[w2.ns[n].v]) {
            return 1;
        } else {
            if (ns1.degrees[w1.ns[n].v] > ns2.degrees[w2.ns[n].v]) {
                return -1;
            }
        }
    }

    if (w1.nscnt < w2.nscnt) {
        return -1;
    } else {
        if (w1.nscnt > w2.nscnt) {
            return 1;
        }
    }

    int n = 0;
    int res = 0;
    while ((res == 0) && (n < w1.nscnt)) {
        res = FPcmp(ns1,ns2,w1.ns[n],w2.ns[n]);
        n++;
    }
    return res;
}

void sortneighbors( neighbors ns, FP* fps, int fpscnt ) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (int n = 0; n < (fpscnt-1); ++n) {
            if (FPcmp(ns,ns,fps[n],fps[n+1]) == -1) {
                //std::cout << "...reversing two... n == " << n << "\n";
                //std::cout << "Before swap: fps[n].v ==" << fps[n].v << ", fps[n+1].v == " << fps[n+1].v << "\n";
                FP tmpfp = fps[n];
                fps[n] = fps[n+1];
                fps[n+1] = tmpfp;
                changed = true;
                //std::cout << "After swap: fps[n].v ==" << fps[n].v << ", fps[n+1].v == " << fps[n+1].v << "\n";
            }
        }
    }
    //std::cout << "sorted.\n";

}


void takefingerprint( neighbors ns, FP* fps, int fpscnt ) {

    if (fpscnt == 0)
        return;

    FP sorted[ns.g.dim];
    int idx = 0;
    int sizeofdegree[ns.maxdegree+1];

    //std::cout << "fpscnt == " << fpscnt << ", ns.maxdegree == " << ns.maxdegree << "\n";

    for (int d = 0; d <= ns.maxdegree; ++d) {
        sizeofdegree[d] = 0;
        //std::cout << "d == " << d << " \n";
        for (int vidx = 0; vidx < fpscnt; ++vidx) {
            int deg = ns.degrees[fps[vidx].v];
            if (deg == d) {
                ++sizeofdegree[d];
                FP* parent = fps[vidx].parent;
                if (parent != nullptr)
                    deg--;
                bool dupe = false;
                while (!dupe && (parent != nullptr)) {
                    //std::cout << "Checking dupes: fps[vidx].v == " << fps[vidx].v << ", parent->v == " << parent->v << "\n";
                    dupe = dupe || (parent->v == fps[vidx].v);
                    parent = parent->parent;
                }
                //std::cout << "vidx == " << vidx << ", dupe == " << dupe << "\n";
                if (dupe) {
                    fps[vidx].ns = nullptr;
                    fps[vidx].nscnt = 0;
                } else {
                    int tmpn = 0;
                    vertextype parentv;
                    if (fps[vidx].parent != nullptr) {
                        parentv = fps[vidx].parent->v;
                    } else {
                        parentv = -1;
                    }
                    if (deg > 0) {
                        fps[vidx].nscnt = deg;
                        fps[vidx].ns = (FP*)malloc(deg*sizeof(FP));
                        for (int n = 0; n < ns.degrees[fps[vidx].v]; ++n) {
                            vertextype nv = ns.neighborslist[ns.g.dim*fps[vidx].v + n];
                            if (nv != parentv) {
                                fps[vidx].ns[tmpn].parent = &(fps[vidx]);
                                fps[vidx].ns[tmpn].v = nv;
                                fps[vidx].ns[tmpn].ns = nullptr;
                                fps[vidx].ns[tmpn].nscnt = 0;
                                ++tmpn;
                            }
                        }
                        takefingerprint( ns, fps[vidx].ns, tmpn );
                    } else {
                        fps[vidx].ns = nullptr;
                        fps[vidx].nscnt = 0;
                    }
                }
                //++sizeofdegree[d];
                sorted[idx] = fps[vidx];
                ++idx;
            }
        }
    }

    int startidx = 0;
    for (int i = 0; i <= ns.maxdegree; ++i) {
        FP fps2[sizeofdegree[i]];
        //std::cout << "i, sizeofdegree[i] == " << i << ", " << sizeofdegree[i] << "\n";
        for (int j = 0; j < sizeofdegree[i]; ++j) {
            fps2[j] = sorted[startidx + j];
        }
        sortneighbors( ns, fps2, sizeofdegree[i] );
        for (int j = 0; j < sizeofdegree[i]; ++j) {
            sorted[startidx+j] = fps2[j];
        }
        startidx = startidx + sizeofdegree[i];
    }
    for (int i=0; i < idx; ++i) {  // idx == fpscnt
        fps[i] = sorted[i];
    }
}

void freefps( FP* fps, int fpscnt ) {
    for( int n = 0; n < fpscnt; ++n ) {
        freefps( fps[n].ns, fps[n].nscnt );
        free( fps[n].ns );
    }
}

neighbors computeneighborslist( graph g ) {
    neighbors ns;
    ns.g = g;
    ns.maxdegree = -1;
    ns.neighborslist = (vertextype*)malloc(g.dim * (g.dim) * sizeof(vertextype));
    ns.degrees = (int*)malloc(g.dim * sizeof(int) );
    for (vertextype n = 0; n < g.dim; ++n) {
        ns.degrees[n] = 0;
        for (vertextype i = 0; i < g.dim; ++i) {
            if (g.adjacencymatrix[n*g.dim + i]) {
                ns.neighborslist[n * g.dim + ns.degrees[n]] = i;
                ns.degrees[n]++;
            }
        }
        ns.maxdegree = (ns.degrees[n] > ns.maxdegree ? ns.degrees[n] : ns.maxdegree );
    }
    return ns;
}

void sortneighborslist( neighbors* nsptr ) {

    int idx = 0;
    int sorted[nsptr->g.dim];
    for (int k = 0; k < nsptr->maxdegree; ++k ) {
        for (int n = 0; n < nsptr->g.dim; ++n) {
            if (nsptr->degrees[n]==k) {
                sorted[n] = idx;
                idx++;
            }
        }
    }

    for (int n = 0; n < nsptr->g.dim; ++n) {
        vertextype ns[nsptr->g.dim];
        for (int l = 0; l < nsptr->g.dim; ++l)
            ns[l] = -1;
        for (int k = 0; k < nsptr->degrees[n]; ++k) {
            ns[sorted[nsptr->neighborslist[nsptr->g.dim*n + k]]] = nsptr->neighborslist[nsptr->g.dim*n + k];
        }
        int idx = 0;
        for (int l = 0; l < nsptr->g.dim; ++l) {
            if (ns[l] >= 0) {
                nsptr->neighborslist[nsptr->g.dim*n + idx] = ns[l];
                idx++;
            }
        }
    }

}

int seqtoindex( vertextype* seq, const int idx, const int sz ) {
    if (idx == sz)
        return 0;
    return seq[idx]*sz + seqtoindex( seq, idx+1, sz);
}

/*
void sortvertices( graph g ) {
    neighbors ns;
    ns = computeneighborslist(g);
    sortneighborslist(ns);
    int* fp;
    int fpsz = std::pow(ns.maxdegree,ns.maxdegree);

    fp = (int*)malloc(fpsz*sizeof(int));
    takegraphfingerprint( ns, fp );

    // ...

    free(ns.degrees);
    free(ns.neighborslist);
}

*/

bool isiso( graph g1, graph g2, vertextype* map ) {
    bool match = true;
    if (g1.dim != g2.dim) {
        return false;
    }
    for (vertextype n = 0; n < g1.dim && match; ++n ) {
        for (vertextype i = 0; i < (g1.dim - n) && match; ++i ) {
            match = match && g1.adjacencymatrix[n*g1.dim + i] == g2.adjacencymatrix[map[n]*g2.dim + map[i]];
        }
    }
    return match;
}

bool areisomorphic( graph g1, graph g2 ) {

    if (g1.dim != g2.dim)
        return false;
    neighbors ns1 = computeneighborslist(g1);
    neighbors ns2 = computeneighborslist(g2);

    FP fps1[g1.dim];
    FP fps2[g2.dim];
    for (vertextype n = 0; n < g1.dim; ++n) {
        fps1[n].v = n;
        fps1[n].ns = nullptr;
        fps1[n].nscnt = 0;
        fps1[n].parent = nullptr;
        fps2[n].v = n;
        fps2[n].ns = nullptr;
        fps2[n].nscnt = 0;
        fps2[n].parent = nullptr;
    }

    takefingerprint(ns1,fps1,g1.dim);
    takefingerprint(ns2,fps2, g2.dim);
    //...
    return FPcmp(ns1,ns2,*fps1,*fps2) != 0;
}

void osfingerprintrecurse( std::ostream &os, neighbors ns, FP* fps, int fpscnt, int depth ) {
    for (int i = 0; i < depth; ++i) {
        os << "   ";
    }
    os << "fpscnt == " << fpscnt << ": \n";
    for (int n = 0; n < fpscnt; ++n) {
        for (int i = 0; i < depth+1; ++i) {
            os << "   ";
        }
        os << "{v, deg v} == {" << fps[n].v << ", " << ns.degrees[fps[n].v] << "}\n";
        osfingerprintrecurse( os, ns, fps[n].ns, fps[n].nscnt,depth+1 );
        if (fps[n].nscnt == 0) {
            for (int i = 0; i < depth+1; ++i) {
                os << "   ";
            }
            FP* parent = &(fps[n]);
            std::cout << "Total walk by degrees: <";
            while (parent != nullptr) {
                std::cout << ns.degrees[parent->v] << ", ";
                parent = parent->parent;
            }
            std::cout << "\b\b>, ";
            parent = &(fps[n]);
            std::cout << " and by vertices: <";
            while (parent != nullptr) {
                std::cout << parent->v << ", ";
                parent = parent->parent;
            }
            std::cout << "\b\b>\n";
        }
    }
    //for (int i = 0; i < depth; ++i) {
    //    os << "   ";
    //}
    //os << "end fpscnt == " << fpscnt << "\n";
}

void osfingerprint( std::ostream &os, neighbors ns, FP* fps, int fpscnt ) {
    osfingerprintrecurse( os, ns, fps, fpscnt, 0 );
}

void osadjacencymatrix( std::ostream &os, graph g ) {
    for (int n = 0; n < g.dim; ++n ) {
        for (int i = 0; i < g.dim; ++i) {
            os << g.adjacencymatrix[n*g.dim + i] << " ";
        }
        os << "\n";
    }
}

void osneighbors( std::ostream &os, neighbors ns ) {
    for (int n = 0; n < ns.g.dim; ++n ) {
        os << "ns.degrees["<<n<<"] == "<<ns.degrees[n]<<": ";
        for (int i = 0; i < ns.degrees[n]; ++i) {
            std::cout << ns.neighborslist[n*ns.g.dim + i] << ", " ;
        }
        std::cout << "\b\b\n";
    }
}


