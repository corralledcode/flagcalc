//
// Created by peterglenn on 6/6/24.
//

#include "graphs.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ranges>
#include <vector>

#define MAXFACTORIAL 0

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

    if (w1.nscnt > w2.nscnt) {
        return -1;
    } else {
        if (w1.nscnt < w2.nscnt) {
            return 1;
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

    // the following loop can be avoided it seems if we replace "parent" the pointer with a list of vertices already visited
    for (int i=0; i < idx; ++i) {  // idx == fpscnt
        fps[i] = sorted[i];
        for(int j = 0; j < fps[i].nscnt; ++j) {
            fps[i].ns[j].parent = &fps[i];
        }
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

/*
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
*/

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

bool ispartialisoonlynewest( graph g1, graph g2, graphmorphism map) {
    bool match = true;
    int i = map.size()-1;
    for (int n = 0; match && (n < map.size()-1); ++n) {
        match = match && (g1.adjacencymatrix[map[n].first*g1.dim + map[i].first] == g2.adjacencymatrix[map[n].second*g2.dim + map[i].second]);
    }
    return match;
}


bool ispartialiso( graph g1, graph g2, graphmorphism map) {
    bool match = true;
    for (int n = 0; match && (n < map.size()-1); ++n) {
        for (int i = n+1; match && (i < map.size()); ++i) {
            match = match && (g1.adjacencymatrix[map[n].first*g1.dim + map[i].first] == g2.adjacencymatrix[map[n].second*g2.dim + map[i].second]);
        }
    }
    return match;
}

bool isiso( graph g1, graph g2, graphmorphism map ) {
    bool match = true;
    if (g1.dim != g2.dim) {
        return false;
    }
    for (vertextype n = 0; (n < g1.dim-1) && match; ++n ) {
        for (vertextype i = n+1; (i < g1.dim) && match; ++i ) {
            match = match && (g1.adjacencymatrix[map[n].first*g1.dim + map[i].first] == g2.adjacencymatrix[map[n].second*g2.dim + map[i].second]);
        }
    }
    return match;
}

std::vector<std::vector<int>> getpermutations( const int i ) {
    std::vector<std::vector<int>> res {};
    std::vector<int> base {0};
    res.push_back(base);
    //std::cout << "getpermutations " << i << "\n";
    if (i == 1)
        return res;
    std::vector<std::vector<int>> tmp = getpermutations(i-1);

    res.clear();
    for (int j = 0; j < i; ++j) {
        for (int n = 0; n < tmp.size(); ++n) {
            std::vector<int> tmpperm {};
            //std::cout << "tmp[n].size()==" << tmp[n].size() << ", i == " << i << "\n";

            for (int k = 0; k < j; ++k) {
                tmpperm.push_back(tmp[n][k]+1);
            }
            tmpperm.push_back(0);
            for (int k = j+1; k < i; ++k) {
                tmpperm.push_back(tmp[n][k-1]+1);
            }
            res.push_back(tmpperm);
        }
    }
    //std::cout << "i, res.size() == " << i << ", " << res.size() << "\n";
    //for (int j = 0; j < res.size(); ++j) {
    //    std::cout << "j, res[j].size() == " << j << ", " << res[j].size() << "\n";
    //}
    return res;
}


// fastgetpermuations: it turns out to be slower, at least on the graphs tested,
// so for now this feature is not used (what it does is to check each possible permutation
// for being a partial iso; the difficulty is that say a graph of four disjoiht
// isomorphic subgraph, each of size n, has (n!)^4*4!;
// we have huge jumps in runtime by adding one more isomorphic disjoint subgraph
// and it may be that with some work fastgetpermutations speeds such a graph's processing
// time up.

/*
fastgetpermutations pseudocope

in: cnt, index, targetset,
out: add one map per pair {index,t}, t from targetset

*/

bool fastgetpermutations( std::vector<vertextype> targetset, graph g1, graph g2, FP* fps1, FP* fps2, int idx, graphmorphism partialmap, std::vector<graphmorphism>* results) {

    // implicit variable "cnt" == targetset.size

    std::vector<graphmorphism> res {};
    if (targetset.size() <= 1) {
        partialmap.push_back( {fps1[idx].v,fps2[targetset[0]].v});
        results->push_back(partialmap);
        if (ispartialisoonlynewest(g1,g2,partialmap)) {
            return true;
        } else {
            results->resize(results->size()-1); // can't seem to get to work erase(partialmap->size()-1);
            return false;
        }
    }
    graphmorphism tmppartial {};
    for (int n = 0; n < targetset.size(); ++n) {
        tmppartial = partialmap;
        tmppartial.push_back( {fps1[idx].v, fps2[targetset[n]].v});
        std::vector<vertextype> newtargetset {};
        for (int i = 0; i < targetset.size(); ++i) { // this for loop because erase doesn't seem to work
            if (i != n)
                newtargetset.push_back(targetset[i]);
        }
        if (ispartialisoonlynewest(g1,g2,tmppartial)) {
            if (fastgetpermutations(newtargetset,g1,g2,fps1,fps2,idx+1,tmppartial,results)) {
                results->push_back(tmppartial);
            }
        }
    }
    return true;
}


std::vector<graphmorphism> enumisomorphisms( neighbors ns1, neighbors ns2 ) {
    graph g1 = ns1.g;
    graph g2 = ns2.g;
    std::vector<graphmorphism> maps {};
    if (g1.dim != g2.dim || ns1.maxdegree != ns2.maxdegree)
        return maps;
    //neighbors ns1 = computeneighborslist(g1);
    //neighbors ns2 = computeneighborslist(g2);

    // to do: save time in most cases by checking that ns1 matches ns2

    FP fps1[g1.dim];
    for (vertextype n = 0; n < g1.dim; ++n) {
        fps1[n].v = n;
        fps1[n].ns = nullptr;
        fps1[n].nscnt = 0;
        fps1[n].parent = nullptr;
    }

    takefingerprint(ns1,fps1,g1.dim);

    //osfingerprint(std::cout,ns1,fps1,g1.dim);

    FP fps2[g2.dim];
    for (vertextype n = 0; n < g2.dim; ++n) {
        fps2[n].v = n;
        fps2[n].ns = nullptr;
        fps2[n].nscnt = 0;
        fps2[n].parent = nullptr;
    }

    takefingerprint(ns2,fps2,g2.dim);

    //osfingerprint(std::cout,ns2,fps2,g2.dim);

    vertextype del[ns1.maxdegree+2];
    int delcnt = 0;
    del[delcnt] = 0;
    ++delcnt;
    for( vertextype n = 0; n < g1.dim-1; ++n ) {
        if (ns1.degrees[fps1[n].v] != ns2.degrees[fps2[n].v])
            return maps; // return empty set of maps
        int res1 = FPcmp(ns1,ns1,fps1[n],fps1[n+1]);
        int res2 = FPcmp(ns2,ns2,fps2[n],fps2[n+1]);
        if (res1 != res2)
            return maps;  // return empty set of maps
        if (res1 < 0) {
            std::cout << "Error: not sorted (n == " << n << ", res1 == "<<res1<<")\n";
            return maps;
        }
        if (res1 > 0) {
            del[delcnt] = n+1;
            ++delcnt;
            //std::cout << "inc'ed delcnt\n";
        }
    }
    del[delcnt] = g1.dim;

    //std::pair<vertextype,vertextype> basepair;
    //basepair = {-1,-1};
    std::vector<std::vector<std::vector<int>>> perms {};
    graphmorphism basemap {};
    maps.push_back(basemap);
    for (int l = 0; l<delcnt; ++l) {
        std::vector<graphmorphism> newmaps {};
        //std::cout << "maps.size == " << maps.size() << "\n";

        int permsidx = del[l+1]-del[l];

        if (permsidx < MAXFACTORIAL) {
            if (permsidx > perms.size())
                perms.resize(permsidx+1);
            if (perms[permsidx].size() == 0)
                perms[permsidx] = getpermutations(permsidx);

            for (int k = 0; k < maps.size(); ++k) {
                //std::cout << "del[l] == " << del[l] << ", del[l+1] == " << del[l+1] << "delcnt == " << delcnt << "\n";
                //std::vector<std::vector<int>> perm = fastgetpermutations(del[l+1]-del[l],g1,g2,fps1,fps2,del[l]);
                std::vector<std::vector<int>> perm = perms[permsidx];
                for (int i = 0; i < perm.size(); ++i) {
                    graphmorphism newmap = maps[k];
                    for (int j = 0; j < perm[i].size(); ++j) {
                        std::pair<vertextype,vertextype> newpair;
                        //std::cout << "i,j, perm[i][j] == "<<i<<", "<<j<<", "<< perm[i][j]<<"\n";
                        newpair = {fps1[del[l]+j].v,fps2[del[l]+perm[i][j]].v};
                        newmap.push_back(newpair);
                    }
                    newmaps.push_back(newmap);
                }
            }
            maps.clear();
            for (int i = 0; i < newmaps.size(); ++i ) {
                if (ispartialiso(g1,g2,newmaps[i])) {
                    maps.push_back(newmaps[i]);
                }
            }
        } else { // the case when permsidx >= MAXFACTORIAL
            for (int k = 0; k < maps.size(); ++k) {
                std::vector<vertextype> targetset {};
                for (int j = 0; j < permsidx; ++j) {
                    targetset.push_back(del[l]+j);
                    //std::cout << targetset[targetset.size()-1] << "\n";;
                }
                if (permsidx > 0) {
                    fastgetpermutations(targetset,g1,g2,fps1,fps2,del[l],maps[k],&(newmaps));

                }
            }
            maps.clear();
            for (int i = 0; i < newmaps.size(); ++i) {
                if (newmaps[i].size() == del[l] + permsidx)
                    maps.push_back(newmaps[i]);
            }
        }
    }
    /*std::vector<graphmorphism> res {};
    for (int i = 0; i < maps.size(); ++i ) {
        if (isiso(g1,g2,maps[i])) {
            res.push_back(maps[i]);
        }
    }*/

    return maps;
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
            FP* parent = fps[n].parent;
            std::cout << "Total walk by degrees: <" << ns.degrees[fps[n].v] << ", ";
            while (parent != nullptr) {
                std::cout << ns.degrees[parent->v] << ", ";
                parent = ((*parent).parent);
            }
            std::cout << "\b\b>, ";
            parent = fps[n].parent;
            std::cout << " and by vertices: <" << fps[n].v << ", ";
            while (parent != nullptr) {
                std::cout << parent->v << ", ";
                parent = (*parent).parent;
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

void osgraphmorphisms( std::ostream &os, std::vector<graphmorphism> maps ) {
    for (int n = 0; n < maps.size(); ++n) {
        os << "Map number " << n+1 << " of " << maps.size() << ":\n";
        for (int i = 0; i < maps[n].size(); ++i) {
            os << maps[n][i].first << " maps to " << maps[n][i].second << "\n";
        }
    }
}

