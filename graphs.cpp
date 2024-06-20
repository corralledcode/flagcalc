//
// Created by peterglenn on 6/6/24.
//

// Use caution: one out of three must be asserted to be #defined:
//define THREADPOOL1
//#define THREADED1
#define NOTTHREADED1

// Use caution: one out of three must be asserted to be #defined:
//#define THREADPOOL2
//#define THREADED2
#define NOTTHREADED2

#ifdef THREADED1
#define THREADED
#endif
#ifdef THREADED2
#define THREADED
#endif
#ifdef THREADPOOL1
#define THREADPOOL
#endif
#ifdef THREADPOOL2
#define THREADPOOL
#endif
#ifdef NOTTHREADED1
#define NOTTHREADED
#endif
#ifdef NOTTHREADED2
#define NOTTHREADED
#endif

// Choose one of the two following; they affect the FPs sorting in the code below "sortneighbors"
//#define QUICKSORT2
#define NAIVESORT2


#include "graphs.h"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <future>
#include <iostream>
#include <ranges>
#include <vector>
#ifdef THREADPOOL
#include "thread_pool.cpp"
#endif
#ifdef THREADED
#include <thread>
#endif

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


int FPcmp( neighbors* ns1, neighbors* ns2, FP* w1, FP* w2 ) { // acts without consideration of self or parents; looks only downwards

    //std::cout << "FPcmp checking in: w1.v == " << w1.v << ", w2.v == " << w2.v << "\n";
/* take out part that checks parents
    if (ns.degrees[w1.v] < ns.degrees[w2.v])
        return 1;
    if (ns.degrees[w1.v] > ns.degrees[w2.v])
        return -1;
*/
    // ... and otherwise ...
/*
    int tmpnscnt1;
    int tmpnscnt2;

    if (w1->invert)
        tmpnscnt1 = ns1->g->dim - w1->nscnt - 1;
    else
        tmpnscnt1 = w1->nscnt;
    if (w2->invert)
        tmpnscnt2 = ns2->g->dim - w2->nscnt - 1;
    else
        tmpnscnt2 = w2->nscnt;

    if (tmpnscnt1 > tmpnscnt2) {
        return -1;
    } else {
        if (tmpnscnt1 < tmpnscnt2) {
            return 1;
        }
    }


    if (!w1->invert)
        if (ns1->degrees[w1->v] < ns2->degrees[w2->v])
            return 1;
        else
            if (ns1->degrees[w1->v] > ns2->degrees[w2->v])
                return -1;

    if (w1->invert)
        if (ns1->degrees[w1->v] < ns2->degrees[w2->v])
            return 1;
        else
            if (ns1->degrees[w1->v] > ns2->degrees[w2->v])
                return -1;

    int n = 0;
    int res = 0;
    while ((res == 0) && (n < w1->nscnt)) {
        res = FPcmp(ns1,ns2,&w1->ns[n],&w2->ns[n]);
        n++;
    }
    return res;
*/
    /*

    for (int n = 0; (n < w1->nscnt) && (n < w2->nscnt); ++n) {
        if (w1->ns[n].invert != w2->ns[n].invert) {
            return (w1->ns[n].invert ? -1 : 1);
        }
        if (ns1->degrees[w1->ns[n].v] < ns2->degrees[w2->ns[n].v]) {
            return 1;
        } else {
            if (ns1->degrees[w1->ns[n].v] > ns2->degrees[w2->ns[n].v]) {
                return -1;
            }
        }
    }


    int n = 0;
    int res = 0;
    while ((res == 0) && (n < w1->nscnt)) {
        res = FPcmp(ns1,ns2,&w1->ns[n],&w2->ns[n]);
        n++;
    }
    return res;
*/


/*    if (w1->invert != w2->invert) {
        return (w1->invert ? -1 : 1);
    }
*/

    if (ns1->g->dim != ns2->g->dim)
        return ns1->g->dim < ns2->g->dim ? 1 : -1;

    /*
    if (w1->nscnt == 0)
        if (w2->nscnt > 0)
            return (w1->invert ? -1 : 1);
    if (w2->nscnt == 0)
        if (w1->nscnt > 0)
            return (w2->invert ? 1 : -1);
*/
    int multiplier = (w1->invert ? -1 : 1);
    //multiplier = 1;
    if (w1->invert != w2->invert)
        return multiplier;

    /*
    if (ns1->degrees[w1->v] > ns2->degrees[w2->v])
        return -1;
    if (ns1->degrees[w1->v] < ns2->degrees[w2->v])
        return 1;
*/
    if (w1->nscnt > w2->nscnt) {
        return (-1)*multiplier;
    } else {
        if (w1->nscnt < w2->nscnt) {
            return multiplier;
        }
    }


    for (int n = 0; (n < w1->nscnt) && (n < w2->nscnt); ++n) {
        if (w1->ns[n].invert != w2->ns[n].invert) {
            return (w1->ns[n].invert ? -1 : 1);
        }
        if (ns1->degrees[w1->ns[n].v] < ns2->degrees[w2->ns[n].v]) {
            return 1;
        } else {
            if (ns1->degrees[w1->ns[n].v] > ns2->degrees[w2->ns[n].v]) {
                return (-1);
            }
        }
    }

    int n = 0;
    int res = 0;
    while ((res == 0) && (n < w1->nscnt)) {
        res = FPcmp(ns1,ns2,&w1->ns[n],&w2->ns[n]);
        n++;
    }
    return res * multiplier;
}

inline int partition2( std::vector<int> &arr, int start, int end, neighbors* ns, FP* fpslist ) {
    int pivot = arr[start];
    int count = 0;
    for (int i = start+1;i <= end; i++) {
        if (FPcmp(ns,ns,&fpslist[arr[i]],&fpslist[pivot]) >= 0) {
            count++;
        }
    }

    int pivotIndex = start + count;
    std::swap(arr[pivotIndex],arr[start]);

    int i = start;
    int j = end;
    while (i < pivotIndex && j > pivotIndex) {
        while (FPcmp(ns,ns,&fpslist[arr[i]],&fpslist[pivot]) >= 0) {
            i++;
        }
        while (FPcmp(ns,ns,&fpslist[arr[j]],&fpslist[pivot]) < 0) {
            j--;
        }
        if (i < pivotIndex && j > pivotIndex) {
            std::swap(arr[i++],arr[j--]);
        }
    }
    return pivotIndex;
}

inline void quickSort2( std::vector<int> &arr, int start, int end,neighbors* ns, FP* fpslist ) {

    if (start >= end)
        return;

    int p = partition2(arr,start,end,ns,fpslist);

    quickSort2(arr, start, p-1,ns,fpslist);
    quickSort2(arr, p+1, end,ns,fpslist);
}


void sortneighbors( neighbors* ns, FP* fps, int fpscnt ) {


#ifdef NAIVESORT2
    bool changed = true;
    while (changed) {
        changed = false;
        for (int n = 0; n < (fpscnt-1); ++n) {
            if (FPcmp(ns,ns,&fps[n],&fps[n+1]) == -1) {
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
#endif

#ifdef QUICKSORT2

    std::vector<int> sorted;
    sorted.resize(fpscnt);
    for (int i = 0; i < fpscnt; ++i) {
        sorted[i] = i;
    }
    quickSort2( sorted,0,fpscnt-1, ns, fps);

    std::vector<FP> tmpfp {};
    tmpfp.resize(fpscnt);
    for (int i = 0; i < fpscnt; ++i) {
        tmpfp[i] = fps[sorted[i]];
    }
    for (int i = 0; i < fpscnt; ++i) {
        fps[i] = tmpfp[i];
    }


#endif

}


void takefingerprint( neighbors* ns, FP* fps, int fpscnt ) {

    if (fpscnt == 0 || ns == nullptr) {

        return;
    }

    const int dim = ns->g->dim;
    FP sorted[dim];
    FP sortedinverted[dim];
    int halfdegree = (dim+1)/2;
    bool invert = false;
    int idx = 0;
    int invertedidx = 0;

    //std::cout << "fpscnt == " << fpscnt << ", ns.maxdegree == " << ns.maxdegree << "\n";
    int md = ns->maxdegree;
    if (ns->maxdegree < halfdegree)
        md = dim - ns->maxdegree;
    //md = dim-1;
    int sizeofdegree[md+1];
    int sizeofinverteddegree[md+1];
    for (int d = 0; d <= md; ++d) {
        sizeofdegree[d] = 0;
        sizeofinverteddegree[d] = 0;
        //std::cout << "d == " << d << " \n";
        for (int vidx = 0; vidx < fpscnt; ++vidx) {
            invert = (ns->degrees[fps[vidx].v] >= halfdegree);
            fps[vidx].invert = invert;
            int deg = ( invert ? dim - ns->degrees[fps[vidx].v] -1  : ns->degrees[fps[vidx].v] );
            if (deg == d) {

                FP* parent = fps[vidx].parent;
                if (parent != nullptr)
                    if (parent->invert == invert) {
                        deg--;
                        //std::cout << "Hello world\n";
                    }
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
                    fps[vidx].invert = false;
                } else {
                    int tmpn = 0;
                    int parentv;
                    if (fps[vidx].parent != nullptr) {
                        parentv = fps[vidx].parent->v;
                    } else {
                        parentv = -1;
                    }
                    if ((deg > 0) && !invert) {
                        fps[vidx].nscnt = deg;
                        //std::cout << "deg " << deg << ", vidx = " << vidx << "\n";
                        fps[vidx].ns = (FP*)malloc((deg)*sizeof(FP));
                        for (int n = 0; n < ns->degrees[fps[vidx].v]; ++n) {
                            vertextype nv = ns->neighborslist[dim*fps[vidx].v + n];
                            if (nv != parentv) {
                                fps[vidx].ns[tmpn].parent = &(fps[vidx]);
                                fps[vidx].ns[tmpn].v = nv;
                                fps[vidx].ns[tmpn].ns = nullptr;
                                fps[vidx].ns[tmpn].nscnt = 0;
                                fps[vidx].ns[tmpn].invert = ns->degrees[fps[vidx].v] >= halfdegree;
                                ++tmpn;
                            }
                        }
                        takefingerprint( ns, fps[vidx].ns, tmpn );
                    } else {
                        if ((deg > 0) && invert) {
                            fps[vidx].nscnt = deg;
                            //std::cout << "deg " << deg << ", vidx = " << vidx << ", v = " << fps[vidx].v << " invert";
                            //std::cout << "degree " << ns->degrees[fps[vidx].v] << "\n";
                            fps[vidx].ns = (FP*)malloc(deg * sizeof(FP));
                            for (int n = 0; n < (dim - ns->degrees[fps[vidx].v] - 1); ++n) {
                                vertextype nv = ns->nonneighborslist[dim*fps[vidx].v + n];
                                //std::cout << "nv = " << nv << "\n";
                                if (nv != parentv) {
                                    fps[vidx].ns[tmpn].parent = &(fps[vidx]);
                                    fps[vidx].ns[tmpn].v = nv;
                                    fps[vidx].ns[tmpn].ns = nullptr;
                                    fps[vidx].ns[tmpn].nscnt = 0;
                                    fps[vidx].ns[tmpn].invert = ns->degrees[fps[vidx].v] >= halfdegree;
                                    ++tmpn;  // tmp keeps a separate count from n in order to account for the omitted parentv
                                }
                            }
                            takefingerprint( ns, fps[vidx].ns, tmpn  );
                        } else {
                            fps[vidx].ns = nullptr;
                            fps[vidx].nscnt = 0;
                            fps[vidx].invert = ns->degrees[fps[vidx].v] >= halfdegree;
                        }
                    }
                }
                //++sizeofdegree[d];
                if (!fps[vidx].invert) {
                    sorted[idx] = fps[vidx];
                    ++idx;
                    ++sizeofdegree[d];
                } else {
                    std::cout << "fps[vidx].invert == " << fps[vidx].invert << "\n";
                    sortedinverted[invertedidx] = fps[vidx];
                    ++invertedidx;
                    ++sizeofinverteddegree[d];
                }
            }
        }
    }

    int startidx = 0;
    for (int i = 0; i <= md; ++i) {
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
    int startidx2 = 0;
    for (int i = 0; i<=md; ++i) {
        FP fps2[sizeofinverteddegree[i]];
        //std::cout << "i, sizeofinverteddegree[i] == " << i << ", " << sizeofinverteddegree[i] << "\n";
        for (int k = 0; k < sizeofinverteddegree[i]; ++k) {
            fps2[k] = sortedinverted[(sizeofinverteddegree[i] - k-1) + startidx2 ];
        }
        sortneighbors( ns, fps2, sizeofinverteddegree[i] );
        for (int j = 0; j < sizeofinverteddegree[i]; ++j) {
            sortedinverted[startidx2+j] = fps2[j];
        }
        startidx2 = startidx2 + sizeofinverteddegree[i];
    }


    // the following loop can be avoided it seems if we replace "parent" the pointer with a list of vertices already visited
    //std::cout << idx << " " << fpscnt << "\n";
    for (int i=0; i < idx; ++i) {  // idx == fpscnt -- nooe
        fps[i] = sorted[i];
        for(int j = 0; j < fps[i].nscnt; ++j) {
            fps[i].ns[j].parent = &fps[i];
        }
    }
    for (int i = 0; i < invertedidx; ++i) {
        fps[i+idx] = sortedinverted[invertedidx - i - 1];
        for(int j = 0; j < fps[i+idx].nscnt; ++j) {
            fps[i+idx].ns[j].parent = &fps[i+idx];
        }

    }
}

void freefps( FP* fps, int fpscnt ) {
    for( int n = 0; n < fpscnt; ++n ) {
        freefps( fps[n].ns, fps[n].nscnt );
        delete fps[n].ns;
    }
}


// superceded by neighbors class in graphs.h

/*
neighbors<vltype> computeneighborslist( graphtype g ) {
    neighbors<vltype> ns;
    ns.g = g;
    ns.maxdegree = -1;
    ns.neighborslist = (vertextype*)malloc(g.dim * (g.dim) * sizeof(vertextype));
    ns.nonneighborslist = (vertextype*)malloc(g.dim * (g.dim) * sizeof(vertextype));
    ns.degrees = (int*)malloc(g.dim * sizeof(int) );
    int nondegrees[g.dim];
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
    for (vertextype n = 0; n < g.dim; ++n) {
        nondegrees[n] = 0;
        for (vertextype i = 0; i < g.dim; ++i) {
            if (!g.adjacencymatrix[n*g.dim + i]) {
                ns.nonneighborslist[n*g.dim + nondegrees[n]] = i;
                nondegrees[n]++;
            }
        }
    }
    for (vertextype n = 0; n < g.dim; ++n) {
        if (ns.degrees[n] + nondegrees[n] != g.dim)
            std::cout << "BUG in computeneighborslist\n";
    }

    return ns;
}
*/

/*
int seqtoindex( vertextype* seq, const int idx, const int sz ) {
    if (idx == sz)
        return 0;
    return seq[idx]*sz + seqtoindex( seq, idx+1, sz);
}
*/

bool ispartialisoonlynewest( graphtype g1, graphtype g2, graphmorphism map) {
    bool match = true;
    int i = map.size()-1;
    for (int n = 0; match && (n < map.size()-1); ++n) {
        match = match && (g1.adjacencymatrix[map[n].first*g1.dim + map[i].first] == g2.adjacencymatrix[map[n].second*g2.dim + map[i].second]);
    }
    return match;
}


bool ispartialiso( graphtype g1, graphtype g2, graphmorphism map) {
    bool match = true;
    for (int n = 0; match && (n < map.size()-1); ++n) {
        for (int i = n+1; match && (i < map.size()); ++i) {
            match = match && (g1.adjacencymatrix[map[n].first*g1.dim + map[i].first] == g2.adjacencymatrix[map[n].second*g2.dim + map[i].second]);
        }
    }
    return match;
}

bool isiso( graphtype g1, graphtype g2, graphmorphism map ) {
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


/*
void fastgetpermutationscoretmp(int n, const std::vector<vertextype> targetset, const graphtype g1, const FP* fps1,graphmorphism tmppartial, std::vector<graphmorphism>* result) {
    std::cout << n << "\n";
    if (targetset.size()>0){
        std::cout << targetset[0] << "\n";
        graphmorphism newmorph;
        newmorph.push_back (std::pair(targetset[0],targetset[0]));
        result->push_back (newmorph);
    }
    return;
}
*/


bool fastgetpermutationscore( const std::vector<vertextype> targetset, const graphtype g1, const graphtype g2, const FP* fps1, const FP* fps2, const int idx, graphmorphism partialmap, std::vector<graphmorphism>* resptr) {
/*
           if (targetset.size()>0) {
                partialmap.push_back({fps1[idx].v,fps2[targetset[0]].v});
                resptr->push_back(partialmap);
                return true;
            }
    */

    // implicit variable "cnt" == targetset.size

    std::vector<graphmorphism> res {};
    if (targetset.size() <= 1) {
        partialmap.push_back( {fps1[idx].v,fps2[targetset[0]].v});
        resptr->push_back(partialmap);
        if (ispartialisoonlynewest(g1,g2,partialmap)) {
            return true;
        } else {
            resptr->resize(resptr->size()-1); // can't seem to get to work erase(partialmap->size()-1);
            return false;
        }
    }
    graphmorphism tmppartial {};
    //std::vector<std::thread> t {};
    //t.resize(targetset.size());
    std::vector<std::vector<graphmorphism>> tmpres {};
    tmpres.resize(targetset.size());
    for (int n = 0; n < targetset.size(); ++n) {
        tmppartial = partialmap;
        tmppartial.push_back( {fps1[idx].v, fps2[targetset[n]].v});

        //t[n] = std::thread(fastgetpermutationscore,n, targetset, g1,g2,&(*fps1),&(*fps2),idx,tmppartial, &(tmpres[n]));
        //t[n] = std::thread(fastgetpermutationscoretmp,n, targetset, g1, &(*fps1), tmppartial, results); //,targetset,g1,g2,fps1,fps2,idx, tmppartial, &(results[n]));
        std::vector<vertextype> newtargetset {};
        for (int i = 0; i < targetset.size(); ++i) { // this for loop because erase doesn't seem to work
            if (i != n)
                newtargetset.push_back(targetset[i]);
        }
        if (ispartialisoonlynewest(g1,g2,tmppartial)) {
            if (fastgetpermutationscore(newtargetset,g1,g2,fps1,fps2,idx+1,tmppartial,resptr)) {
                resptr->push_back(tmppartial);
            }
        }

    }

    /*
        for (int n = 0; n < targetset.size(); ++n) {
            t[n].join();
            for (int i = 0; i < results[i].size(); ++i) {
                results->push_back(tmpres[n][i]);
            }
        }
    */
    return true;
}


bool fastgetpermutations( const std::vector<vertextype> targetset, const graphtype g1, const graphtype g2,
    const FP* fps1, const FP* fps2, const int idx, graphmorphism partialmap, std::vector<graphmorphism>* results) {

    // implicit variable "cnt" == targetset.size

#ifdef NOTTHREADED2

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
    //std::vector<std::thread> t {};
    //t.resize(targetset.size());
    std::vector<std::vector<graphmorphism>> tmpres {};
    tmpres.resize(targetset.size());
    for (int n = 0; n < targetset.size(); ++n) {
        tmppartial = partialmap;
        tmppartial.push_back( {fps1[idx].v, fps2[targetset[n]].v});

        //t[n] = std::thread(fastgetpermutationscore,n, targetset, g1,g2,&(*fps1),&(*fps2),idx,tmppartial, &(tmpres[n]));
        //t[n] = std::thread(fastgetpermutationscoretmp,n, targetset, g1, &(*fps1), tmppartial, results); //,targetset,g1,g2,fps1,fps2,idx, tmppartial, &(results[n]));
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

/*
    for (int n = 0; n < targetset.size(); ++n) {
        t[n].join();
        for (int i = 0; i < results[i].size(); ++i) {
            results->push_back(tmpres[n][i]);
        }
    }
*/
#endif

#ifdef THREADED2

    unsigned const thread_count = std::thread::hardware_concurrency();
    //unsigned const thread_count = 1;


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
    std::vector<std::future<bool>> t {};
    t.resize(targetset.size());
    bool tmpbool[targetset.size()];
    std::vector<std::vector<graphmorphism>> tmpres {};
    tmpres.resize(targetset.size());
    for (int n = 0; n < targetset.size(); ++n) {
        tmppartial = partialmap;
        tmppartial.push_back( {fps1[idx].v, fps2[targetset[n]].v});

        std::vector<vertextype> newtargetset {};
        for (int i = 0; i < targetset.size(); ++i) { // this for loop because erase doesn't seem to work
            if (i != n)
                newtargetset.push_back(targetset[i]);
        }
        tmpbool[n] = ispartialisoonlynewest(g1,g2,tmppartial);
        if (tmpbool[n]) {
            t[n] = std::async(fastgetpermutationscore, newtargetset, g1,g2,&(*fps1),&(*fps2),idx,tmppartial, &(tmpres[n]));
        }
    }
    for (int n = 0; n < targetset.size(); ++n) {
        if (t[n].get()) {
            for (int i = 0; i < tmpres[n].size(); ++i) {
                results->push_back(tmpres[n][i]);
            }
        }
    }

    /*
        for (int n = 0; n < targetset.size(); ++n) {
            t[n].join();
            for (int i = 0; i < results[i].size(); ++i) {
                results->push_back(tmpres[n][i]);
            }
        }
    */







#endif

#ifdef THREADPOOL2

#endif
    return true;
}





std::vector<graphmorphism> threadrecurseisomorphisms(const int l, const int permsidx, const graphtype g1, const graphtype g2, const int* del,
    const FP* fps1, const FP* fps2, graphmorphism parentmap ) {
    //std::vector<graphmorphism> newmaps {};
    std::vector<graphmorphism> results {};
    std::vector<vertextype> targetset {};
    for (int j = 0; j < permsidx; ++j) {
        targetset.push_back(del[l]+j);
        //std::cout << targetset[targetset.size()-1] << "\n";;
    }
    if (permsidx > 0) {
        fastgetpermutations(targetset,g1,g2,fps1,fps2,del[l],parentmap,&(results));
        return results;
        //fastgetpermutations(targetset,g1,g2,fps1,fps2,del[l],parentmap,&(newmaps));

    }
    //return newmaps;
    return results;
}


/*
bool threadrecurseisomorphisms2(const int l, const int permsidx, const graph g1, const graph g2, const int* del, const FP* fps1, const FP* fps2, const std::vector<graphmorphism>* parentmaps, const int startidx, const int stopidx, std::vector<graphmorphism>* res) {
    for (int k = startidx; k < stopidx; ++k) {
        threadrecurseisomorphisms(l,permsidx,g1,g2,del,fps1,fps2,(*parentmaps)[k],res);
        //for (int i = 0; i < tmpnewmaps.size(); ++i) {
        //    res->push_back(tmpnewmaps[i]);
        //}
    }
    return true;
}
*/


std::vector<graphmorphism>* enumisomorphisms( neighborstype* ns1, neighborstype* ns2 ) {
    if (ns1 == nullptr || ns2 == nullptr)
        return {};
    graphtype* g1 = ns1->g;
    graphtype* g2 = ns2->g;
    std::vector<graphmorphism>* maps = new std::vector<graphmorphism>;
    maps->clear();
    if (g1->dim != g2->dim || ns1->maxdegree != ns2->maxdegree)
        return maps;

    // to do: save time in most cases by checking that ns1 matches ns2

    int dim = g1->dim;

    FP* fps1ptr = (FP*)malloc(dim * sizeof(FP));
    for (int n = 0; n < dim; ++n) {
        fps1ptr[n].v = n;
        fps1ptr[n].ns = nullptr;
        fps1ptr[n].nscnt = 0;
        fps1ptr[n].parent = nullptr;
        fps1ptr[n].invert = ns1->degrees[n] >= int(dim+1/2);
    }

    takefingerprint(ns1,fps1ptr,dim);

    sortneighbors(ns1,fps1ptr,dim);


    //osfingerprint(std::cout,ns1,fps1ptr,dim);

    FP* fps2ptr = (FP*)malloc(dim * sizeof(FP));

    for (int n = 0; n < dim; ++n) {
        fps2ptr[n].v = n;
        fps2ptr[n].ns = nullptr;
        fps2ptr[n].nscnt = 0;
        fps2ptr[n].parent = nullptr;
        fps2ptr[n].invert = ns2->degrees[n] >= int(dim+1/2);
    }

    takefingerprint(ns2,fps2ptr,dim);
    sortneighbors(ns2,fps2ptr,dim);

    //osfingerprint(std::cout,ns2,fps2,g2.dim);

    //vertextype del[ns1.maxdegree+2];
    //vertextype del[g1.dim+2];

    vertextype* delptr;
    delptr = (vertextype*)malloc((dim+2)*sizeof(vertextype));


    int delcnt = 0;
    delptr[delcnt] = 0;
    ++delcnt;
    for( vertextype n = 0; n < dim-1; ++n ) {
        if (ns1->degrees[fps1ptr[n].v] != ns2->degrees[fps2ptr[n].v])
            return maps; // return empty set of maps
        int res1 = FPcmp(ns1,ns1,&fps1ptr[n],&fps1ptr[n+1]);
        int res2 = FPcmp(ns2,ns2,&fps2ptr[n],&fps2ptr[n+1]);
        if (res1 != res2)
            return maps;  // return empty set of maps
        if (res1 < 0) {
            std::cout << "Error: not sorted (n == " << n << ", res1 == "<<res1<<")\n";
            return maps;
        }
        if (res1 > 0) {
            delptr[delcnt] = n+1;
            ++delcnt;
            //std::cout << "inc'ed delcnt\n";
        }
    }
    if (dim > 0)
        if (ns1->degrees[fps1ptr[dim-1].v] != ns2->degrees[fps2ptr[dim-1].v])
            return maps; // return empty set of maps

    delptr[delcnt] = dim;


#ifdef THREADED1
    unsigned const thread_count = std::thread::hardware_concurrency();
    //unsigned const thread_count = 1;
#endif

#ifdef THREADPOOL1
    thread_pool* pool = new thread_pool; //if not using the pool feature, uncommenting this leads to a stray thread
#endif

    //std::pair<vertextype,vertextype> basepair;
    //basepair = {-1,-1};
    std::vector<std::vector<std::vector<int>>> perms {};
    graphmorphism basemap {};
    maps->push_back(basemap);
    for (int l = 0; l<delcnt; ++l) {
        std::vector<graphmorphism> newmaps {};
        //std::cout << "maps.size == " << maps.size() << "\n";

        int permsidx = delptr[l+1]-delptr[l];

        if (permsidx < MAXFACTORIAL) {
            if (permsidx > perms.size())
                perms.resize(permsidx+1);
            if (perms[permsidx].size() == 0)
                perms[permsidx] = getpermutations(permsidx);

            for (int k = 0; k < maps->size(); ++k) {
                //std::cout << "del[l] == " << del[l] << ", del[l+1] == " << del[l+1] << "delcnt == " << delcnt << "\n";
                //std::vector<std::vector<int>> perm = fastgetpermutations(del[l+1]-del[l],g1,g2,fps1,fps2,del[l]);
                std::vector<std::vector<int>> perm = perms[permsidx];
                for (int i = 0; i < perm.size(); ++i) {
                    graphmorphism newmap = (*maps)[k];
                    for (int j = 0; j < perm[i].size(); ++j) {
                        std::pair<vertextype,vertextype> newpair;
                        //std::cout << "i,j, perm[i][j] == "<<i<<", "<<j<<", "<< perm[i][j]<<"\n";
                        newpair = {fps1ptr[delptr[l]+j].v,fps2ptr[delptr[l]+perm[i][j]].v};
                        newmap.push_back(newpair);
                    }
                    newmaps.push_back(newmap);
                }
            }
            maps->clear();
            for (int i = 0; i < newmaps.size(); ++i ) {
                if (ispartialiso(*g1,*g2,newmaps[i])) {
                    maps->push_back(newmaps[i]);
                }
            }
        } else {
            // the case when permsidx >= MAXFACTORIAL


#ifndef THREADED1
            for (int k = 0; k < maps->size(); ++k) {
                std::vector<vertextype> targetset {};
                for (int j = 0; j < permsidx; ++j) {
                    targetset.push_back(delptr[l]+j);
                    //std::cout << targetset[targetset.size()-1] << "\n";;
                }
                if (permsidx > 0) {
                    fastgetpermutations(targetset,*g1,*g2,fps1ptr,fps2ptr,delptr[l],(*maps)[k],&(newmaps));

                }
            }
            maps->clear();
            for (int i = 0; i < newmaps.size(); ++i) {
                if (newmaps[i].size() == delptr[l] + permsidx)
                    maps->push_back(newmaps[i]);
            }

#endif

#ifdef THREADPOOL1


            std::vector<std::future<std::vector<graphmorphism>>> threadpool;
            threadpool.resize(maps->size());
            //std::vector<std::future<int>> threadpool;
            //std::vector<std::vector<graphmorphism>> rireturn {};
            //threadholder* th = new threadholder;





            std::vector<std::vector<graphmorphism>> res;


            res.resize(maps->size());
            for (int k = 0; k < maps->size(); ++k) {
                //int a = 4;
                //threadpool[k] = pool.submit(std::bind(&threadholder::helloworld,this,a,tmpfps1,g1,&maps));
                res[k].clear();

                threadpool[k] = pool->submit(std::bind(&threadrecurseisomorphisms,l,
                    permsidx,g1, g2, delptr,fps1ptr,fps2ptr, (*maps)[k]));

                //                bool threadrecurseisomorphisms(int l, int permsidx, graph g1, graph g2, int* del, FP* fps1, FP* fps2, graphmorphism parentmap,std::vector<graphmorphism>* res) {

            }

            //            std::vector<graphmorphism> threadrecurseisomorphisms( const int permsidx, int* del, int k, int l, graph g1, graph g2, FP* fps1, FP* fps2,std::vector<graphmorphism>* maps ) {

            //            for (int m = 0; m < maps.size(); ++m) {
            //                threadpool[m] = pool.submit(std::bind(&Hellytheory::threadfindcovers,this,&Cvrs[m],&es) );
            //            }
            for (int m = 0; m < maps->size(); ++m) {
                while (threadpool[m].wait_for(std::chrono::seconds(0)) == std::future_status::timeout) {
                    pool->run_pending_task();
                }
                //std::vector<graphmorphism> mapreturned = threadpool[m].get();
                //bool mapreturned = threadpool[m].get();
                res[m] = threadpool[m].get();  // don't use the always true boolean return value, it is meaningless
                //threadpool[m].wait();
                //std::cout << "CvrReturned.size() " << CvrReturned.size() << "\n";
                for (int r = 0; r < res[m].size();++r) {
                    newmaps.push_back(res[m][r]);
                }
            }

            maps->clear();
            for (int i = 0; i < newmaps.size(); ++i) {
                if (newmaps[i].size() == (delptr[l] + permsidx))
                    maps->push_back(newmaps[i]);
            }

#endif

#ifdef THREADED1

// below is non-pooled standard threaded, working but not using the concurrent processes as of yet...

            float section = float(maps.size()) / float(thread_count);
            //std::cout << "maps.size() " <<maps.size()<< ", section size: " << section << ", thread_count: " << thread_count << "\n";

            //std::vector<std::future<bool>> t {};

            std::vector<std::future<bool>> t {};
            t.resize(thread_count);
            std::vector<graphmorphism> tmpmaps {};
            std::vector<std::vector<graphmorphism>> res {};
            if (maps.size() > 10000*thread_count) {
                res.resize(thread_count);
                for (int m = 0; m < thread_count; ++m) {
                    const int startidx = int(m*section);
                    int stopidx = int((m+1.0)*section);
                    std::cout << "start, stop " << startidx << " " << stopidx<< "\n";
                    t[m] = std::async(&threadrecurseisomorphisms2, l,permsidx,g1,g2,delptr,fps1ptr,fps2ptr,&maps, startidx, stopidx,&(res[m]));
                    //t[m] = std::async(&threadrecurseisomorphisms2, l,permsidx,g1,g2,delptr,fps1ptr,fps2ptr,&maps, startidx, stopidx,&(res[m]));
                }
                //std::vector<std::vector<graphmorphism>> gm {};
                //gm.resize(thread_count);
                for (int m = 0; m < thread_count; ++m) {
                    bool returned {};
                    const int startidx = int(m*section);
                    const int stopidx = int((m+1.0)*section);
                    //t[m].join();
                    //t[m].detach();
                    returned = t[m].get(); // "get" waits for the thread to return
                }
            } else {
                res.resize(1);
                threadrecurseisomorphisms2(l,permsidx,g1,g2,delptr,fps1ptr,fps2ptr,&maps,0,maps.size(),&(res[0]));
            }
            for (int n = 0; n < res.size(); ++n) {
                for (int i = 0; i < res[n].size(); ++i) {
                    if (res[n][i].size() == (del[l] + permsidx)) {
                        newmaps.push_back(res[n][i]);
                        //graphmorphism gm {};
                        //for (int i = 0; i < returned[n].size();++i) {
                        //    gm.push_back(returned[n][i]);
                        //}
                        //newmaps.push_back(gm);
                    }
                }
            }
            maps.clear();
            for (int n = 0; n < newmaps.size();++n) {
                maps.push_back(newmaps[n]);
            }
            //osgraphmorphisms(std::cout,maps);

#endif


        }
    }
    /*std::vector<graphmorphism> res {};
    for (int i = 0; i < maps.size(); ++i ) {
        if (isiso(g1,g2,maps[i])) {
            res.push_back(maps[i]);
        }
    }*/

    free(delptr);
    freefps(fps1ptr,g1->dim);
    free(fps1ptr);
    freefps(fps2ptr,g2->dim);
    free(fps2ptr);
#ifdef THREADPOOL1
    free(pool);
#endif
    //std::vector<graphmorphism>* tmp = new std::vector<graphmorphism>;
    //tmp->clear();

    return maps;
}

int edgecnt( graphtype* g ) {
    int res = 0;
    for (int n = 0; n < g->dim; ++n) {
        for (int i = n+1; i < g->dim; ++i) {
            if (g->adjacencymatrix[n*g->dim + i]) {
                res++;
            }
        }
    }
    return res;
}

void osfingerprintrecurse( std::ostream &os, neighbors* ns, FP* fps, int fpscnt, int depth ) {
    for (int i = 0; i < depth; ++i) {
        os << "   ";
    }
    os << "fpscnt == " << fpscnt << ": \n";
    for (int n = 0; n < fpscnt; ++n) {
        for (int i = 0; i < depth+1; ++i) {
            os << "   ";
        }
        os << "{v, deg v} == {" << fps[n].v << ", " << ns->degrees[fps[n].v] << "}\n";
        osfingerprintrecurse( os, ns, fps[n].ns, fps[n].nscnt,depth+1 );
        if (fps[n].nscnt == 0) {
            for (int i = 0; i < depth+1; ++i) {
                os << "   ";
            }
            FP* parent = fps[n].parent;
            //std::cout << "Total walk by degrees: <" << ns.degrees[fps[n].v] << ", ";
            while (parent != nullptr) {
                //std::cout << ns.degrees[parent->v] << ", ";
                parent = ((*parent).parent);
            }
            //std::cout << "\b\b>, ";
            parent = fps[n].parent;
            //std::cout << " and by vertices: <" << fps[n].v << ", ";
            while (parent != nullptr) {
                //std::cout << parent->v << ", ";
                parent = (*parent).parent;
            }
            //std::cout << "\b\b>\n";
        }
    }
    //for (int i = 0; i < depth; ++i) {
    //    os << "   ";
    //}
    //os << "end fpscnt == " << fpscnt << "\n";
}


void osfingerprintrecurseminimal( std::ostream &os, neighbors* ns, FP* fps, int fpscnt, std::vector<vertextype> path ) {
    std::vector<vertextype> tmppath {};
    for (int n = 0; n < fpscnt; ++n) {
        tmppath.clear();
        for (int j = 0; j < path.size(); ++j) {
            tmppath.push_back(path[j]);
        }
        tmppath.push_back(fps[n].v);
        osfingerprintrecurseminimal(os, ns, fps[n].ns, fps[n].nscnt,tmppath);
        if (fps[n].nscnt == 0) {
            for (int i = 0; i < tmppath.size(); ++i) {
                os << ns->degrees[tmppath[i]] << " ";
            }
            os << "\n";
        }
    }
}



void osfingerprint( std::ostream &os, neighbors* ns, FP* fps, int fpscnt ) {
    osfingerprintrecurse( os, ns, fps, fpscnt, 0 );
    os << "\n";
}

void osfingerprintminimal( std::ostream &os, neighbors* ns, FP* fps, int fpscnt ) {
    std::vector<vertextype> nullpath {};
    osfingerprintrecurseminimal( os, ns, fps, fpscnt, nullpath );
}


void osadjacencymatrix( std::ostream &os, graphtype* g ) {
    for (int n = 0; n < g->dim; ++n ) {
        for (int i = 0; i < g->dim; ++i) {
            os << g->adjacencymatrix[n*g->dim + i] << " ";
        }
        os << "\n";
    }
}

void osneighbors( std::ostream &os, neighborstype* ns ) {
    for (int n = 0; n < ns->g->dim; ++n ) {
        os << "ns.degrees["<<n<<"] == "<<ns->degrees[n]<<": ";
        for (int i = 0; i < ns->degrees[n]; ++i) {
            os << ns->neighborslist[n*ns->g->dim + i] << ", " ;
        }
        os << "\b\b\n";
    }
    for (int n = 0; n < ns->g->dim; ++n ) {
        os << "ns.degrees["<<n<<"] == "<<ns->degrees[n]<<" (non-neighbors): ";
        for (int i = 0; i < ns->g->dim - ns->degrees[n] - 1; ++i) {
            os << ns->nonneighborslist[n*ns->g->dim + i] << ", " ;
        }
        os << "\b\b\n";
    }
}

void osedges( std::ostream &os, graphtype* g) {
    if (g->adjacencymatrix == nullptr) {
        return;
    }
    bool found = false;
    for (int i = 0; i < g->dim-1; ++i) {
        for (int j = i+1; j < g->dim; ++j) {
            if (g->adjacencymatrix[g->dim*i + j]) {
                found = true;
                os << "[" << i << "," << j << "], ";
            }
        }
        if (found) {
            os << "\b\b\n";
            found = false;
        }
    }

}

void osgraphmorphisms( std::ostream &os, std::vector<graphmorphism>* maps ) {
    for (int n = 0; n < maps->size(); ++n) {
        os << "Map number " << n+1 << " of " << maps->size() << ":\n";
        for (int i = 0; i < maps[n].size(); ++i) {
            os << (*maps)[n][i].first << " maps to " << (*maps)[n][i].second << "\n";
        }
    }
}

