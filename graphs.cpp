//
// Created by peterglenn on 6/6/24.
//

// Use caution: one out of three must be asserted to be #defined:
// The working thread for 1 is THREADED1 (not THREADPOOL1) but slower
//#define THREADPOOL1
//#define THREADED1
#define NOTTHREADED1

// Use caution: one out of three must be asserted to be #defined:
// The THREADED2 functionality is now working, in some cases faster and in some slower
//#define THREADPOOL2
#define THREADED2
//#define NOTTHREADED2

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
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <ranges>
#include <vector>
#include <string>

#include "mathfn.cpp"
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


int FPcmp( const neighbors* ns1, const neighbors* ns2, const FP* w1, const FP* w2 ) { // acts without consideration of self or parents; looks only downwards

    if (ns1->g->dim != ns2->g->dim)
        return ns1->g->dim < ns2->g->dim ? 1 : -1;

    int multiplier = (w1->invert ? -1 : 1);
    //multiplier = 1;
    if (w1->invert != w2->invert)
        return multiplier;

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


int FPcmpextends( const vertextype* m1, const vertextype* m2, const neighbors* ns1, const neighbors* ns2, const FP* w1, const FP* w2 ) { // acts without consideration of self or parents; looks only downwards

    if (ns1->g->dim != ns2->g->dim)
        return ns1->g->dim < ns2->g->dim ? 1 : -1;

    int multiplier = (w1->invert ? -1 : 1);
    //multiplier = 1;
    if (w1->invert != w2->invert)
        return multiplier;

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
        res = FPcmpextends(m1,m2,ns1,ns2,&w1->ns[n],&w2->ns[n]);

        if (res == 0) {
            if (m1[w1->ns[n].v] != 0)
                if (m1[w1->ns[n].v] != (w2->ns[n].v +1) )
                    res = -1;
            if (m2[w2->ns[n].v] != 0)
                if (m2[w2->ns[n].v] != (w1->ns[n].v +1) )
                    res = -1;
        }

            // for (int i = 0; i < m->size(); ++i) {
                // if ((*m)[i].first == w1->ns[n].v)
                    // res = ((*m)[i].second == w2->ns[n].v) ? res : -1;
                // if ((*m)[i].second == w2->ns[n].v)
                    // res = ((*m)[i].first == w1->ns[n].v) ? res : -1;
            // }
        n++;
    }
    return res * multiplier;
}


inline int partition2( std::vector<int> &arr, int start, int end, const neighbors* ns, FP* fpslist ) {
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

inline void quickSort2( std::vector<int> &arr, int start, int end,const neighbors* ns, FP* fpslist ) {

    if (start >= end)
        return;

    int p = partition2(arr,start,end,ns,fpslist);

    quickSort2(arr, start, p-1,ns,fpslist);
    quickSort2(arr, p+1, end,ns,fpslist);
}

/*
inline int partition3( int start, int end, neighbors* ns, FP* fpslist ) {
    int pivot = start;
    int count = 0;
    for (int i = start+1;i <= end; i++) {
        if (FPcmp(ns,ns,&fpslist[i],&fpslist[pivot]) >= 0) {
            count++;
        }
    }

    int pivotIndex = start + count;
    std::swap(fpslist[pivotIndex],fpslist[start]);

    int i = start;
    int j = end;
    while (i < pivotIndex && j > pivotIndex) {
        while (FPcmp(ns,ns,&fpslist[i],&fpslist[pivot]) >= 0) {
            i++;
        }
        while (FPcmp(ns,ns,&fpslist[j],&fpslist[pivot]) < 0) {
            j--;
        }
        if (i < pivotIndex && j > pivotIndex) {
            std::swap(fpslist[i++],fpslist[j--]);
        }
    }
    return pivotIndex;
}

inline void quickSort3( int start, int end,neighbors* ns, FP* fpslist ) {

    if (start >= end)
        return;

    int p = partition3(start,end,ns,fpslist);

    quickSort3( start, p-1,ns,fpslist);
    quickSort3(p+1, end,ns,fpslist);
}
*/

void sortneighbors( const neighbors* ns, FP* fps, int fpscnt ) {


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

    // NOTE the code marked "ABSOLUTELY ESSENTIAL" below (it reassigns "parent" to point correctly after the sort)

#endif

}


void takefingerprint( const neighbors* ns, FP* fps, int fpscnt ) {

    if (fpscnt == 0 || ns == nullptr) {

        return;
    }

    const int dim = ns->g->dim;
    FP* sorted = new FP[dim];
    FP* sortedinverted = new FP[dim];
    int halfdegree = (dim+1)/2;
    bool invert = false;
    int idx = 0;
    int invertedidx = 0;

    //std::cout << "fpscnt == " << fpscnt << ", ns.maxdegree == " << ns.maxdegree << "\n";
    int md = ns->maxdegree;
    if (ns->maxdegree < halfdegree)
        md = dim - ns->maxdegree;
    //md = dim-1;
    int* sizeofdegree = new int[md+1];
    int* sizeofinverteddegree = new int[md+1];
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
                    //std::cout << "fps[vidx].invert == " << fps[vidx].invert << "\n";
                    sortedinverted[invertedidx] = fps[vidx];
                    ++invertedidx;
                    ++sizeofinverteddegree[d];
                }
            }
        }
    }


    int startidx = 0;
    int startidxinverted = 0;

    for (int i = 0; i <= md; ++i) {
        FP* fps2 = new FP[sizeofdegree[i]];
        //std::cout << "i, sizeofdegree[i] == " << i << ", " << sizeofdegree[i] << "\n";
        for (int j = 0; j < sizeofdegree[i]; ++j) {
            fps2[j] = sorted[startidx + j];
        }
        sortneighbors( ns, fps2, sizeofdegree[i] );
        for (int j = 0; j < sizeofdegree[i]; ++j) {
            sorted[startidx+j] = fps2[j];
        }
        startidx = startidx + sizeofdegree[i];
        delete fps2;
    }
    int startidx2 = 0;
    for (int i = 0; i<=md; ++i) {
        FP* fps2 = new FP[sizeofinverteddegree[i]];
        //std::cout << "i, sizeofinverteddegree[i] == " << i << ", " << sizeofinverteddegree[i] << "\n";
        for (int k = 0; k < sizeofinverteddegree[i]; ++k) {
            fps2[k] = sortedinverted[startidx2 + k];
        }
        sortneighbors( ns, fps2, sizeofinverteddegree[i] );
        for (int j = 0; j < sizeofinverteddegree[i]; ++j) {
            sortedinverted[startidx2+j] = fps2[sizeofinverteddegree[i] - j - 1];
        }
        startidx2 = startidx2 + sizeofinverteddegree[i];
        delete fps2;
    }



    // ABSOLUTELY ESSENTIAL CODE NOT TO FORGET WHEN CALLING SORTNEIGHBORS IN ANY OTHER CONTEXT:

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
    delete sorted;
    delete sortedinverted;
    delete sizeofdegree;
    delete sizeofinverteddegree;
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

bool ispartialisoonlynewest( const graphtype* g1, const graphtype* g2, const graphmorphism* map) {
    bool match = true;
    int i = map->size()-1;
    for (int n = 0; match && (n < map->size()-1); ++n) {
        match = match && (g1->adjacencymatrix[(*map)[n].first*g1->dim + (*map)[i].first] == g2->adjacencymatrix[(*map)[n].second*g2->dim + (*map)[i].second]);
    }
    return match;
}


bool ispartialiso( const graphtype* g1, const graphtype* g2, graphmorphism* map) {
    bool match = true;
    for (int n = 0; match && (n < map->size()-1); ++n) {
        for (int i = n+1; match && (i < map->size()); ++i) {
            match = match && (g1->adjacencymatrix[(*map)[n].first*g1->dim + (*map)[i].first] == g2->adjacencymatrix[(*map)[n].second*g2->dim + (*map)[i].second]);
        }
    }
    return match;
}

bool isiso( const graphtype* g1, const graphtype* g2, const graphmorphism* map ) {
    bool match = true;
    if (g1->dim != g2->dim) {
        return false;
    }
    for (vertextype n = 0; (n < g1->dim-1) && match; ++n ) {
        for (vertextype i = n+1; (i < g1->dim) && match; ++i ) {
            match = match && (g1->adjacencymatrix[(*map)[n].first*g1->dim + (*map)[i].first] == g2->adjacencymatrix[(*map)[n].second*g2->dim + (*map)[i].second]);
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


bool fastgetpermutationscore( const std::vector<vertextype>* targetset, const graphtype* g1, const graphtype* g2, const FP* fps1, const FP* fps2, const int idx, graphmorphism partialmap, std::vector<graphmorphism>* resptr) {
/*
           if (targetset.size()>0) {
                partialmap.push_back({fps1[idx].v,fps2[targetset[0]].v});
                resptr->push_back(partialmap);
                return true;
            }
    */

    // implicit variable "cnt" == targetset.size

    std::vector<graphmorphism> res {};
    if (targetset->size() <= 1) {
        partialmap.push_back( {fps1[idx].v,fps2[(*targetset)[0]].v});
        //resptr->push_back(partialmap);
        if (ispartialisoonlynewest(g1,g2,&partialmap)) {
            resptr->push_back(partialmap);
            return true;
        } else {
            //resptr->resize(resptr->size()-1); // can't seem to get to work erase(partialmap->size()-1);
            return false;
        }
    }
    graphmorphism tmppartial {};
    //std::vector<std::thread> t {};
    //t.resize(targetset->size());
    std::vector<std::vector<graphmorphism>> tmpres {};
    tmpres.resize(targetset->size());
    for (int n = 0; n < targetset->size(); ++n) {
        tmppartial = partialmap;
        tmppartial.push_back( {fps1[idx].v, fps2[(*targetset)[n]].v});

        //t[n] = std::thread(&fastgetpermutationscore, targetset, g1,g2,fps1,fps2,idx+1,tmppartial, &(tmpres[n]));
        //t[n] = std::thread(fastgetpermutationscoretmp,n, targetset, g1, &(*fps1), tmppartial, results); //,targetset,g1,g2,fps1,fps2,idx, tmppartial, &(results[n]));
        std::vector<vertextype> newtargetset {};
        for (int i = 0; i < targetset->size(); ++i) { // this for loop because erase doesn't seem to work
            if (i != n)
                newtargetset.push_back((*targetset)[i]);
        }
        if (ispartialisoonlynewest(g1,g2,&tmppartial)) {
            if (fastgetpermutationscore(&newtargetset,g1,g2,fps1,fps2,idx+1,tmppartial,resptr)) {
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


bool fastgetpermutations( const std::vector<vertextype>* targetset, const graphtype* g1, const graphtype* g2,
    const FP* fps1, const FP* fps2, const int idx, graphmorphism partialmap, std::vector<graphmorphism>* results) {
    // implicit variable "cnt" == targetset.size

#ifdef NOTTHREADED2

    std::vector<graphmorphism> res {};
    if (targetset->size() <= 1) {
        partialmap.push_back( {fps1[idx].v,fps2[(*targetset)[0]].v});
        results->push_back(partialmap);
        if (ispartialisoonlynewest(g1,g2,&partialmap)) {
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
    tmpres.resize(targetset->size());
    for (int n = 0; n < targetset->size(); ++n) {
        tmppartial = partialmap;
        tmppartial.push_back( {fps1[idx].v, fps2[(*targetset)[n]].v});

        //t[n] = std::thread(fastgetpermutationscore,n, targetset, g1,g2,&(*fps1),&(*fps2),idx,tmppartial, &(tmpres[n]));
        //t[n] = std::thread(fastgetpermutationscoretmp,n, targetset, g1, &(*fps1), tmppartial, results); //,targetset,g1,g2,fps1,fps2,idx, tmppartial, &(results[n]));
        std::vector<vertextype> newtargetset {};
        for (int i = 0; i < targetset->size(); ++i) { // this for loop because erase doesn't seem to work
            if (i != n)
                newtargetset.push_back((*targetset)[i]);
        }
        if (ispartialisoonlynewest(g1,g2,&tmppartial)) {
            if (fastgetpermutations(&newtargetset,g1,g2,fps1,fps2,idx+1,tmppartial,results)) {
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
    if (targetset->size() <= 1) {
        partialmap.push_back( {fps1[idx].v,fps2[(*targetset)[0]].v});
        //results->push_back(partialmap);
        if (ispartialisoonlynewest(g1,g2,&partialmap)) {
            results->push_back(partialmap);
            return true;
        } else {
            //results->resize(results->size()-1); // can't seem to get to work erase(partialmap->size()-1);
            return false;
        }
    }
    std::vector<graphmorphism> tmppartialv {};
    tmppartialv.resize(targetset->size());
    std::vector<std::future<bool>> t {};
    t.resize(targetset->size());
    bool* tmpbool = new bool[targetset->size()];
    std::vector<bool> boolres {};
    boolres.resize(targetset->size());
    std::vector<std::vector<graphmorphism>> tmpres {};
    tmpres.resize(targetset->size());
    std::vector<std::vector<vertextype>> newtargetsetv {};
    newtargetsetv.resize(targetset->size());
    for (int n = 0; n < targetset->size(); ++n) {
        tmppartialv[n] = partialmap;
        tmppartialv[n].push_back( {fps1[idx].v, fps2[(*targetset)[n]].v});

        //std::vector<vertextype> newtargetset {};
        newtargetsetv[n].clear();
        for (int i = 0; i < targetset->size(); ++i) { // this for loop because erase doesn't seem to work
            if (i != n)
                newtargetsetv[n].push_back((*targetset)[i]);
        }
    }
    for (int n =0; n < targetset->size(); ++n) {
        tmpbool[n] = ispartialisoonlynewest(g1,g2,&tmppartialv[n]);
        if (tmpbool[n]) {
            tmpres[n].clear();
            t[n] = std::async(&fastgetpermutationscore, &newtargetsetv[n], g1,g2,fps1,fps2,idx+1,tmppartialv[n], &tmpres[n]);
            //boolres[n] = fastgetpermutationscore(&newtargetsetv[n],g1,g2,fps1,fps2,idx+1,tmppartialv[n],&tmpres[n]);
        }
    }

    for (int n = 0; n < targetset->size(); ++n) {
        if (tmpbool[n]) {
            boolres[n] = (t[n].get());
        }
    }
    for (int n = 0; n < targetset->size();++n)
        if (tmpbool[n] && boolres[n])
            for (int i = 0; i < tmpres[n].size(); ++i) {
                results->push_back(tmpres[n][i]);
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

    //unsigned const thread_count = std::thread::hardware_concurrency();
    //unsigned const thread_count = 1;

    thread_pool* pool = new thread_pool; //if not using the pool feature, uncommenting this leads to a stray thread



    std::vector<graphmorphism> res {};
    if (targetset->size() <= 1) {
        partialmap.push_back( {fps1[idx].v,fps2[(*targetset)[0]].v});
        //results->push_back(partialmap);
        if (ispartialisoonlynewest(g1,g2,&partialmap)) {
            results->push_back(partialmap);
            return true;
        } else {
            //results->resize(results->size()-1); // can't seem to get to work erase(partialmap->size()-1);
            return false;
        }
    }
    std::vector<graphmorphism> tmppartialv {};
    tmppartialv.resize(targetset->size());
    std::vector<std::future<bool>> t {};
    t.resize(targetset->size());
    bool tmpbool[targetset->size()];
    std::vector<bool> boolres {};
    boolres.resize(targetset->size());
    std::vector<std::vector<graphmorphism>> tmpres {};
    tmpres.resize(targetset->size());
    std::vector<std::vector<vertextype>> newtargetsetv {};
    newtargetsetv.resize(targetset->size());
    for (int n = 0; n < targetset->size(); ++n) {
        tmppartialv[n] = partialmap;
        tmppartialv[n].push_back( {fps1[idx].v, fps2[(*targetset)[n]].v});

        //std::vector<vertextype> newtargetset {};
        newtargetsetv[n].clear();
        for (int i = 0; i < targetset->size(); ++i) { // this for loop because erase doesn't seem to work
            if (i != n)
                newtargetsetv[n].push_back((*targetset)[i]);
        }
    }
    for (int n =0; n < targetset->size(); ++n) {
        tmpbool[n] = ispartialisoonlynewest(g1,g2,&tmppartialv[n]);
        if (tmpbool[n]) {
            tmpres[n].clear();
            t[n] = std::async(&fastgetpermutationscore, &newtargetsetv[n], g1,g2,fps1,fps2,idx+1,tmppartialv[n], &tmpres[n]);
            //boolres[n] = fastgetpermutationscore(&newtargetsetv[n],g1,g2,fps1,fps2,idx+1,tmppartialv[n],&tmpres[n]);
        }
    }

    for (int n = 0; n < targetset->size(); ++n) {
        if (tmpbool[n]) {
            boolres[n] = (t[n].get());
        }
    }
    for (int n = 0; n < targetset->size();++n)
        if (tmpbool[n] && boolres[n])
            for (int i = 0; i < tmpres[n].size(); ++i) {
                results->push_back(tmpres[n][i]);
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

    delete tmpbool;

    return true;
}





std::vector<graphmorphism> threadrecurseisomorphisms(const int l, const int permsidx, const graphtype* g1, const graphtype* g2, const int* del,
    const FP* fps1, const FP* fps2, graphmorphism parentmap ) {
    //std::vector<graphmorphism> newmaps {};
    std::vector<graphmorphism> results {};
    //results = new std::vector<graphmorphism>;
    std::vector<vertextype> targetset {};
    for (int j = 0; j < permsidx; ++j) {
        targetset.push_back(del[l]+j);
        //std::cout << targetset[targetset.size()-1] << "\n";;
    }
    if (permsidx > 0) {
        fastgetpermutations(&targetset,g1,g2,fps1,fps2,del[l],parentmap,&results);
        return results;
        //fastgetpermutations(targetset,g1,g2,fps1,fps2,del[l],parentmap,&(newmaps));

    }
    //return newmaps;
    return results;
}



bool threadrecurseisomorphisms2(const int l, const int permsidx, const graphtype* g1, const graphtype* g2, const int* del, const FP* fps1, const FP* fps2, const std::vector<graphmorphism>* parentmaps, const int startidx, const int stopidx, std::vector<graphmorphism>* res) {
    for (int k = startidx; k < stopidx; ++k) {
        std::vector<graphmorphism> gm {};
        gm = threadrecurseisomorphisms(l,permsidx,g1,g2,del,fps1,fps2,(*parentmaps)[k]);
        for (int n = 0; n < gm.size();++n)
            res->push_back(gm[n]);
        //for (int i = 0; i < tmpnewmaps.size(); ++i) {
        //    res->push_back(tmpnewmaps[i]);
        //}
    }
    return true;
}


std::vector<graphmorphism>* enumisomorphismscore( const neighborstype* ns1, const neighborstype* ns2, const FP* fps1ptr, const FP* fps2ptr ) {
    if (ns1 == nullptr || ns2 == nullptr)
        return {};
    graphtype* g1 = ns1->g;
    graphtype* g2 = ns2->g;
    int dim = g1->dim;
    std::vector<graphmorphism>* maps = new std::vector<graphmorphism>;
    maps->clear();

    vertextype* delptr;
    delptr = (vertextype*)malloc((dim+2)*sizeof(vertextype));

    if (dim != g2->dim) {
        return maps;
    }


    int delcnt = 0;
    delptr[delcnt] = 0;
    int* delsizes = new int[dim+2];
    int* delsortedbysize = new int[dim+2];
    for (int n = 0; n < dim+2; ++n)
        delsizes[n] = 0;
    delsizes[0] = 1;
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
        delsizes[delcnt-1]++;
    }
    for (int n = 0; n <= delcnt; ++n) {
        delsortedbysize[n] = n;
    }
    delsizes[delcnt]--;

    struct {
        int* delsizes;
        bool operator()(int a, int b) const { return delsizes[a]<delsizes[b]; }
    }
    customLess;

    customLess.delsizes = delsizes;
    delptr[delcnt] = dim;

    std::sort(&delsortedbysize[0],&delsortedbysize[delcnt],customLess);

    if (dim > 0)
        if (ns1->degrees[fps1ptr[dim-1].v] != ns2->degrees[fps2ptr[dim-1].v])
            return maps; // return empty set of maps
    //for (int n = 0; n <= delcnt; ++n) {
    //    std::cout << "delsizes " << delsizes[n] <<  "\n";
    //}

    //for (int n = 0; n < delcnt+1; ++n) {
    //    std::cout << "n "<<n<<"delsortedbysize " << delsortedbysize[n] << " delptr[delsortedbysize[]]= " << delptr[delsortedbysize[n]]<< " "<<delsizes[delsortedbysize[n]] << "\n";
   // }


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
    int fullmapsize = 0;
    for (int l = 0; l<delcnt; ++l) {
        std::vector<graphmorphism> newmaps {};
        //std::cout << "maps.size == " << maps.size() << "\n";

        int permsidx = delsizes[delsortedbysize[l]]; //delptr[delsortedbysize[l+1]]-delptr[delsortedbysize[l]];
        fullmapsize += permsidx;
        //std::cout << "permsidx = " << permsidx << "\n";
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
                        newpair = {fps1ptr[delptr[delsortedbysize[l]]+j].v,fps2ptr[delptr[delsortedbysize[l]]+perm[i][j]].v};
                        newmap.push_back(newpair);
                    }
                    newmaps.push_back(newmap);
                }
            }
            maps->clear();
            for (int i = 0; i < newmaps.size(); ++i ) {
                if (ispartialiso(g1,g2,&newmaps[i])) {
                    maps->push_back(newmaps[i]);
                }
            }
        } else {
            // the case when permsidx >= MAXFACTORIAL


#ifdef NOTTHREADED1
            for (int k = 0; k < maps->size(); ++k) {
                std::vector<vertextype> targetset {};
                for (int j = 0; j < permsidx; ++j) {
                    targetset.push_back(delptr[delsortedbysize[l]]+j);
                    //std::cout << targetset[targetset.size()-1] << "\n";;
                }
                if (permsidx > 0) {
                    fastgetpermutations(&targetset,g1,g2,fps1ptr,fps2ptr,delptr[delsortedbysize[l]],(*maps)[k],&(newmaps));

                }
            }
            maps->clear();
            for (int i = 0; i < newmaps.size(); ++i) {
                //int fullmapsize = 0;
                //for (int j =0; j < l; ++j)
                //    fullmapsize += delsizes[delsortedbysize[j]];
                if (newmaps[i].size() == fullmapsize)
                    maps->push_back(newmaps[i]);


                //if (newmaps[i].size() == delptr[delsortedbysize[l]] + permsidx)
                //    maps->push_back(newmaps[i]);
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

                threadpool[k] = pool->submit(std::bind(&threadrecurseisomorphisms,l,
                    permsidx,g1, g2, delptr,fps1ptr,fps2ptr, &((*maps)[k])));

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

            double section = double(maps->size()) / double(thread_count);
            //std::cout << "maps.size() " <<maps.size()<< ", section size: " << section << ", thread_count: " << thread_count << "\n";

            //std::vector<std::future<bool>> t {};

            std::vector<std::future<bool>> t {};
            t.resize(thread_count);
            std::vector<graphmorphism> tmpmaps {};
            std::vector<std::vector<graphmorphism>> res {};
            res.resize(thread_count);
            for (int m = 0; m < thread_count; ++m) {
                const int startidx = int(m*section);
                int stopidx = int((m+1.0)*section);
                //std::cout << "start, stop " << startidx << " " << stopidx<< "\n";
                t[m] = std::async(&threadrecurseisomorphisms2, l,permsidx,g1,g2,delptr,fps1ptr,fps2ptr,maps, startidx, stopidx,&(res[m]));
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
            for (int n = 0; n < res.size(); ++n) {
                for (int i = 0; i < res[n].size(); ++i) {
                    if (res[n][i].size() == (delptr[l] + permsidx)) {
                        newmaps.push_back(res[n][i]);
                        //graphmorphism gm {};
                        //for (int i = 0; i < returned[n].size();++i) {
                        //    gm.push_back(returned[n][i]);
                        //}
                        //newmaps.push_back(gm);
                    }
                }
            }
            maps->clear();
            for (int n = 0; n < newmaps.size();++n) {
                maps->push_back(newmaps[n]);
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
#ifdef THREADPOOL1
    free(pool);
#endif
    //std::vector<graphmorphism>* tmp = new std::vector<graphmorphism>;
    //tmp->clear();

    delete delsizes;
    delete delsortedbysize;
    return maps;


}

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
        fps1ptr[n].invert = ns1->degrees[n] >= int((dim+1)/2);
    }

    takefingerprint(ns1,fps1ptr,dim);

    //sortneighbors(ns1,fps1ptr,dim);


    //osfingerprint(std::cout,ns1,fps1ptr,dim);

    FP* fps2ptr;

    if (g1 == g2)
        fps2ptr = fps1ptr;
    else {
        fps2ptr = (FP*)malloc(dim * sizeof(FP));

        for (int n = 0; n < dim; ++n) {
            fps2ptr[n].v = n;
            fps2ptr[n].ns = nullptr;
            fps2ptr[n].nscnt = 0;
            fps2ptr[n].parent = nullptr;
            fps2ptr[n].invert = ns2->degrees[n] >= int((dim+1)/2);
        }

        takefingerprint(ns2,fps2ptr,dim);
        //sortneighbors(ns2,fps2ptr,dim);
    }
    //osfingerprint(std::cout,ns2,fps2,g2.dim);

    //vertextype del[ns1.maxdegree+2];
    //vertextype del[g1.dim+2];

    maps = enumisomorphismscore(ns1,ns2,fps1ptr,fps2ptr);

    freefps(fps1ptr,g1->dim);
    free(fps1ptr);
    if (fps2ptr != fps1ptr) {
        freefps(fps2ptr,g2->dim);
        free(fps2ptr);
    }

    return maps;
}

int edgecnt( const graphtype* g ) {
    int res = 0;
    for (int n = 0; n < g->dim-1; ++n) {
        for (int i = n+1; i < g->dim; ++i) {
            if (g->adjacencymatrix[n*g->dim + i]) {
                res++;
            }
        }
    }
    return res;
}


bool existsisocore( const neighbors* ns1, const neighbors* ns2, const FP* fp1, const FP* fp2) {
    return (FPcmp(ns1,ns2,fp1,fp2) == 0 ? true : false );

}


bool existsiso( const neighbors* ns1, FP* fps1ptr, const neighbors* ns2) {
    if (ns1 == nullptr || ns2 == nullptr)
        return {};
    graphtype* g1 = ns1->g;
    graphtype* g2 = ns2->g;
    if (g1->dim != g2->dim || ns1->maxdegree != ns2->maxdegree)
        return false;

    // to do: save time in most cases by checking that ns1 matches ns2

    int dim = g1->dim;

    bool responsibletofree = false;

    if (fps1ptr == nullptr) {
        responsibletofree = true;
        FP* fps1ptr = (FP*)malloc(dim * sizeof(FP));
        for (int n = 0; n < dim; ++n) {
            fps1ptr[n].v = n;
            fps1ptr[n].ns = nullptr;
            fps1ptr[n].nscnt = 0;
            fps1ptr[n].parent = nullptr;
            fps1ptr[n].invert = ns1->degrees[n] >= int((dim+1)/2);
        }

        takefingerprint(ns1,fps1ptr,dim);

        //sortneighbors(ns1,fps1ptr,dim);


        //osfingerprint(std::cout,ns1,fps1ptr,dim);
    }

    FP* fps2ptr;

    if (g1 == g2)
        fps2ptr = fps1ptr;
    else {
        fps2ptr = (FP*)malloc(dim * sizeof(FP));

        for (int n = 0; n < dim; ++n) {
            fps2ptr[n].v = n;
            fps2ptr[n].ns = nullptr;
            fps2ptr[n].nscnt = 0;
            fps2ptr[n].parent = nullptr;
            fps2ptr[n].invert = ns2->degrees[n] >= int((dim+1)/2);
        }

        takefingerprint(ns2,fps2ptr,dim);
        //sortneighbors(ns2,fps2ptr,dim);
    }
    //osfingerprint(std::cout,ns2,fps2,g2.dim);

    //vertextype del[ns1.maxdegree+2];
    //vertextype del[g1.dim+2];

    bool res = existsisocore(ns1,ns2,fps1ptr,fps2ptr);

    if (responsibletofree) {
        freefps(fps1ptr,g1->dim);
        free(fps1ptr);
    }
    if (fps2ptr != fps1ptr) {
        freefps(fps2ptr,g2->dim);
        free(fps2ptr);
    }

    return res;

}



bool existsiso2(const int* m1, const int* m2, const graphtype* g1, const neighbors* ns1, const graphtype* g2, const neighbors* ns2 )
{
    int dim1 = g1->dim;
    int dim2 = g2->dim;

    if (dim1 != dim2)
        return false;


    bool res;

    FP* fps1 = new FP[dim1];
    for (vertextype n = 0; n < dim1; ++n) {
        fps1[n].v = n;
        fps1[n].ns = nullptr;
        fps1[n].nscnt = 0;
        fps1[n].parent = nullptr;
    }

    takefingerprint(ns1,fps1,dim1);

    //osfingerprint(std::cout,ns5,fps5,g5.dim);

    FP* fps2 = new FP[dim2];
    for (vertextype n = 0; n < dim2; ++n) {
        fps2[n].v = n;
        fps2[n].ns = nullptr;
        fps2[n].nscnt = 0;
        fps2[n].parent = nullptr;
    }

    takefingerprint(ns2,fps2,dim2);

    /*
    FP fpstmp5;
    fpstmp5.parent = nullptr;
    fpstmp5.ns = fps5;
    fpstmp5.nscnt = dim;

    FP fpstmp6;
    fpstmp6.parent = nullptr;
    fpstmp6.ns = fps6;
    fpstmp6.nscnt = dim;
    */
    //osfingerprint(std::cout,ns6,fps6,g6.dim);
    //if (FPcmp(ns5,ns6,&fpstmp5,&fpstmp6) == 0) {



    if (m1 != nullptr)
        if (FPcmpextends(m1,m2,ns1,ns2,fps1,fps2) == 0) {
            //std::cout << "Fingerprints MATCH\n";
            res = true;
        } else {
            //std::cout << "Fingerprints DO NOT MATCH\n";
            res = false;
        }
    else
        if (FPcmp(ns1,ns2,fps1,fps2) == 0) {
            res = true;
        } else {
            res = false;
        }

    freefps(fps1, dim1);
    freefps(fps2, dim2);
    delete fps1;
    delete fps2;
    return res;
}






void enumsizedsubsets(int sizestart, int sizeend, int* seq, int start, int stop, std::vector<int>* res) {
    if (start > stop)
        return;
    if (sizestart >= sizeend) {
        for (int i = 0; i < sizeend; ++i)
            res->push_back(seq[i]);
        return;
    }
    // pseudo code:
    //      for each n in [start,stop-1]
    //          call enumsizedsubsets on (s + n), sizestart+1,sizeend, n+1,stop;
    //          call enumsizedsubsets on s, sizestart,sizeend, n+1, stop

    int* newseq = new int[sizestart+1];
    for (int i = 0; i < sizestart; ++i) {
        newseq[i] = seq[i];
    }
    newseq[sizestart] = start;
    enumsizedsubsets(sizestart+1,sizeend,newseq,start+1,stop,res);
    enumsizedsubsets(sizestart,sizeend,seq,start+1,stop,res);
    delete newseq;
}




bool embeds( const neighbors* ns1, FP* fp, const neighbors* ns2, const int mincnt ) {
    //graphtype* g1 = ns1->g;
    graphtype* g2 = ns2->g;
    int dim1 = ns1->g->dim;
    int dim2 = g2->dim;
    if (dim2 < dim1)
        return false;
    int numberofsubsets; // = nchoosek(dim2,dim1);
    //int* subsets = (int*)malloc(numberofsubsets*dim1*sizeof(int));
    std::vector<int> subsets {};
    enumsizedsubsets(0,dim1,nullptr,0,dim2,&subsets);
    numberofsubsets = subsets.size()/dim1;
    /*if (numberofsubsets*dim1 != subsets.size()) {
        std::cout << "Counting error in 'embeds': "<< numberofsubsets << " != "<<subsets.size()<< "\n";
        return false;
    }*/
    int cnt = 0;
    auto gtemp = new graphtype(dim1);
    for (int n = 0; (cnt < mincnt) && (n < numberofsubsets); ++n) {
        for (int i = 0; i < dim1; ++i) {
            vertextype g2vertex1 = subsets[dim1*n + i];
            for (int j = 0; j < dim1; ++j) {
                vertextype g2vertex2 = subsets[dim1*n + j];
                gtemp->adjacencymatrix[i*dim1+j] = g2->adjacencymatrix[g2vertex1*dim2 + g2vertex2];
            }
        }

        // note this code obviously might be much faster
        // when instead simply checking iso for the identity map
        // (that is, allowing the subsets above to be a larger set
        // that contains rearrangements of things already in the set)
        //nstemp->computeneighborslist();
        auto nstemp = new neighbors(gtemp);
        cnt = (existsiso(ns1,fp,nstemp) ? cnt+1 : cnt);
        //res = res || existsiso( ns1, fp, nstemp );
        free(nstemp);

    }
    free(gtemp);

    //free(subsets);
    return cnt >= mincnt;
}

class quicktest {
public:
    virtual bool test(int* testseq) {return false;}
};

class embedsquicktest : public quicktest {
public:
    graphtype* gtemp;
    const graphtype* g2;
    const neighbors* ns1;
    FP* fp;
    const int dim1;
    const int dim2;
    bool test(int* testseq) override {
        for (int i = 0; i < dim1; ++i) {
            vertextype g2vertex1 = testseq[i];
            for (int j = 0; j < dim1; ++j) {
                vertextype g2vertex2 = testseq[j];
                gtemp->adjacencymatrix[i*dim1+j] = g2->adjacencymatrix[g2vertex1*dim2 + g2vertex2];
            }
        }

        bool resbool;
        auto nstemp = new neighbors(gtemp);
        resbool = existsiso(ns1,fp,nstemp);
        free(nstemp);
        return resbool;
    }
    embedsquicktest( graphtype* gtempin, const graphtype* g2in, const neighbors* ns1in, FP* fpin, const int dim1in, const int dim2in)
        : gtemp{gtempin}, g2{g2in}, ns1{ns1in}, fp{fpin}, dim1{dim1in}, dim2{dim2in} {}

};



bool enumsizedsubsetsquick(int sizestart, int sizeend, int* seq, int start, int stop, int* cnt, const int mincnt, quicktest* test) {
    if (start > stop)
        return false;
    if (sizestart >= sizeend) {
        //for (int i = 0; i < sizeend; ++i) {
        if (test->test(seq)) {
            ++(*cnt);
            if (*cnt >= mincnt)
                return true;
        }
        //res->push_back(seq[i]);
        return false;
    }
    // pseudo code:
    //      for each n in [start,stop-1]
    //          call enumsizedsubsets on (s + n), sizestart+1,sizeend, n+1,stop;
    //          call enumsizedsubsets on s, sizestart,sizeend, n+1, stop

    int* newseq = new int[sizestart+1];
    for (int i = 0; i < sizestart; ++i) {
        newseq[i] = seq[i];
    }
    newseq[sizestart] = start;
    if (enumsizedsubsetsquick(sizestart + 1, sizeend, newseq, start + 1, stop, cnt, mincnt, test)) {
        delete newseq;
        return true;
    }
    if (enumsizedsubsetsquick(sizestart, sizeend, seq, start + 1, stop, cnt, mincnt, test)) {
        delete newseq;
        return true;
    }
    delete newseq;
    return false;
}


bool embedsquick( const neighbors* ns1, FP* fp, const neighbors* ns2, const int mincnt ) {
    //graphtype* g1 = ns1->g;
    graphtype* g2 = ns2->g;
    int dim1 = ns1->g->dim;
    int dim2 = g2->dim;
    if (dim2 < dim1)
        return false;
    int numberofsubsets; // = nchoosek(dim2,dim1);
    //int* subsets = (int*)malloc(numberofsubsets*dim1*sizeof(int));
    //std::vector<int> subsets {};
    //enumsizedsubsets(0,dim1,nullptr,0,dim2,&subsets);
    //numberofsubsets = subsets.size()/dim1;
    /*if (numberofsubsets*dim1 != subsets.size()) {
        std::cout << "Counting error in 'embeds': "<< numberofsubsets << " != "<<subsets.size()<< "\n";
        return false;
    }*/

    int cnt = 0;
    auto gtemp = new graphtype(dim1);
    auto test = new embedsquicktest(gtemp,g2,ns1,fp,dim1,dim2);

    bool resbool = enumsizedsubsetsquick(0,dim1,nullptr,0,dim2,&cnt,mincnt, test);

        // note this code obviously might be much faster
        // when instead simply checking iso for the identity map
        // (that is, allowing the subsets above to be a larger set
        // that contains rearrangements of things already in the set)
        //nstemp->computeneighborslist();

    delete gtemp;
    delete test;

    //free(subsets);
    return resbool;
}



bool enumsizedsubsetsquicktally(int sizestart, int sizeend, int* seq, int start, int stop, int* cnt, quicktest* test) {
    if (start > stop)
        return false;
    if (sizestart >= sizeend) {
        //for (int i = 0; i < sizeend; ++i) {
        if (test->test(seq)) {
            ++(*cnt);
//            if (*cnt >= mincnt)
//                return true;
        }
        //res->push_back(seq[i]);
        return false;
    }
    // pseudo code:
    //      for each n in [start,stop-1]
    //          call enumsizedsubsets on (s + n), sizestart+1,sizeend, n+1,stop;
    //          call enumsizedsubsets on s, sizestart,sizeend, n+1, stop

    int* newseq = new int[sizestart+1];
    for (int i = 0; i < sizestart; ++i) {
        newseq[i] = seq[i];
    }
    newseq[sizestart] = start;
    if (enumsizedsubsetsquicktally(sizestart + 1, sizeend, newseq, start + 1, stop, cnt, test)) {
        delete newseq;
        return true;
    }
    if (enumsizedsubsetsquicktally(sizestart, sizeend, seq, start + 1, stop, cnt, test)) {
        delete newseq;
        return true;
    }
    delete newseq;
    return false;
}





int embedscount( const neighbors* ns1, FP* fp, const neighbors* ns2) {
    //graphtype* g1 = ns1->g;
    graphtype* g2 = ns2->g;
    int dim1 = ns1->g->dim;
    int dim2 = g2->dim;
    if (dim2 < dim1)
        return false;
    int numberofsubsets; // = nchoosek(dim2,dim1);
    //int* subsets = (int*)malloc(numberofsubsets*dim1*sizeof(int));
    //std::vector<int> subsets {};
    //enumsizedsubsets(0,dim1,nullptr,0,dim2,&subsets);
    //numberofsubsets = subsets.size()/dim1;
    /*if (numberofsubsets*dim1 != subsets.size()) {
        std::cout << "Counting error in 'embeds': "<< numberofsubsets << " != "<<subsets.size()<< "\n";
        return false;
    }*/

    int cnt = 0;
    auto gtemp = new graphtype(dim1);
    auto test = new embedsquicktest(gtemp,g2,ns1,fp,dim1,dim2);

    enumsizedsubsetsquicktally(0,dim1,nullptr,0,dim2,&cnt,test);

    // note this code obviously might be much faster
    // when instead simply checking iso for the identity map
    // (that is, allowing the subsets above to be a larger set
    // that contains rearrangements of things already in the set)
    //nstemp->computeneighborslist();

    delete gtemp;
    delete test;

    //free(subsets);
    return cnt;
}



class kconnectedtest : public quicktest {
public:
    graphtype* gtemp;
    neighborstype* nstemp;
    const int seqsize;
    bool test(int* testseq) {
        int dim = gtemp->dim;
        bool* subsetv = (bool*)malloc(dim*sizeof(bool));
        memset(subsetv,true,dim*sizeof(bool));
        int idx = 0;
        if (seqsize > 0)
            subsetv[testseq[idx++]] = false;
        while (idx < seqsize) {
            subsetv[testseq[idx]] = false;
            ++idx;
        }
        int dim2 = dim - seqsize;
        auto gptr = new graphtype(dim2);
        int offseti = 0;
        int offsetj = 0;
        for (int i = 0; i < dim2; ++i) {
            while (!subsetv[offseti])
                ++offseti;
            if (offseti < dim) {
                offsetj = offseti + 1;
                for (int j = i+1; j < dim2; ++j) {
                    while (!subsetv[offsetj])
                        ++offsetj;
                    if (offsetj < dim) {
                        gptr->adjacencymatrix[i*dim2 + j] = gtemp->adjacencymatrix[offseti*dim + offsetj];
                        gptr->adjacencymatrix[j*dim2 + i] = gtemp->adjacencymatrix[offsetj*dim + offseti];
                    }
                    ++offsetj;
                }
            }
            ++offseti;
        }

        for (int i = 0; i < dim2; ++i)
            gptr->adjacencymatrix[i*dim2 + i] = false;

        bool resbool;
        auto nsptr = new neighbors(gptr);
        // osadjacencymatrix(std::cout, gtemp);
        // osadjacencymatrix(std::cout, gptr);
        auto c = connectedcount(gptr,nsptr,2);
        free(nsptr);
        free(gptr);
        delete subsetv;
        return (c > 1 ? false : true);
    }
    kconnectedtest( graphtype* gtempin, neighborstype* nsin, const int seqsizein)
        : gtemp{gtempin},  nstemp{nsin}, seqsize{seqsizein} {}

};








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


void osadjacencymatrix( std::ostream &os, const graphtype* g ) {
    int* labelssize = new int[g->dim];
    int maxlabelsize = 0;
    bool labels = g->vertexlabels.size() == g->dim;
    if (labels) {
        for (int n = 0; n < g->dim; ++n) {
            labelssize[n] = g->vertexlabels[n].size();
            maxlabelsize = maxlabelsize >= labelssize[n] ? maxlabelsize : labelssize[n];
        }
        os << " ";
        for (int n = 0; n < maxlabelsize; ++n) {
            os << " ";
        }
        for (int n = 0; n < g->dim; ++n) {
            os << g->vertexlabels[n] << " ";
        }
        os << "\n";
    }
    for (int n = 0; n < g->dim; ++n ) {
        if (labels) {
            os << g->vertexlabels[n] << " ";
            for (int j = 0; j < maxlabelsize - labelssize[n]; ++j)
                os << " ";
        }
        for (int i = 0; i < g->dim; ++i) {
            char truechar = '1';
            char falsechar = '0';
            if (g->adjacencymatrix[n*g->dim + i])
                os << truechar << " ";
            else
                os << falsechar << " ";
            //os << g->adjacencymatrix[n*g->dim + i] << " ";
            if (labels)
                for (int j = 0; j < labelssize[i]-1; ++j)
                    os << " ";
        }
        os << "\n";
    }
    delete labelssize;
}

void osneighbors( std::ostream &os, const neighborstype* ns ) {
    bool labels = ns->g->vertexlabels.size()==ns->g->dim;
    int* labelssize = new int[ns->g->dim];
    int maxlabelsize = 0;
    if (labels) {
        for (int n = 0; n < ns->g->dim; ++n) {
            labelssize[n] = ns->g->vertexlabels[n].size();
            maxlabelsize = maxlabelsize >= labelssize[n] ? maxlabelsize : labelssize[n];
        }
    }
    for (int n = 0; n < ns->g->dim; ++n ) {
        os << "ns.degrees[";
        if (labels) {
            os << ns->g->vertexlabels[n];
        }
        else {
            os << n;
        }
        if (labels) {
            for (int i = 0; i < maxlabelsize - labelssize[n]; ++i )
                os << " ";
        }
        os <<"] == "<<ns->degrees[n]<<": ";
        for (int i = 0; i < ns->degrees[n]; ++i) {
            if (labels) {
                os << ns->g->vertexlabels[ns->neighborslist[n*ns->g->dim + i]];
            } else {
                os << ns->neighborslist[n*ns->g->dim + i];
            }
            os << ", ";
        }
        os << "\b\b\n";
    }

/*
    for (int n = 0; n < ns->g->dim; ++n ) {
        os << "ns.degrees["<<n<<"] == "<<ns->degrees[n]<<" (non-neighbors): ";
        for (int i = 0; i < ns->g->dim - ns->degrees[n] - 1; ++i) {
            os << ns->nonneighborslist[n*ns->g->dim + i] << ", " ;
        }
        os << "\b\b\n";
    }*/
    delete labelssize;
}

void osedges( std::ostream &os, const graphtype* g) {
    int* labelssize = new int[g->dim];
    int maxlabelsize = 0;
    bool labels = g->vertexlabels.size() == g->dim;
    if (labels) {
        for (int n = 0; n < g->dim; ++n) {
            labelssize[n] = g->vertexlabels[n].size();
            maxlabelsize = maxlabelsize >= labelssize[n] ? maxlabelsize : labelssize[n];
        }
        if (g->adjacencymatrix == nullptr) {
            return;
        }
    }
    bool found = false;
    for (int i = 0; i < g->dim-1; ++i) {
        for (int j = i+1; j < g->dim; ++j) {
            if (g->adjacencymatrix[g->dim*i + j]) {
                found = true;
                os << "[";
                if (labels)
                    os << g->vertexlabels[i]<<","<<g->vertexlabels[j];
                else
                    os << i << "," << j;
                os << "], ";
            }
        }
        if (found) {
            os << "\b\b\n";
            found = false;
        }
    }
    delete labelssize;
}

void osgraphmorphisms( std::ostream &os, const graphtype* g1, const graphtype* g2, const std::vector<graphmorphism>* maps ) {
    int* labelssize1 = new int[g1->dim];
    int maxlabelsize1 = 0;
    bool labels1 = g1->vertexlabels.size() == g1->dim;
    if (labels1) {
        for (int n = 0; n < g1->dim; ++n) {
            labelssize1[n] = g1->vertexlabels[n].size();
            maxlabelsize1 = maxlabelsize1 >= labelssize1[n] ? maxlabelsize1 : labelssize1[n];
        }
        if (g1->adjacencymatrix == nullptr) {
            delete labelssize1;
            return;
        }
    }
    int* labelssize2 = new int[g2->dim];
    int maxlabelsize2 = 0;
    bool labels2 = g2->vertexlabels.size() == g2->dim;
    if (labels2) {
        for (int n = 0; n < g2->dim; ++n) {
            labelssize2[n] = g2->vertexlabels[n].size();
            maxlabelsize2 = maxlabelsize2 >= labelssize2[n] ? maxlabelsize2 : labelssize2[n];
        }
        if (g2->adjacencymatrix == nullptr) {
            delete labelssize1;
            delete labelssize2;
            return;
        }
    }
    for (int n = 0; n < maps->size(); ++n) {
        os << "Map number " << n+1 << " of " << maps->size() << ":\n";
        for (int i = 0; i < (*maps)[n].size(); ++i) {
            if (labels1)
                os << g1->vertexlabels[(*maps)[n][i].first];
            else
                os << (*maps)[n][i].first;
            os << " maps to ";
            if (labels2)
                os << g2->vertexlabels[(*maps)[n][i].second];
            else
                os << (*maps)[n][i].second;
            os << "\n";
        }
    }
    delete labelssize1;
    delete labelssize2;
}

void osmachinereadablegraph(std::ostream &os, graphtype* g) {
    if (g->vertexlabels.size() != g->dim) {
        char ch = 'a';
        int idx = 1;
        for (int i = 0; i < g->dim; ++i) {
            if (idx == 1)
                g->vertexlabels.push_back( std::string{ ch++ });
            else
                g->vertexlabels.push_back(std::string{ch++} + std::to_string(idx));
            if (ch == 'z') {
                idx++;
                ch = 'a';
            }
        }
    }
    for (auto l : g->vertexlabels) {
        os << l << " ";
    }
    os << "END\n";
    for (int i = 0; i < g->dim-1; ++i) {
        bool emptyline = true;
        for (int j = i+1; j < g->dim; ++j) {
            if (g->adjacencymatrix[i*g->dim + j]) {
                os << g->vertexlabels[i] << g->vertexlabels[j] << " ";
                emptyline = false;
            }
        }
        if (!emptyline)
            os << "\n";
    }
    os << "END\n";
};

void zerograph(graphtype* g) {
    for (int i = 0; i < g->dim; ++i)
        for (int j = 0; j < g->dim; ++j)
            g->adjacencymatrix[i*g->dim + j] = false;
}



graphtype* cyclegraph( const int dim ) {
    auto res = new graphtype(dim);
    zerograph(res);
    if (dim > 2) {
        for (int i = 0; i < dim-1; ++i) {
            res->adjacencymatrix[i*dim + (i + 1)] = true;
            res->adjacencymatrix[(i+1)*dim + i] = true;
        }
        res->adjacencymatrix[0 + dim - 1] = true;
        res->adjacencymatrix[(dim-1)*dim + 0] = true;
    }
    return res;
}


int connectedcount(graphtype *g, neighborstype *ns, const int breaksize) {
    int dim = g->dim;
    if (dim <= 0)
        return 0;

    int* visited = (int*)malloc(dim*sizeof(int));

    for (int i = 0; i < dim; ++i)
    {
        visited[i] = -1;
    }

    visited[0] = 0;
    int res = 1;
    bool allvisited = false;
    while (!allvisited)
    {
        bool changed = false;
        for ( int i = 0; i < dim; ++i)
        {
            if (visited[i] >= 0)
            {
                for (int j = 0; j < ns->degrees[i]; ++j)
                {
                    vertextype nextv = ns->neighborslist[i*dim+j];
                    // loop = if a neighbor of vertex i is found in visited
                    // and that neighbor is not the origin of vertex i

                    if (visited[nextv] < 0)
                    {
                        visited[nextv] = i;
                        changed = true;
                    }
                }
            }
        }
        if (!changed) {
            allvisited = true;
            int firstunvisited = 0;
            while( allvisited && (firstunvisited < dim))
            {
                allvisited = allvisited && (visited[firstunvisited] != -1);
                ++firstunvisited;
            }
            if (allvisited)
            {
                free (visited);
                return res;
            }
            res++;
            if (breaksize >=0 && res >= breaksize)
            {
                free (visited);
                return res;
            }
            visited[firstunvisited-1] = firstunvisited-1;
            changed = true;
        }
    }
    return res;
}

bool kconnectedfn( graphtype* g, neighborstype* ns, const int k ) {

    bool res = true;
    for (int i = 0; res && (i < k); ++i) {
        int cnt = 0;
        auto test = new kconnectedtest(g,ns,i);
        int c = nchoosek(g->dim,i);
        res = res && enumsizedsubsetsquick(0,i,nullptr, 0,g->dim,&cnt,c, test);
    }

    return res;

}

void copygraph( graphtype* g1, graphtype* g2 ) {
    if (g1->dim != g2->dim) {
        std::cout << "Mismatched dimensions in copygraph\n";
        return;
    }
    for (int i = 0; i < g1->dim * g1->dim; ++i)
        g2->adjacencymatrix[i] = g1->adjacencymatrix[i];
}

int pathsbetweentally( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2) {
    if (v1 == v2)
        return 1;
    auto g2 = new graphtype(g->dim);
    //copygraph( g, g2 );
    int res = 0;
    for (int i = 0; i < ns->degrees[v1]; ++i) {
        copygraph( g, g2 );
        vertextype v = ns->neighborslist[v1*g->dim + i];
        for (int j = 0; j < ns->degrees[v1]; ++j)
        {
            vertextype v3 = ns->neighborslist[v1*g->dim + j];
            g2->adjacencymatrix[v1*g->dim + v3] = false;
            g2->adjacencymatrix[v3*g->dim + v1] = false;
        }
        // g2->adjacencymatrix[v1*g->dim + v] = false;
        // g2->adjacencymatrix[v*g->dim + v1] = false;
        auto ns2 = new neighborstype(g2);
        res += pathsbetweentally(g2,ns2, v,v2);
        delete ns2;
    }
    delete g2;
    return res;
}



void pathsbetweentuplesinternal( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2, std::vector<vertextype> existingpath, std::vector<std::vector<vertextype>>& out )
{
    if (v1 == v2)
    {
        out.push_back(existingpath);
        return;
    }
    auto g2 = new graphtype(g->dim);
    //copygraph( g, g2 );
    // int res = 0;
    for (int i = 0; i < ns->degrees[v2]; ++i) {
        copygraph( g, g2 );
        vertextype v = ns->neighborslist[v2*g->dim + i];
        // existingpath.insert(existingpath.begin(),v);
        for (int j = 0; j < ns->degrees[v2]; ++j)
        {
            vertextype v3 = ns->neighborslist[v2*g->dim + j];
            g2->adjacencymatrix[v2*g->dim + v3] = false;
            g2->adjacencymatrix[v3*g->dim + v2] = false;
        }
        existingpath.push_back(v);
        // g2->adjacencymatrix[v2*g->dim + v] = false;
        // g2->adjacencymatrix[v*g->dim + v2] = false;
        auto ns2 = new neighborstype(g2);
        pathsbetweentuplesinternal(g2,ns2, v1,v,existingpath, out);
        delete ns2;
        existingpath.resize(existingpath.size()-1);
        // .erase(existingpath.begin(),existingpath.begin()+1);
    }
    delete g2;
}




void pathsbetweentuples( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2, std::vector<std::vector<vertextype>>& out )
{
    // std::cout << "v1 == " << v1 << ", v2 == " << v2 << "\n";
    std::vector<vertextype> existingpath {};
    existingpath.push_back(v1);
    pathsbetweentuplesinternal( g, ns, v2, v1, existingpath, out );
}

int directedcyclestallyinternal( graphtype* g, neighborstype* ns, vertextype v1, vertextype v2) {

    if (v1 == v2)
        return 1;
    auto g2 = new graphtype(g->dim);
    //copygraph( g, g2 );
    int res = 0;
    for (int i = 0; i < ns->degrees[v1]; ++i) {
        copygraph( g, g2 );
        vertextype v = ns->neighborslist[v1*g->dim + i];
        for (int j = 0; j < ns->degrees[v1]; ++j)
        {
            vertextype v3 = ns->neighborslist[v1*g->dim + j];
            g2->adjacencymatrix[v1*g->dim + v3] = false;
            g2->adjacencymatrix[v3*g->dim + v1] = false;
        }
        // g2->adjacencymatrix[v1*g->dim + v] = false;
        // g2->adjacencymatrix[v*g->dim + v1] = false;
        auto ns2 = new neighborstype(g2);
        res += pathsbetweentally(g2,ns2, v,v2);
        delete ns2;
    }
    delete g2;
    return res;
}


int directedcyclestally( graphtype* g, neighborstype* ns, vertextype v1 )
{
    auto g2 = new graphtype(g->dim);
    //copygraph( g, g2 );
    int res = 0;
    for (int i = 0; i < ns->degrees[v1]; ++i) {
        copygraph( g, g2 );
        vertextype v = ns->neighborslist[v1*g->dim + i];
        g2->adjacencymatrix[v1*g->dim + v] = false;
        g2->adjacencymatrix[v*g->dim + v1] = false;
        auto ns2 = new neighborstype(g2);
        res += directedcyclestallyinternal(g2,ns2, v,v1);
        delete ns2;
    }
    delete g2;
    return res;
}

void cyclesset( graphtype* g, neighborstype* ns, vertextype v, std::vector<std::vector<vertextype>>& out )
{
    auto g2 = new graphtype(g->dim);
    for (int i = 0; i < ns->degrees[v]; ++i)
    {
        copygraph( g, g2 );
        vertextype v2 = ns->neighborslist[v*g->dim + i];
        g2->adjacencymatrix[v*g->dim + v2] = false;
        g2->adjacencymatrix[v2*g->dim + v] = false;
        auto ns2 = new neighborstype(g2);
        pathsbetweentuples(g2,ns2, v, v2, out);
        delete ns2;
    }
    for (auto p : out)
        p.push_back(v);
    delete g2;
}
