//
// Created by peterglenn on 6/10/24.
//

#ifndef PROB_H
#define PROB_H
#include <future>
#include <iostream>
#include <random>
#include <iostream>

#include "asymp.h"
#include "graphs.h"

#define RANDOMRANGE 10000
#define RANDOMRANGEFLOAT 10000.0

using weightstype = std::vector<float>;

class abstractrandomgraph {
public:
    virtual std::string shortname() {return "_arg";};
    const std::string name;

    virtual void randomgraph( graphtype* gptr ) {}
    virtual std::vector<graphtype*> randomgraphs( const int cnt ) { return {};}
    abstractrandomgraph(const std::string namein) : name{namein} {}
};

class abstractparameterizedrandomgraph : public abstractrandomgraph {
public:
    std::vector<std::string> ps;
    virtual void setparams( std::vector<std::string> psin ) {
        ps = psin;
    }
    abstractparameterizedrandomgraph(const std::string namein) : abstractrandomgraph(namein) {}
};

class legacyabstractrandomgraph : public abstractrandomgraph {
public:
    std::vector<graphtype*> randomgraphsinternal(const int dim, const float edgecnt, const int cnt) {
        std::vector<graphtype*> res {};
        res.resize(cnt+1);
        for (int i = 0; i < cnt+1; ++i) {
            graphtype* rg = new graphtype(dim);
            randomgraph(rg,edgecnt);;
            res[i] = rg;
        }
        return res;
    }


    virtual void randomgraph( graphtype* gptr, const float edgecnt ) {};
    /* The rules for the above virtual function: it must not modify the object's variables
     * lest it create a contention when threaded. Use local variables instead,
     * e.g. such as _edgecnt; just use the local version of it. */

    std::vector<graphtype*> randomgraphs( const int dim, const float edgecnt, const int cnt ) {
        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        float section = float(cnt)/float(thread_count);
        // Note: MUST play safe in the abstractrandomgraph class so it won't contend

        std::vector<std::future<std::vector<graphtype*>>> t {};
        t.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            t[j] = std::async(&legacyabstractrandomgraph::randomgraphsinternal,this,dim,edgecnt,section);
        }
        std::vector<std::vector<graphtype*>> gvv {};
        gvv.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            gvv[j] = t[j].get();
        }
        std::vector<graphtype*> res {};
        res.resize(cnt);
        //std::cout << (thread_count-1)*section + int(section) << "<-- highest index\n";
        for (int j = 0; j < thread_count; ++j) {
            //std::cout<<"section " << section << " , j*section  ="<< j*section << "\n";
            for (int i = 0; i < gvv[j].size(); ++i) {
                res[int(j*section + i)] = gvv[j][i];
                //res.push_back(gvv[j][i]);
            }
        }
        return res;
    }

    legacyabstractrandomgraph(std::string namein) : abstractrandomgraph(namein) {}
};

/*
class abstractrandomgraph {

    std::vector<graphtype*> randomgraphsinternal(const int dim, const float edgecnt, const int cnt) {
        std::vector<graphtype*> res {};
        res.resize(cnt+1);
        for (int i = 0; i < cnt+1; ++i) {
            graphtype* rg = new graphtype(dim);
            randomgraph(rg,edgecnt);;
            res[i] = rg;
        }
        return res;
    }

public:
    virtual std::string shortname() {return "";};
    std::string name;

    virtual void randomgraph( graphtype* gptr, const float edgecnt ) {};
    //* The rules for the above virtual function: it must not modify the object's variables
    // * lest it create a contention when threaded. Use local variables instead,
    // * e.g. such as _edgecnt; just use the local version of it.

    std::vector<graphtype*> randomgraphs( const int dim, const float edgecnt, const int cnt ) {
        unsigned const thread_count = std::thread::hardware_concurrency();
        //unsigned const thread_count = 1;

        float section = float(cnt)/float(thread_count);
        // Note: MUST play safe in the abstractrandomgraph class so it won't contend

        std::vector<std::future<std::vector<graphtype*>>> t {};
        t.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            t[j] = std::async(&abstractrandomgraph::randomgraphsinternal,this,dim,edgecnt,section);
        }
        std::vector<std::vector<graphtype*>> gvv {};
        gvv.resize(thread_count);
        for (int j = 0; j < thread_count; ++j) {
            gvv[j] = t[j].get();
        }
        std::vector<graphtype*> res {};
        res.resize(cnt);
        //std::cout << (thread_count-1)*section + int(section) << "<-- highest index\n";
        for (int j = 0; j < thread_count; ++j) {
            //std::cout<<"section " << section << " , j*section  ="<< j*section << "\n";
            for (int i = 0; i < gvv[j].size(); ++i) {
                res[int(j*section + i)] = gvv[j][i];
                //res.push_back(gvv[j][i]);
            }
        }
        return res;
    }

    abstractrandomgraph() {};
    ~abstractrandomgraph() {};


};*/

/*
class abstractrandomgraphptrtof : public abstractrandomgraph {
public:
    void (*randomgraphfunc)(graph* gptr, float edgecnt);

    void randomgraph( graph* gptr, float edgecnt ) override {
        abstractrandomgraph::randomgraph(gptr, edgecnt);
        randomgraphfunc(gptr,edgecnt);
    }
};*/

// E.G. a non-OOP function for use with abstractrandomgraphptrtof

/*
inline void stdrandomgraphfunc( graph* gptr, float edgecnt ) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);
    for (int n = 0; n < gptr->dim; ++n) {
        gptr->adjacencymatrix[n*gptr->dim + n] = 0;
        for (int i = n+1; i < gptr->dim; ++i) {
            gptr->adjacencymatrix[n*gptr->dim + i] = (dist10000(rng) < (RANDOMRANGE*float(edgecnt)/(gptr->dim * (gptr->dim-1)/2.0)));
            gptr->adjacencymatrix[i*gptr->dim + n] = gptr->adjacencymatrix[n*gptr->dim+i];
        }
    }

}*/

class legacystdrandomgraph : public legacyabstractrandomgraph {
    float _edgecnt;
public:
    std::string shortname() {return "r1";};
    legacystdrandomgraph() : legacyabstractrandomgraph("random graph with edgecnt probability") {}
    virtual void randomgraph( graphtype* gptr, float edgecnt ) override {
        legacyabstractrandomgraph::randomgraph(gptr,edgecnt);
        //name = "random graph with edgecnt probability " + std::to_string(_edgecnt);
        //_edgecnt = edgecnt;

        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);
        for (int n = 0; n < gptr->dim; ++n) {
            gptr->adjacencymatrix[n*gptr->dim + n] = 0;
            for (int i = n+1; i < gptr->dim; ++i) {
                gptr->adjacencymatrix[n*gptr->dim + i] = (dist10000(rng) < (RANDOMRANGE*float(edgecnt)/(gptr->dim * (gptr->dim-1)/2.0)));
                gptr->adjacencymatrix[i*gptr->dim + n] = gptr->adjacencymatrix[n*gptr->dim+i];
            }
        }
    }
};

class legacyrandomgraphonnedges : public legacyabstractrandomgraph {
    int _edgecnt;
public:
    std::string shortname() {return "r2";}
    legacyrandomgraphonnedges() : legacyabstractrandomgraph("random graph with given edgecnt") {}
    void randomgraph( graphtype*  gptr, float edgecnt ) {
        //_edgecnt = (int)edgecnt;
        //name = "random graph with edgecnt == " + std::to_string(edgecnt);
        if (edgecnt > (gptr->dim * gptr->dim / 2)) {
            std::cout << "Too many edges requested of randomgraph\n";
            return;
        }
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);
        for (int n = 0; n < gptr->dim; ++n) {
            for (int i = 0; i < gptr->dim; ++i) {
                gptr->adjacencymatrix[n*gptr->dim + i] = false;
            }
        }
        int n = 0;

        while (n < edgecnt) {
            vertextype v1 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
            vertextype v2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
            if ((v1 != v2) && !gptr->adjacencymatrix[v1*gptr->dim + v2]) {
                gptr->adjacencymatrix[v1*gptr->dim + v2] = true;
                gptr->adjacencymatrix[v2*gptr->dim + v1] = true;
                ++n;
            }
        }
    }
};


class legacyrandomconnectedgraphfixededgecnt : public legacyabstractrandomgraph {
    int _edgecnt;

public:
    std::string shortname() {return "r3";}
    legacyrandomconnectedgraphfixededgecnt() : legacyabstractrandomgraph("random connected graph with given fixed edge count (ignoring unconnected outliers)") {}
    void randomgraph( graphtype* gptr, float edgecnt ) {
        //_edgecnt = edgecnt;
        //name = "random connected graph with fixed edge count " + std::to_string(_edgecnt) + " (ignoring unconnected outliers)";
        for (int i = 0; i < gptr->dim; ++i) {
            for (int j = 0; j < gptr->dim; ++j) {
                gptr->adjacencymatrix[gptr->dim*i + j] = false;
            }
        }
        if (edgecnt == 0)
            return;
        if (edgecnt >= (int)((gptr->dim*(gptr->dim-1)/2))) {
            //populate entire adjacency matrix
            for (int i = 0; i < gptr->dim; ++i) {
                for (int j = 0; j < gptr->dim; ++j) {
                    gptr->adjacencymatrix[i*gptr->dim + j] = (i != j);
                }
                return;
            }
        }
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);
        std::vector<vertextype> edges {};
        vertextype v1 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
        vertextype v2 = v1;
        while (v2 == v1) {
            v2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
        }
        bool visited[gptr->dim];
        for (vertextype i = 0; i < (gptr->dim); ++i) {
            visited[i]=false;
        }
        visited[v1] = true;
        visited[v2] = true;
        int visitedcnt = 2;
        gptr->adjacencymatrix[v1*(gptr->dim) + v2] = true;
        gptr->adjacencymatrix[v2*(gptr->dim) + v1] = true;
        int cnt = edgecnt-1;
        while (cnt > 0) {
            vertextype candidatev1index = (vertextype)(float((dist10000(rng)) * float(visitedcnt))/RANDOMRANGEFLOAT);
            int cnt2 = 0;
            vertextype i = -1;
            while (cnt2 <= candidatev1index) {
                ++i;
                if (visited[i]) {
                    cnt2++;
                }
            }
            vertextype candidatev1 = i;
            //std::cout << "candidatev1 == i == " << i << "\n";
            vertextype candidatev2 = candidatev1;
            while (candidatev2 == candidatev1) {
                candidatev2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
            }
            if ((candidatev2 >= gptr->dim) || (candidatev1 >= gptr->dim))
                std::cout << "ERROR\n";
            if (!visited[candidatev2]) {
                visitedcnt++;
                visited[candidatev2] = true;
            }
            if (!(gptr->adjacencymatrix[(candidatev1*gptr->dim) + candidatev2])) {
                gptr->adjacencymatrix[candidatev1*(gptr->dim) + candidatev2] = true;
                gptr->adjacencymatrix[candidatev2*(gptr->dim) + candidatev1] = true;
                cnt--;
            }
        }
    }
};

class legacyrandomconnectedgraph : public legacyabstractrandomgraph {
    int _edgecnt;
public:
    std::string shortname() {return "r4";}
    legacyrandomconnectedgraph() : legacyabstractrandomgraph("random connected graph (algorithm does not find all such graphs...) with given edgecnt") {}
    void randomgraph( graphtype*  gptr, float edgecnt ) override {
        legacyabstractrandomgraph::randomgraph(gptr, edgecnt);
        //_edgecnt = (int)edgecnt;
        //name = "random connected graph (algorithm does not find all such graphs...) edgecnt == " + std::to_string(_edgecnt);
        for (int i = 0; i < gptr->dim; ++i) {
            for (int j = 0; j < gptr->dim; ++j) {
                gptr->adjacencymatrix[gptr->dim*i + j] = false;
            }
        }
        if (gptr->dim == 0)
            return;
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);
        std::vector<vertextype> edges {};
        vertextype v1 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
        vertextype v2 = v1;
        while (v2 == v1) {
            v2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
        }
        bool visited[gptr->dim];
        for (vertextype i = 0; i < (gptr->dim); ++i) {
            visited[i]=false;
        }
        visited[v1] = true;
        visited[v2] = true;
        int visitedcnt = 2;
        gptr->adjacencymatrix[v1*(gptr->dim) + v2] = true;
        gptr->adjacencymatrix[v2*(gptr->dim) + v1] = true;
        while (visitedcnt<gptr->dim) {
            vertextype candidatev1index = (vertextype)(float((dist10000(rng)) * float(visitedcnt))/RANDOMRANGEFLOAT);
            int cnt2 = 0;
            vertextype i = -1;
            while (cnt2 <= candidatev1index) {
                ++i;
                if (visited[i]) {
                    cnt2++;
                }
            }
            vertextype candidatev1 = i;
            //std::cout << "candidatev1 == i == " << i << "\n";
            vertextype candidatev2 = candidatev1;
            while (candidatev2 == candidatev1) {
                candidatev2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
            }
            if ((candidatev2 >= gptr->dim) || (candidatev1 >= gptr->dim))
                std::cout << "ERROR\n";
            if (!visited[candidatev2]) {
                visitedcnt++;
                visited[candidatev2] = true;
            }
            if (!(gptr->adjacencymatrix[(candidatev1*gptr->dim) + candidatev2])) {
                gptr->adjacencymatrix[candidatev1*(gptr->dim) + candidatev2] = true;
                gptr->adjacencymatrix[candidatev2*(gptr->dim) + candidatev1] = true;
            }
        }
    }
};



inline std::vector<weightstype> computeweights(int n) {
    std::vector<weightstype> res {};
    for (int i = 0; i < n; ++i) {
        weightstype thisweights {};
        float weight = 1.0;
        for (int k = 0; k < i; ++k) {
            weight = weight/2.0;
            thisweights.push_back(weight);
        }
        weightstype tmpweights {};
        for (int k = 1; k <= i; ++k) {
            tmpweights.push_back(thisweights[i-k]);
        }
        res.push_back(tmpweights);
        //for (int i = 0; i < tmpweights.size(); ++i)
        //    std::cout << i << " with prob " << tmpweights[i] << ", ";
        //std::cout << "\n";
    }
    return res;
}

class legacyweightedrandomconnectedgraph : public legacyabstractrandomgraph {
    //std::vector<weightstype> _weights;
public:
    std::string shortname() {return "r5";}
    legacyweightedrandomconnectedgraph() : legacyabstractrandomgraph("random connected graph with balanced/weighted search") {}
    void randomgraph( graphtype*  gptr, float edgecnt ) {
        std::vector<weightstype> weights = computeweights(gptr->dim); // obviously would be nice to only do this once per dim...
        //name = "random connected graph with balanced/weighted search";
        for (int i = 0; i < gptr->dim; ++i) {
            for (int j = 0; j < gptr->dim; ++j) {
                gptr->adjacencymatrix[gptr->dim*i + j] = false;
            }
        }
        if (gptr->dim == 0)
            return;
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);
        std::vector<vertextype> edges {};
        vertextype v1 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
        vertextype v2 = v1;
        while (v2 == v1) {
            v2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
        }
        bool visited[gptr->dim];
        for (vertextype i = 0; i < (gptr->dim); ++i) {
            visited[i]=false;
        }
        visited[v1] = true;
        visited[v2] = true;
        int visitedcnt = 2;
        gptr->adjacencymatrix[v1*(gptr->dim) + v2] = true;
        gptr->adjacencymatrix[v2*(gptr->dim) + v1] = true;
        vertextype lastv1 = v1;
        while (visitedcnt < gptr->dim) {
            int tmprnd = dist10000(rng);
            vertextype candidatev1index = 0;
            while (candidatev1index < visitedcnt-1 && tmprnd > RANDOMRANGEFLOAT*weights[visitedcnt][candidatev1index+1])
                ++candidatev1index;
            if (candidatev1index > visitedcnt-1) {
                std::cout << "Error in candidatev1index\n";
                return;
            }

            //vertextype candidatev1index = (vertextype)(float((dist10000(rng)) * float(visitedcnt))/10000.0);


            int cnt2 = 0;
            vertextype i = -1;
            while (cnt2 <= candidatev1index) {
                ++i;
                if (visited[i]) {
                    cnt2++;
                }
            }
            vertextype candidatev1 = i;
            //std::cout << "candidatev1 == i == " << i << "\n";
            vertextype candidatev2 = candidatev1;
            while (candidatev2 == candidatev1) {
                candidatev2 = (vertextype)((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT);
            }
            if ((candidatev2 >= gptr->dim) || (candidatev1 >= gptr->dim))
                std::cout << "ERROR in exceeding dim\n";
            if (!visited[candidatev2]) {
                visitedcnt++;
                visited[candidatev2] = true;
            }
            if (!(gptr->adjacencymatrix[(candidatev1*gptr->dim) + candidatev2])) {
                gptr->adjacencymatrix[candidatev1*(gptr->dim) + candidatev2] = true;
                gptr->adjacencymatrix[candidatev2*(gptr->dim) + candidatev1] = true;
            }
            lastv1 = candidatev2;
        }
        if (dist10000(rng) > RANDOMRANGEFLOAT/2.0) {
            vertextype lastv2 = (vertextype((float)dist10000(rng) * (float)(gptr->dim)/RANDOMRANGEFLOAT));
            gptr->adjacencymatrix[lastv1*gptr->dim + lastv2] = true;
            gptr->adjacencymatrix[lastv2*gptr->dim + lastv1] = true;
        }
    }

};

int samplematchingrandomgraphs( abstractparameterizedrandomgraph* rg, const int dim, const float edgecnt, const int outof );

std::vector<graphtype*> randomgraphs( abstractparameterizedrandomgraph* rg, const int dim, const float edgecnt, const int cnt );
/*

void randomgraph( graphtype*  gptr, const float edgecnt ); // legacy replaced as above by class

void randomconnectedgraphfixededgecnt( graphtype*  gptr, const int edgecnt ); // legacy replaced as above by class

void randomconnectedgraph( graphtype*  gptr );  // legacy replaced as above by class
*/

template<typename T>
class legacyrandomgraph : public abstractparameterizedrandomgraph {
protected:
    T* legacyrandomgraphptr {};
public:
    std::string shortname() {return legacyrandomgraphptr->shortname();}

    void randomgraph(graphtype *gptr) override {
        legacyrandomgraphptr->randomgraph(gptr,stoi(ps[1]));
    }
    std::vector<graphtype*> randomgraphs(const int cnt) override {
        int dim;
        float edgecnt;
        if (ps.size()>0)
            dim = stoi(ps[0]);
        if (ps.size()>1)
            edgecnt = stof(ps[1]);
        return legacyrandomgraphptr->randomgraphs(dim,edgecnt,cnt);
    }

    legacyrandomgraph() : legacyrandomgraphptr{new T}, abstractparameterizedrandomgraph("legacy random graph") {
    }
    ~legacyrandomgraph() {
        delete legacyrandomgraphptr;
    }
};


#endif //PROB_H
