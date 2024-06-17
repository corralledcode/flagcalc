//
// Created by peterglenn on 6/10/24.
//

#ifndef PROB_H
#define PROB_H
#include <iostream>
#include <random>
#include <iostream>

#include "graphs.h"

#define RANDOMRANGE 10000
#define RANDOMRANGEFLOAT 10000.0

using weightstype = std::vector<float>;

class abstractrandomgraph {
public:
    virtual std::string shortname() {return "";};
    std::string name;
    virtual void randomgraph( graph* gptr, const float edgecnt ) {};
    abstractrandomgraph() {};
    ~abstractrandomgraph() {};
};


class stdrandomgraph : public abstractrandomgraph {
    float _edgecnt;
public:
    std::string shortname() {return "r1";};
    stdrandomgraph() : abstractrandomgraph() {
        //_edgecnt = edgecnt;
        //name = "random graph with edgecnt probability " + std::to_string(_edgecnt);
        name = "random graph with given edgecnt";
    }
    void randomgraph( graph* gptr, float edgecnt ) {
        name = "random graph with edgecnt probability " + std::to_string(_edgecnt);
        _edgecnt = edgecnt;
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_int_distribution<std::mt19937::result_type> dist10000(0,RANDOMRANGE-1);
        for (int n = 0; n < gptr->dim; ++n) {
            gptr->adjacencymatrix[n*gptr->dim + n] = 0;
            for (int i = n+1; i < gptr->dim; ++i) {
                gptr->adjacencymatrix[n*gptr->dim + i] = (dist10000(rng) < (RANDOMRANGE*float(_edgecnt)/(gptr->dim * (gptr->dim-1)/2.0)));
                gptr->adjacencymatrix[i*gptr->dim + n] = gptr->adjacencymatrix[n*gptr->dim+i];
            }
        }
    }
};

class randomgraphonnedges : public abstractrandomgraph {
    int _edgecnt;
public:
    std::string shortname() {return "r2";}
    randomgraphonnedges() : abstractrandomgraph() {
        name = "random graph with given edgecnt";
    }
    void randomgraph( graph* gptr, float edgecnt ) {
        _edgecnt = (int)edgecnt;
        name = "random graph with edgecnt == " + std::to_string(edgecnt);
        if (_edgecnt > (gptr->dim * gptr->dim / 2)) {
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

        while (n < _edgecnt) {
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


class randomconnectedgraphfixededgecnt : public abstractrandomgraph {
    int _edgecnt;

public:
    std::string shortname() {return "r3";}
    randomconnectedgraphfixededgecnt() : abstractrandomgraph() {
        name = "random connected graph with given fixed edge count (ignoring unconnected outliers)";
    }
    void randomgraph( graph* gptr, float edgecnt ) {
        _edgecnt = edgecnt;
        name = "random connected graph with fixed edge count " + std::to_string(_edgecnt) + " (ignoring unconnected outliers)";
        for (int i = 0; i < gptr->dim; ++i) {
            for (int j = 0; j < gptr->dim; ++j) {
                gptr->adjacencymatrix[gptr->dim*i + j] = false;
            }
        }
        if (_edgecnt == 0)
            return;
        if (_edgecnt >= (int)((gptr->dim*(gptr->dim-1)/2))) {
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
        int cnt = _edgecnt-1;
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

class randomconnectedgraph : public abstractrandomgraph {
    int _edgecnt;
public:
    std::string shortname() {return "r4";}
    randomconnectedgraph() : abstractrandomgraph() {
        name = "random connected graph (algorithm does not find all such graphs...) with given edgecnt";
    }
    void randomgraph( graph* gptr, float edgecnt ) {
        _edgecnt = (int)edgecnt;
        name = "random connected graph (algorithm does not find all such graphs...) edgecnt == " + std::to_string(_edgecnt);
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

class weightedrandomconnectedgraph : public abstractrandomgraph {
    std::vector<weightstype> _weights;
public:
    std::string shortname() {return "r5";}
    weightedrandomconnectedgraph() : abstractrandomgraph() {
        name = "random connected graph with balanced/weighted search";
    }
    void randomgraph( graph* gptr, float edgecnt ) {
        _weights = computeweights(gptr->dim);
        name = "random connected graph with balanced/weighted search";
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
            while (candidatev1index < visitedcnt-1 && tmprnd > RANDOMRANGEFLOAT*_weights[visitedcnt][candidatev1index+1])
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

int samplematchingrandomgraphs( abstractrandomgraph* rg, const int dim, const float edgecnt, const int outof );

std::vector<graph> randomgraphs( abstractrandomgraph* rg, const int dim, const float edgecnt, const int cnt );
/*

void randomgraph( graph* gptr, const float edgecnt ); // legacy replaced as above by class

void randomconnectedgraphfixededgecnt( graph* gptr, const int edgecnt ); // legacy replaced as above by class

void randomconnectedgraph( graph* gptr );  // legacy replaced as above by class
*/


#endif //PROB_H
