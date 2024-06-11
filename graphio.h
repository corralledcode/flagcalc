//
// Created by peterglenn on 6/10/24.
//

#ifndef GRAPHIO_H
#define GRAPHIO_H
#include <iostream>
#include <fstream>

#include "graphs.h"


class graphio {
    int id = -1;
    virtual graph readgraph();
    virtual int writegraph(graph g);
    virtual int newgraphid();
};

class graphsqlio : graphio {
    graph readgraph();
    int writegraph(graph g);
    int newgraphid();
};

class graphstreamio : graphio {
    std::ifstream ifs;
    std::istream* s = &std::cin;
    graph readgraph();
    int writegraph(graph g);
};

inline graph graphio::readgraph() {

}

inline int graphio::writegraph(graph g) {

}

int graphio::newgraphid() {

}


inline graph graphsqlio::readgraph() {

}

inline int graphsqlio::newgraphid() {

}

inline int graphsqlio::writegraph(graph g) {

}


inline graph graphstreamio::readgraph() {

}

inline int graphstreamio::writegraph(graph g) {

}


#endif //GRAPHIO_H
