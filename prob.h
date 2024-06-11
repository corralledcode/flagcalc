//
// Created by peterglenn on 6/10/24.
//

#ifndef PROB_H
#define PROB_H
#include "graphs.h"

void randomgraph( graph* gptr, int edgecnt );

void randomconnectedgraphfixededgecnt( graph* gptr, const int edgecnt );

void randomconnectedgraph( graph* gptr );

#endif //PROB_H
