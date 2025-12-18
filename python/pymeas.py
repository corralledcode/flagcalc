# This is a first repository of Python methods for use with flagcalc
# (flagcalc is hosted at Github: https://github.com/corralledcode/flagcalc)
# The convention is that a minimum of two parameters are required:
# adjmatrix and dim: adjmatrix is indexed as in [row*dim + col]
# Moreover, the default return type is only double precision float
# there is no support for deliberately returning an integer
# or sets, tuples, graphs, strings, etc., at this time
# Moreover, since the GIL is in effect in Python 3,
# these only work when flagcalc is not threading over several graphs at
# one time. Moreover, the additional parameters in user-defined methods
# are either integers or double-precision floats
# In any case for now integers are only passed as integers if parameter
# count is less than three, and both parameters are integers (mtdiscrete)
# on the C++/flagcalc side of things.
# This is easy to extend; moreover another easily extended limit
# is that only up to ten (ten plus two) parameters are supported.

from operator import truediv

import numpy as np

def pyac( adjmatrix, dim, u, v ):
#    ui = int(u)
#    vi = int(v) This is no longer needed since the C++ flagcalc code
    # now checkes types of parameters before passing them
    # (but only up to two parameters, and only if both are integers;
    # otherwise it casts them as double precision floats)
    return adjmatrix[u][v]

def pytest( adjmatrix, dim ):
    for i in range(dim):
        for j in range(dim):
            print ("Python: received data:" + str(adjmatrix[i,j]))
#            print ("Python: received data:" + str(adjmatrix[i*dim + j]))
    return 1

def pyDeltat( adjmatrix, dim ):
    max = 0
    for i in range(dim):
        cnt = 0
        for j in range(dim):
            if adjmatrix[i,j]:
                cnt += 1
        if cnt > max:
            max = cnt
    return max

def pyTestreturnset( adjmatrix, dim, m, n ):
    r = np.zeros((dim,dim), dtype=bool)
    for i in range(m):
        for j in range(n):
            r[i][j] = adjmatrix[i][j]
    return r

def pytestacceptset( adjmatrix, dim, set ):
    return set

def pyfindspanningtree( adjmatrix, dim, Es ):
    visited = np.zeros(dim, dtype=bool)
    newEs = []
    if len(Es) == 0:
        root = 0
    else:
        root = Es[0][0]
    visited[root] = True
    more = True
    while more:
        more = False
        for n in range(dim):
            if not visited[n]:
                found = 0
                more = True
                for e in Es:
                    if found == 0:
                        if e[0] == n and visited[e[1]]:
                            newEs.append(e)
                            visited[n] = True
                            found += 1
                            Es.remove(e)
                        else:
                            if e[1] == n and visited[e[0]]:
                                newEs.append(e)
                                visited[n] = True
                                found += 1
                                Es.remove(e)
                            else:
                                if e[0] == n or e[1] == n:
                                    more = True
                if found == 0:
                    for v in range(dim):
                        if (visited[v]):
                            if (found == 0) and pyac( adjmatrix, dim, n, v):
                                visited[n] = True
                                newEs.append([v,n])
                                found += 1
                else:
                    if found > 1:
                        print("pyfindspanningtree: cycle found")
                        return []
        if found == 0:
            return []
    all = True
    for n in range(dim):
        all = all and visited[n]
    if all and len(Es) == 0:
        return newEs
    else:
        return []

def pathdoescycle( E, visited ):
    for e in E:
        if visited[e[0]] and visited[e[1]]:
            return True
        if visited[e[0]] or visited[e[1]]:
            visited[e[0]] = True
            visited[e[1]] = True
            E.remove(e)
            return pathdoescycle( E, visited )
    if len(E) == 0:
        return False
    visited[E[0][0]] = True
    return pathdoescycle( E, visited )

def pyedgesetcontainscycle( dim, E ):
    visited = np.zeros(dim, dtype=bool)
    if len(E) == 0:
        return False
    visited[E[0][0]] = True
    return pathdoescycle( E, visited )



def pytestEdgesparameter( Edges, Nonedges ):
    print (Edges)
    print (Nonedges)
    return Edges

def pytestNonedgesparameter( Nonedgeslist ):
    return Nonedgeslist

def pytestNeighborslistparameter( Neighborslist, Nonneighborslist, degrees ):
    print (Neighborslist)
    print (Nonneighborslist)
    print (degrees)
    return Neighborslist

testgraph = [[0,1,1,1,1],[1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]
testgraphdim = 5
# print(pyfindspanningtree(testgraph,testgraphdim,[[0,1],[2,3]]))

pyTestreturnset( [[0,1,1],[1,0,1],[1,1,0]], 3, 3, 3)

# print (pyedgesetcontainscycle(6,[[0,1],[1,2],[3,4],[4,5]]))

# print (pyedgesetcontainscycle(8,[[0, 2], [0, 4], [0, 5], [0, 6], [0, 7], [1, 2], [1, 3], [1, 7], [2, 5], [2, 7], [3, 4], [3, 7], [4, 6], [5, 6], [5, 7]]))