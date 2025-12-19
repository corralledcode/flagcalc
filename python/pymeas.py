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

# def pyfindnormalspanningtree( adjmatrix, dim, rootvertex ):



def pyTdownclosure( root, tree, vertex ):

    def recurse( root, tree, u ):
        if u == root:
            return []
        for e in tree:
            if e[0] == u or e[1] == u:
                if e[0] == u:
                    w = e[1]
                else:
                    w = e[0]
                newtree = tree.copy()
                newtree.remove(e)
                # newtree2 = newtree.copy()
                if w != root:
                    set = recurse( root, newtree, w )
                    if root in set:
                        return set + [w]
                    set = recurse( root, newtree, u )
                    if root in set:
                        return set + [u]
                else:
                    return [w]
        return []

    return recurse( root, tree, vertex ) + [vertex]


def pyfindspanningtree( adjmatrix, dim, Es ):

    def mergecomponents( components, idxa, idxb ):
        components[idxa] = components[idxa] + components[idxb]
        components.pop(idxb)
        return

    def lookupvertex( components, v ):
        for i in range(len(components)):
            if v in components[i]:
                return i
        return -1

    components = []
    for v in range(dim):
        components.append([v])
    newEs = []
    i = 0
    overallchanged = True
    while overallchanged:
        overallchanged = False
        changed = True
        while changed and i < len(components):
            changed = False
            c = components[i]
            for e in Es:
                if e[0] in c:
                    if e[1] in c:
                        return []
                    j = lookupvertex( components, e[1] )
                    mergecomponents(components,i,j)
                    if i > j:
                        i -= 1
                    newEs.append(e)
                    Es.remove(e)
                    changed = True
                    break
                if e[1] in c:
                    if e[0] in c:
                        return []
                    j = lookupvertex( components, e[0] )
                    mergecomponents(components,i,j)
                    if i > j:
                        i -= 1
                    newEs.append(e)
                    Es.remove(e)
                    changed = True
                    break
            if not changed:
                i += 1
                changed = True
            else:
                overallchanged = True
    if len(Es) > 0:
        print ("Error: Es should be empty:" + str(Es))
        return []
    changed = True
    while changed and len(components) > 1:
        c = components[0]
        changed = False
        i = 0
        while i < len(c) and not changed:
            for v in range(dim):
                if adjmatrix[v][c[i]] and v not in c:
                    j = lookupvertex(components, v)
                    newEs.append([c[i], v])
                    mergecomponents(components,0,j)
                    changed = True
                    break
            i += 1

    if len(components) == 1:
        return newEs
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

def pyordervertices( Neighborslist, degrees, dim, startvertex ):
    list = [startvertex]
    i = 0
    changed = True
    while len(list) < dim and changed:
        changed = False
        for j in list:
            for k in range(degrees[j]):
                if Neighborslist[j][k] not in list:
                    list.append(Neighborslist[j][k])
                    changed = True
    if len(list) < dim:
        return []
    return list




testgraph = [[0,1,1,1,1],[1,0,1,1,1],[1,1,0,1,1],[1,1,1,0,1],[1,1,1,1,0]]
testgraphdim = 5
# print(pyfindspanningtree(testgraph,testgraphdim,[[0,1]]))
# print(pyfindspanningtree(testgraph,testgraphdim,[[0,1],[2,3]]))
# print(pyfindspanningtree(testgraph,testgraphdim,[[0,1],[2,3],[1,2],[0,3]]))
# print(pyfindspanningtree(testgraph,testgraphdim,[]))

# pyTestreturnset( [[0,1,1],[1,0,1],[1,1,0]], 3, 3, 3)

# print (pyedgesetcontainscycle(6,[[0,1],[1,2],[3,4],[4,5]]))

# print (pyedgesetcontainscycle(8,[[0, 2], [0, 4], [0, 5], [0, 6], [0, 7], [1, 2], [1, 3], [1, 7], [2, 5], [2, 7], [3, 4], [3, 7], [4, 6], [5, 6], [5, 7]]))