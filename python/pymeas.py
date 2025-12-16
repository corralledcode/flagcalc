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
    # print ("Hi " + str(u) + ", " + str(v))
    # print("Python: received data:" + str(adjmatrix[u * dim + v]))
    # return 1
#    ui = int(u)
#    vi = int(v) This is no longer needed since the C++ flagcalc code
    # now checkes types of parameters before passing them
    # (but only up to two parameters, and only if both are integers;
    # otherwise it casts them as double precision floats)
#    return adjmatrix[ui*dim + vi]
    return adjmatrix[u*dim + v]

def pytest( adjmatrix, dim ):
    for i in range(dim):
        for j in range(dim):
            print ("Python: received data:" + str(adjmatrix[i*dim + j]))
    return 1

def pyDeltat( adjmatrix, dim ):
    max = 0
    for i in range(dim):
        cnt = 0
        for j in range(dim):
            if adjmatrix[i*dim + j]:
                cnt += 1
        if cnt > max:
            max = cnt
    return max

