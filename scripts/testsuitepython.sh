# Inaugurating the Python integration with some simple Python implementations of measures
PTH=${PTH:-'../bin'}

$PTH/flagcalc -r 10000 24997500 1 -a ipy="pymeas" s="Deltam == pyDeltat" all -v i=minimal3.cfg

$PTH/flagcalc -r 10000 24997500 1 -a z="Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 10000 24997500 1 -a ipy="pymeas" z="pyDeltat" all -v i=minimal3.cfg

#... use j=1 to specify one thread (necessary for Python's GIL)
$PTH/flagcalc -r 5000 6248750 5 -a j=1 z="Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 5000 6248750 5 -a j=1 ipy="pymeas" z="pyDeltat" all -v i=minimal3.cfg

# The above has revealed difficulty (a bug) in flagcalc reading in large (1000 vertices) graphs from a file
# after all flagcalc was mostly used with large numbers of smaller graphs
$PTH/flagcalc -r 100 2475 1 -g o=out.dat overwrite -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a z="Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a ipy="pymeas" z="pyDeltat" all -v i=minimal3.cfg

$PTH/flagcalc -r 25 150 100 -g o=out.dat overwrite -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a j=1 z="Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a j=1 ipy="pymeas" z="pyDeltat" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abc defg" -a ipy="pymeas" p="pyTestreturnset(7,7)" all -v i=minimal3.cfg allsets
$PTH/flagcalc -r 25 150 100 -a j=1 ipy="pymeas" s="pyTestreturnset(dimm,dimm) == TUPLE (u IN V, TUPLE (v IN V, ac(u,v)))" all -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abcd efg de ghij" -a ipy="pymeas" p="pyfindspanningtree({{0,1},{2,3}})"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -r 8 14 25 -a j=1 ipy="pymeas" s="FORALL (S IN Ps(E), st(pyfindspanningtree(S))>0) IFF treec"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -r 6 7.5 30 -a j=1 ipy="pymeas" s1="conn1c" s2="FORALL (S IN Ps(E), st(pyfindspanningtree(S))>0 IMPLIES FORALL (e1 IN S, e2 IN S, e3 IN S, e1 != e2 AND e1 != e3 AND e2 != e3, st(e1 CUP e2 CUP e3) > 3))"  all -v i=minimal3.cfg allsets
