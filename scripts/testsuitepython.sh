# Inaugurating the Python integration with some simple Python implementations of measures
PTH=${PTH:-'../bin'}

$PTH/flagcalc -r 10000 24997500 1 -a ipy="pymeas" s="Deltam == pyDeltat" all -v i=minimal3.cfg

$PTH/flagcalc -r 10000 24997500 1 -a z="Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 10000 24997500 1 -a ipy="pymeas" z="pyDeltat" all -v i=minimal3.cfg

#... use j=1 to specify one thread (necessary for Python's GIL)
$PTH/flagcalc -r 5000 6248750 5 -a j=1 z="Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 5000 6248750 5 -a j=1 ipy="pymeas" z="pyDeltat" all -v i=minimal3.cfg

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

$PTH/flagcalc -r 15 p=0.1 500 -a j=1 ipy="pymeas" s="pyedgesetcontainscycle(E) IFF NOT forestc"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -r 11 p=0.5 25 -a j=1 ipy="pymeas" s="pyedgesetcontainscycle(E) IFF NOT FORALL (v IN V, cyclesvt(v) == 0)"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -r 8 p=0.3 75 -a j=1 ipy="pymeas" s="FORALL (S IN Ps(E), pyedgesetcontainscycle(S) IMPLIES st(pyfindspanningtree(S)) == 0)"  all -v i=minimal3.cfg allsets

# Find some graphs and acyclic edge sets that cannot be extended to a spanning tree
$PTH/flagcalc -r 5 p=0.3 75 -a j=1 ipy="pymeas" s1=conn1c s2="EXISTS (S IN Ps(E), NOT pyedgesetcontainscycle(S) AND st(pyfindspanningtree(S)) == 0)" all -g o=out.dat overwrite passed -v i=minimal3.cfg allsets
$PTH/flagcalc -d out.dat -a j=1 ipy="pymeas" e="SETD (S IN Ps(E), NOT pyedgesetcontainscycle(S) AND st(pyfindspanningtree(S)) == 0, S)" all -v i=minimal3.cfg allsets

$PTH/flagcalc -r 8 p=0.3 75 -a j=1 ipy="pymeas" s1=conn1c s2="FORALL (S IN Ps(E), pyedgesetcontainscycle(S) IFF st(pyfindspanningtree(S)) == 0)"  all -v i=minimal3.cfg allsets

# Very important to note in the following: use "E" on the left of the equality to cast the tuple of tuples returned by pyfindspanningtree into sets,
# or do as in the second line
$PTH/flagcalc -r 15 p=0.2 250 -a j=1 ipy="pymeas" s="edgecm > 0" s2="E == pyfindspanningtree(Nulls) IFF treec"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -r 16 p=0.3 400 -a j=1 ipy="pymeas" s="edgecm > 0" s2="SETD (e IN pyfindspanningtree(Nulls), TupletoSet(e)) == E IFF treec"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -r 8 p=0.4 25 -a j=1 ipy="pymeas" s="FORALL (S IN Ps(E), NAMING (T AS pyfindspanningtree(S), st(T) == 0 OR st(T) + 1 == dimm))"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -r 20 p=0.2 250 -a j=1 ipy="pymeas" s1="NOT conn1c" s2="pyfindspanningtree({}) == Nulls"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -r 20 p=0.2 2500 -a j=1 ipy="pymeas" s="BIGCUP (e IN pyfindspanningtree({}), e) == V OR NOT conn1c"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -r 20 p=0.2 2500 -a j=1 ipy="pymeas" s="BIGCUPD (e IN pyfindspanningtree({}), e) >= V OR NOT conn1c"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -r 10 p=0.5 10 -a j=1 ipy=pymeas p="pyordervertices(0)" all -v crit allsets

$PTH/flagcalc -r 14 p=0.5 1000 -a j=1 ipy=pymeas s1=conn1c s2="NAMING (O AS pyordervertices(0), FORALL (i IN st(O), conn1c(SubgraphonUg(Sp(O,i)))))" all -v i=minimal3.cfg

# borrowed from testsuitesettheory.sh: Diestel Cor 1.5.2
$PTH/flagcalc -r 9 p=0.2 100 -a s="treec" s2="EXISTS (P IN Perms(V), FORALL (v IN V, P[v] >= 1, EXISTS (n IN NN(dimm), P[n] < P[v] AND ac(n,v), FORALL (m IN NN(dimm), (P[m] < P[v] AND ac(m,v)) IMPLIES m == n))))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 18 p=0.1 10000 -a j=1 ipy=pymeas s="treec" s2="NAMING (P AS pyordervertices(0), FORALL (i IN NN(dimm), i >= 1, EXISTS (j IN NN(dimm), j < i AND ac(P[i],P[j]), FORALL (k IN NN(dimm), k < i AND ac(P[i],P[k])) IMPLIES j == k))))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -r 10 p=0.25 1000 -a j=1 ipy=pymeas isp="../scripts/storedprocedures.dat" s="treec" all -g o=out.dat overwrite passed -v set allsets i=minimal3.cfg
$PTH/flagcalc -d out.dat -a j=1 ipy=pymeas isp="../scripts/storedprocedures.dat" s="treec" e2="SETD (v IN V, pyTdownclosure(0,treefromorderedvertices(pyordervertices(0)),v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d out.dat -a j=1 ipy=pymeas isp="../scripts/storedprocedures.dat" s="treec" s2="TnormalinG(treefromorderedvertices(pyordervertices(0)))" all -v set allsets i=minimal3.cfg

# Diestel p. 15 "T is normal in G" for a tree T = G with any vertex as its root
$PTH/flagcalc -r 12 p=0.15 10000 -a j=1 ipy=pymeas isp="../scripts/storedprocedures.dat" s="treec" s2="NAMING (P AS pyordervertices(0), TnormalinG( treefromorderedvertices(P) ))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 p=0.15 10000 -a j=1 ipy=pymeas isp="../scripts/storedprocedures.dat" s="treec" s2="FORALL (root IN V, NAMING (P AS pyordervertices(root), TnormalinG(treefromorderedvertices(P)) ))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d out.dat -a j=1 ipy=pymeas isp="../scripts/storedprocedures.dat" s="treec" e2="NAMING (T AS treefromorderedvertices(pyordervertices(0)), NAMING (H AS BIGCUP( e IN T, e ), SETD (v IN H, pyTdownclosure(0,T,v))))" all -v set allsets i=minimal3.cfg

# Diestel p. 15 "T is normal in G" for a spanning tree T contained in G: the algorithm does not (yet) seek normalcy
$PTH/flagcalc -r 10 p=0.25 1000 -a j=1 ipy=pymeas isp="../scripts/storedprocedures.dat" s="conn1c" s2="TnormalinG(pyfindspanningtree({}))" all -v set allsets i=minimal3.cfg

