PTH=${PTH:-'../bin'}

$PTH/flagcalc -r 10 10 1 -a e="V CUP E" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a e="V CAP {2,3}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a e="V SETMINUS {2,4,8}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a e="V SETMINUS {2,4,8,11,15}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a e="V SETXOR {2,4,8,13}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 4 4 1 -a e="Ps(V)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 5 5 1 -a isp=storedprocedures.dat e="PairSet(V)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 5 5 1 -a s="FORALL (s IN Ps(V), Edgess(s) == SET (e IN E, e[0] ELT s AND e[1] ELT s, e))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 1 -a s="FORALL (n IN dimm, FORALL (s IN Sizedsubset(V,n), st(s) == n))" all
$PTH/flagcalc -r 5 5 1 -a s="<<0,2,4>> <= <<0,2,4,5,1>>" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 1 -a s="<<0,2,4>> <= <<0,2,3,4,5,1>>" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 5 1 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), s == t OR st(s SETXOR t) > 0))" all
# Note: false below
$PTH/flagcalc -r 8 5 1 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), s CAP t == Nulls IMPLIES FORALL (v IN V, v ELT s IFF NOT v ELT t)))" all
# Note: true below
$PTH/flagcalc -r 8 5 1 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), (s CAP t == Nulls AND s CUP t == V) IMPLIES FORALL (v IN V, v ELT s XOR v ELT t)))" all
$PTH/flagcalc -r 8 5 1 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), s CAP t == Nulls IMPLIES st(s) + st(t) == st(s CUP t)))" all
$PTH/flagcalc -d f="abc" -a s="nwisec({{0,1},{0,2,3},{2,5,0},{1,2}},\"MEET\",2,1)" all -v i=minimal3.cfg
# Note: true below
$PTH/flagcalc -d f="abc" -a s="nwisec(E,\"MEET\",2,1)" all -v i=minimal3.cfg
# Note: false below
$PTH/flagcalc -d f="abcd" -a s="nwisec(E,\"MEET\",2,1)" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="E DISJOINT {{3,4,6}}" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="E DISJOINT {{1,3}}" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="E DISJOINT {{3,1},{4}}" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 20 100 -a s="FORALL (a IN Setpartition(V), nwisec(a,\"DISJOINT\",2,1))" all -v i=minimal3.cfg

# should be false
$PTH/flagcalc -r 6 20 100 -a s="FORALL (a IN Setpartition(V), nwisec(a,\"DISJOINT\",2,(-3)))" all -v i=minimal3.cfg

# True: (for all a sized below 3, they three-wise meet in all but 3 elements)
$PTH/flagcalc -r 6 20 100 -a s="FORALL (a IN Setpartition(V), st(a) < 3, nwisec(a,\"MEET\",3,(-3)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 20 100 -a s="FORALL (a IN Setpartition({0,1,2,3,4,5,6,7}), st(a) < 3, nwisec(a,\"MEET\",3,(-3)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 3 2 100 -a s="SET (p IN Ps(Ps(V)), nwisec(p,\"DISJOINT\",2,1) AND SUM (x IN p, st(x)) == dimm AND FORALL (x IN p, x != Nulls), p) <= Setpartition(V)" all -v i=minimal3.cfg
$PTH/flagcalc -r 3 2 100 -a s="SET (p IN Ps(Ps(V)), nwisec(p,\"DISJOINT\",2,1) AND SUM (x IN p, st(x)) == dimm AND FORALL (x IN p, x != Nulls), p) == Setpartition(V)" all -v i=minimal3.cfg
$PTH/flagcalc -r 3 2 100 -a s="SET (p IN Ps(Ps(V)), nwisec(p,\"DISJOINT\",2,1) AND BIGCUP (x IN p, x) == V AND FORALL (x IN p, x != Nulls), p) == Setpartition(V)" all -v i=minimal3.cfg
$PTH/flagcalc -r 3 2 100 -a s="SET (p IN Ps(Ps(V)), nwisec(p,\"DISJOINT\",2,1) AND st(BIGCUPD (x IN p, x)) == dimm AND FORALL (x IN p, x != Nulls), p) == Setpartition(V)" all -v i=minimal3.cfg
$PTH/flagcalc -r 3 2 100 -a s="Ps(Ps(V)) >= Setpartition(V)" all -v i=minimal3.cfg
$PTH/flagcalc -r 3 2 100 -a s="SET (p IN Ps(Ps(V)), p) >= Setpartition(V)" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="SET (ps IN Ps(Pathss(0,1)), nwisec(ps,\"DISJOINT\",2,3), ps)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="BIGCUP (v IN V, SET (p IN Cyclesvs(v), p))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="SET (p IN Pathss(0,1), p <= <<0,2,1,3,4,5>>, p)" all -v set allsets i=minimal3.cfg

# should be false
$PTH/flagcalc -d f="abcd ef -agha" -a s="FORALL (p IN Pathss(0,1), EXISTS (A IN Ps(V), (NOT 0 ELT A) AND p <= A))" all

# true
$PTH/flagcalc -d f="abcd ef -agha" -a s="FORALL (p IN Pathss(0,1), EXISTS (A IN Ps(V), p <= A))" all
$PTH/flagcalc -d f="abcd ef -agha" -a s="FORALL (p IN Pathss(0,1), EXISTS (A IN Ps(V), A <= p))" all
$PTH/flagcalc -r 8 14 100 -a z="SUM (A IN Ps(V), COUNT (B IN Ps(V), A <= B))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -a z="SUM (A IN Ps(V), COUNT (B IN Ps(V), A <= {0,1,2,3,4,5,6,7}))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 22.5 10 -a z="SUM (A IN Ps(V), st(A CUP {0,1,2,3}))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 22.5 10 -a z="SUM (A IN Ps(V), SUM (B IN Ps(V), st(A CUP B)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 22.5 10 -a z="SUM (A IN Ps(V), SUM (B IN Ps(V), st(A) + st(B)))" all -v i=minimal3.cfg

# not all true
$PTH/flagcalc -r 8 14 1000 -a s="FORALL (p IN Pathss(0,1), EXISTS (q IN Pathss(0,1), q != p AND (q SETMINUS {0,1}) MEET p))" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 14 10 -a s="EXISTS (p IN Setpartition(V), st(p) > 1, FORALL (v IN V, FORALL (u IN V, FORALL (q IN Pathss(u,v), EXISTS (r IN p, q <= r)))))" \
 s2="st(Componentss) > 1" all -v i=minimal3.cfg

 # not true
$PTH/flagcalc -r 8 14 10 -a s="EXISTS (p IN Setpartition(V), st(p) > 1, FORALL (v IN V, FORALL (u IN V, FORALL (q IN Pathss(u,v), EXISTS (r IN p, TupletoSet(q) <= r)))))" \
 s2="st(Componentss) > 1" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 10 -a s="FORALL (p IN Setpartition(V), st(p) == 2, FORALL (v IN V, FORALL (u IN V, FORALL (q IN Pathss(u,v), FORALL (r IN p, TupletoSet(q) <= r IFF q <= r)))))" all -v i=minimal3.cfg

# The claim below is that paths between any intermediate point between two vertices can be pasted together to recover the original paths
$PTH/flagcalc -r 6 7.5 10 -a s="FORALL (v IN V, FORALL (u IN V, u != v, EXISTS (w IN V, Pathss(u,v) >= BIGCUP (x IN Pathss(u,w), SET (y IN Pathss(w,v),  Sp(x,st(x)-1) CUP y)))))" all -v i=minimal3.cfg

# runtime 17 seconds on an i9 5/12/2025