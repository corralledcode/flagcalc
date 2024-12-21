PTH='../cmake-build-debug'
$PTH/flagcalc -r 10 10 1 -a e="V CUP E" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a e="V CAP {2,3}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a e="V SETMINUS {2,4,8}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a e="V SETXOR {2,4,8,13}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 4 4 1 -a e="Ps(V)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 5 5 1 -a isp=storedprocedures.dat e="PairSet(V)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 5 5 1 -a s="FORALL (s IN Ps(V), Edgess(s) == SET (e IN E, e(0) ELT s AND e(1) ELT s, e))" all -v i=minimal3.cfg
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
$PTH/flagcalc -d f="abc" -a s="nwisec(E,\"MEET\",2,1)" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="nwisec(E,\"MEET\",2,1)" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="E DISJOINT {{3,4,6}}" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="E DISJOINT {{1,3}}" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="E DISJOINT {{3,1},{4}}" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 20 100 -a s="FORALL (a IN Setpartition(V), nwisec(a,\"DISJOINT\",2,1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 20 100 -a s="FORALL (a IN Setpartition(V), nwisec(a,\"DISJOINT\",2,(-3)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 20 100 -a s="FORALL (a IN Setpartition(V), st(a) < 3, nwisec(a,\"MEET\",3,(-3)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 20 100 -a s="FORALL (a IN Setpartition(V), st(a) < 3, nwisec(a,\"MEET\",3,(-3)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 20 100 -a s="FORALL (a IN Setpartition({0,1,2,3,4,5,6,7}), st(a) < 3, nwisec(a,\"MEET\",3,(-3)))" all -v i=minimal3.cfg







