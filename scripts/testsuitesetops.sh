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





