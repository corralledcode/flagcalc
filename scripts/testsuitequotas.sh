# Inaugurating quantifiers ANY, ANYN, EXISTSUNIQUE, EXISTSNUNIQUE
PTH=${PTH:-'../bin'}

$PTH/flagcalc -r 5000 6248750 2 -a e="ANYN (3, e IN E, e)"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abcde" -a s="EXISTSN (3, v IN V, v < 2)"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="abcde" -a s="EXISTSN (3, v IN V, v < 3)"  all -v i=minimal3.cfg allsets

# FORALLN means For all BUT n-mamy
$PTH/flagcalc -d f="abcde" -a s="FORALLN (3, v IN V, v > 2)"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="abcde" -a s="FORALLN (3, v IN V, v > 3)"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abcde" -a s="EXISTSN (3, v IN V, v > 0, v < 3)"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="abcde" -a s="EXISTSN (3, v IN V, v > 0, v < 4)"  all -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abcde" -a s="FORALLN (3, v IN V, v > 0, v > 3)"  all -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="abcde" -a s="FORALLN (3, v IN V, v > 0, v > 4)"  all -v i=minimal3.cfg allsets

# Diestel Cor 1.5.2, duplicated in testsuitesettheory.sh without the EXISTSUNIQUE
$PTH/flagcalc -r 9 p=0.2 100 -a s="treec" s2="EXISTS (P IN Perms(V), FORALL (v IN V, P[v] >= 1, EXISTSUNIQUE (n IN NN(dimm), P[n] < P[v] AND ac(n,v))))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -r 6 p=0.2 100 -a z="EXISTSNUNIQUE( 16, P IN Perms(V), FORALLN (2, w IN V, P[w] == w))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -r 6 p=0.2 100 -a s="EXISTSNUNIQUE( 1, P IN Perms(V), FORALLN (1, w IN V, P[w] == w))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.2 100 -a s="EXISTSNUNIQUE( nchoosek(dimm,2)+1, P IN Perms(V), FORALLN (2, w IN V, P[w] == w))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.2 100 -a s="EXISTSNUNIQUE( nchoosek(dimm,3)*2 + nchoosek(dimm,2) + 1, P IN Perms(V), FORALLN (3, w IN V, P[w] == w))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.2 100 -a s="EXISTSNUNIQUE( nchoosek(dimm,4)*9 + nchoosek(dimm,3)*2 + nchoosek(dimm,2) + 1, P IN Perms(V), FORALLN (4, w IN V, P[w] == w))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.2 100 -a s="EXISTSNUNIQUE( nchoosek(dimm,5)*44 + nchoosek(dimm,4)*9 + nchoosek(dimm,3)*2 + nchoosek(dimm,2) + 1, P IN Perms(V), FORALLN (5, w IN V, P[w] == w))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.2 100 -a s="EXISTSNUNIQUE( nchoosek(dimm,6)*5*(44+9) + nchoosek(dimm,5)*4*(9+2) + nchoosek(dimm,4)*3*(2+1) + nchoosek(dimm,3)*2*(1+0) + nchoosek(dimm,2) + 1, P IN Perms(V), FORALLN (6, w IN V, P[w] == w))" all -v set allsets i=minimal3.cfg
