PTH=${PTH:-'../bin'}

# should be all TRUE
$PTH/flagcalc -r 10 p=0.5 10000 -a s="SUM (n IN NN(dimm+1), st(ncyclet(n))) == cyclest" -v i=minimal3.cfg
$PTH/flagcalc -r 10 p=0.5 10000 -a s="TALLY (n IN NN(dimm+1), st(ncyclet(n))) == cyclest" -v i=minimal3.cfg

# should be all TRUE (similar query is in testsuitesettheory.sh)
$PTH/flagcalc -r 10 p=0.5 10000 -a s="SUM (v IN V, SUM (C IN Cyclesvs(v), 1/st(C))) == cyclest" -v i=minimal3.cfg


$PTH/flagcalc -r 8 p=0.5 500 -a s="NAMING (Cs AS Cycless, FORALL (C IN Cs, FORALL (n IN st(C), n > 0, NOT (TUPLE (i IN st(C), C[(i+n) % st(C)]) ELT Cs) AND NOT (TUPLE (i IN st(C), C[(0 - (i+n)) % st(C)]) ELT Cs))))" -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 500 -a z="st(BIGCUPD (U IN Ps(V), st(U) > 2, SETD (P IN Perms(U), FORALL (i IN st(U), ac(P[i],P[(i+1)%st(U)])), P))) == TALLY (C IN Cycless, st(C) > 2, 2 * st(C))" -v i=minimal3.cfg

