PTH=${PTH:-'../bin'}

# should be all TRUE
$PTH/flagcalc -r 10 p=0.5 1000 -a s="SUM (n IN NN(dimm+1), st(ncyclet(n))) == cyclest" -v i=minimal3.cfg
$PTH/flagcalc -r 10 p=0.5 1000 -a s="TALLY (n IN NN(dimm+1), st(ncyclet(n))) == cyclest" -v i=minimal3.cfg

# should be all TRUE (similar query is in testsuitesettheory.sh)
$PTH/flagcalc -r 10 p=0.5 1000 -a s="SUM (v IN V, SUM (C IN Cyclesvs(v), 1/st(C))) == cyclest" -v i=minimal3.cfg


$PTH/flagcalc -r 7 p=0.5 500 -a s="NAMING (Cs AS Cycless, FORALL (C IN Cs, FORALL (n IN st(C), n > 0, NOT (TUPLE (i IN st(C), C[(i+n) % st(C)]) ELT Cs) AND NOT (TUPLE (i IN st(C), C[(0 - (i+n)) % st(C)]) ELT Cs))))" -v i=minimal3.cfg
$PTH/flagcalc -r 7 p=0.5 500 -a s="st(BIGCUPD (U IN Ps(V), st(U) > 2, SETD (P IN Perms(U), FORALL (i IN st(U), ac(P[i],P[(i+1)%st(U)])), P))) == TALLY (C IN Cycless, st(C) > 2, 2 * st(C))" -v i=minimal3.cfg
$PTH/flagcalc -r 7 p=0.5 500 -a s="TALLY (U IN Ps(V), st(U) > 2, COUNT (P IN Perms(U), FORALL (i IN st(U), ac(P[i],P[(i+1)%st(U)])))) == TALLY (C IN Cycless, st(C) > 2, 2 * st(C))" -v i=minimal3.cfg

# Three equiv queries, the second twice slower due to using a stored procedure, the third using native coded Reversep but slower than the first due to Subp usage
# All relate walks to paths/cycles
$PTH/flagcalc -r 6 p=0.5 500 -a s="FORALL (n IN dimm, n > 2, FORALL (v IN V, NAMING (Cv AS Cyclesvs(v), FORALL (nW IN nWalksvs(n,v,v), FORALL (i IN n, j IN n, i < j, nW[i] != nW[j]) IMPLIES \
(TUPLE (j IN n, nW[n-j]) ELT Cv OR Sp(nW,n) ELT Cv)))))" -v i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.5 500 -a isp=storedprocedures.dat s="FORALL (n IN dimm, n > 2, FORALL (v IN V, NAMING (Cv AS Cyclesvs(v), FORALL (nW IN nWalksvs(n,v,v), FORALL (i IN n, j IN n, i < j, nW[i] != nW[j]) IMPLIES \
(Reversep(Subp(nW,1,n+1)) ELT Cv OR Sp(nW,n) ELT Cv)))))" -v i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.5 500 -a s="FORALL (n IN dimm, n > 2, FORALL (v IN V, NAMING (Cv AS Cyclesvs(v), FORALL (nW IN nWalksvs(n,v,v), FORALL (i IN n, j IN n, i < j, nW[i] != nW[j]) IMPLIES \
(Reversep(Subp(nW,1,n+1)) ELT Cv OR Sp(nW,n) ELT Cv)))))" -v i=minimal3.cfg

# Diversion to look at new graph-valued measures Kg, Ig, Cg, pPartiteg
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (n IN dimm, embedsc(Kg(n)) IFF Knc(n,1))" -v i=minimal3.cfg
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (n IN dimm, embedsc(Ig(n)) IFF EXISTS (S IN Sizedsubset(V,n), FORALL (u IN S, v IN S, u < v, NOT ac(u,v))))" -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="FORALL (n IN dimm, n > 2, embedsgenerousc(Cg(n)) IFF EXISTS (C IN Cycless, st(C) == n))" -v i=minimal3.cfg
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (n IN dimm, n > 2, embedsc(Cg(n)) IMPLIES circm >= n)" -v i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.5 1 -a e="SETD (S IN Setpartition(V), pPartiteg(TUPLE (s IN S, st(s))))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 8 p=0.5 100 -a isp=storedprocedures.dat s="EXISTS (n IN dimm, n > 0 AND n+1 < dimm, pPartitec(<<n,dimm-n>>)) IFF Bipartiteviacycles" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 10 p=0.5 100 -a isp=storedprocedures.dat s="FORALL (n IN dimm, n > 2 AND st(n) % 2 == 1, NOT embedsgenerousc(Cg(n))) IFF Bipartiteviacycles" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d graph6.fcg -a isp=storedprocedures.dat s="FORALL (P IN Setpartition(V), NAMING (PS AS TUPLE (p IN P, st(p)), embedsc(pPartiteg(PS)) IMPLIES pPartitec(PS)))" -v i=minimal3.cfg allsets set

# testing new feature of passing a graph as first parameter to a stored procedure
$PTH/flagcalc -r 6 0 1 -a isp=storedprocedures.dat s="FORALL (P IN Setpartition(V), NAMING (PS AS TUPLE (p IN P, st(p)), NAMING (pPg AS pPartiteg(PS), pPartitec(pPg,PS))))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 15 0 1 -a isp=storedprocedures.dat s="FORALL (n IN dimm, n > 2, Cc(Cg(n)))" -v i=minimal3.cfg allsets set

