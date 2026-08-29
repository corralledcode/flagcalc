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
$PTH/flagcalc -r 6 p=0.5 500 -a isp=graphlinearalgebra.dat s="FORALL (n IN dimm, n > 2, FORALL (v IN V, NAMING (Cv AS Cyclesvs(v), FORALL (nW IN nWalksvs(n,v,v), FORALL (i IN n, j IN n, i < j, nW[i] != nW[j]) IMPLIES \
(Reversep(Subp(nW,1,n+1)) ELT Cv OR Sp(nW,n) ELT Cv)))))" -v i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.5 500 -a s="FORALL (n IN dimm, n > 2, FORALL (v IN V, NAMING (Cv AS Cyclesvs(v), FORALL (nW IN nWalksvs(n,v,v), FORALL (i IN n, j IN n, i < j, nW[i] != nW[j]) IMPLIES \
(Reversep(Subp(nW,1,n+1)) ELT Cv OR Sp(nW,n) ELT Cv)))))" -v i=minimal3.cfg

# Diversion to look at new graph-valued measures Kg, Ig, Cg, pPartiteg
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (n IN dimm, embedsc(Kg(n)) IFF Knc(n,1))" -v i=minimal3.cfg
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (n IN dimm, embedsc(Ig(n)) IFF EXISTS (S IN Sizedsubset(V,n), FORALL (u IN S, v IN S, u < v, NOT ac(u,v))))" -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="FORALL (n IN dimm, n > 2, embedsgenerousc(Cg(n)) IFF EXISTS (C IN Cycless, st(C) == n))" -v i=minimal3.cfg
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (n IN dimm, n > 2, embedsc(Cg(n)) IMPLIES circm >= n)" -v i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.5 1 -a e="SETD (S IN Setpartition(V), pPartiteg(TUPLE (s IN S, st(s))))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 8 p=0.5 100 -a isp=storedprocedures.dat isp=graphlinearalgebra.dat s="EXISTS (n IN dimm, n > 0 AND n+1 < dimm, pPartitec(<<n,dimm-n>>)) IFF Bipartiteviacycles" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 10 p=0.5 100 -a isp=storedprocedures.dat isp=graphlinearalgebra.dat s="FORALL (n IN dimm, n > 2 AND st(n) % 2 == 1, NOT embedsgenerousc(Cg(n))) IFF Bipartiteviacycles" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d graph6.fcg -a isp=storedprocedures.dat isp=graphlinearalgebra.dat s="FORALL (P IN Setpartition(V), NAMING (PS AS TUPLE (p IN P, st(p)), embedsc(pPartiteg(PS)) IMPLIES pPartitec(PS)))" -v i=minimal3.cfg allsets set

# testing new feature of passing a graph as first parameter to a stored procedure
$PTH/flagcalc -r 6 0 1 -a isp=storedprocedures.dat isp=graphlinearalgebra.dat s="FORALL (P IN Setpartition(V), NAMING (PS AS TUPLE (p IN P, st(p)), NAMING (pPg AS pPartiteg(PS), pPartitec(pPg,PS))))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 15 0 1 -a isp=graphlinearalgebra.dat s="FORALL (n IN dimm, n > 2, Cc(Cg(n)))" -v i=minimal3.cfg allsets set

# "IF" works, "IFF" works as in the second line below
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (C IN Cycless, embedsc(Cg(st(C))) IF FORALL (i IN st(C), j IN st(C), i < j, ac(C[i],C[j]) IMPLIES j - i == 1 OR (j == st(C) - 1 AND i == 0)  ))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 10 p=0.5 100 -a s="FORALL (C IN Cycless, embedsc(SubgraphonUg(C),Cg(st(C))) IFF FORALL (i IN st(C), j IN st(C), i < j, ac(C[i],C[j]) IMPLIES j - i == 1 OR (j == st(C) - 1 AND i == 0)  ))" -v i=minimal3.cfg allsets set

# Now investigating linear algebra

$PTH/flagcalc -d graph4.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 4" s2="FORALL (F IN Ps(Ps(E)), dimedgesubspace(F) + dimedgesubspace(edgesubspaceortho(F)) == edgecm)" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph6.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 5" s2="FORALL (F IN Sizedsubset(Ps(E),4), dimedgesubspace(F) + dimedgesubspace(edgesubspaceortho(F)) == edgecm)" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 6 7.5 5 -a isp=graphlinearalgebra.dat s="dimedgesubspace(edgestandardbasis) == edgecm" -v i=minimal3.cfg allsets set
$PTH/flagcalc -r 6 7.5 5 -a isp=graphlinearalgebra.dat e="edgestandardbasis" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d f="abc d e f" -a isp=graphlinearalgebra.dat s="FORALL (F IN Ps(Ps(E)), dimedgesubspace(F) + dimedgesubspace(edgesubspaceortho(F)) == edgecm)" z="dimedgesubspace({{},{{0,2}},{{1,2}},{{0,2},{1,2}}})" e="edgesubspaceortho({{},{{0,2}},{{1,2}},{{0,2},{1,2}}})" -v i=minimal3.cfg allsets set alltally

$PTH/flagcalc -d graph4.fcg -a isp=graphlinearalgebra.dat s="cyclomaticnumber == edgecm - dimm + st(Connc)" z="edgecm - dimm + st(Connc)" -v i=minimal3.cfg allsets set

# Diestel Prop 1.9.1
$PTH/flagcalc -d graph4.fcg -a isp=graphlinearalgebra.dat s="NAMING (cs AS cyclespace, FORALL (D IN Ps(E), FORALL (v IN V, COUNT (d IN D, v ELT d) % 2 == 0) IFF D ELT cs))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph5.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 8" s2="NAMING (cs AS cyclespace, FORALL (D IN Ps(E), FORALL (v IN V, COUNT (d IN D, v ELT d) % 2 == 0) IFF D ELT cs))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph6.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 9" s2="NAMING (cs AS cyclespace, FORALL (D IN Ps(E), FORALL (v IN V, COUNT (d IN D, v ELT d) % 2 == 0) IFF D ELT cs))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph6.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 9" s2="NAMING (cs AS cyclespace, FORALL (D IN Ps(E), NAMING (BD AS BIGCUPD (e IN D, e), FORALL (v IN V, COUNT (u IN BD, u == v) % 2 == 0)) IFF D ELT cs))" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d graph6.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 9" s2="NAMING (cs AS cyclespace, FORALL (D IN Ps(E), NAMING (BD AS BIGCUPD (e IN D, e), FORALL (vd IN Tallyp(BD,dimm), vd % 2 == 0)) IFF D ELT cs))" -v i=minimal3.cfg allsets set

# Check new tuple-valued measures "Tallyp" and "Flattenp"
$PTH/flagcalc -d graph6.fcg -a s="Tallyp( BIGCUPD (e IN E, e), dimm ) == TUPLE (v IN V, vdt(v))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph6.fcg -a s="Flattenp (Tallyp( BIGCUPD (e IN E, e), dimm ), FALSE ) == TUPLE (v IN V, vdt(v) > 0, v)" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph6.fcg -a s="Flattenp (Tallyp( BIGCUPD (e IN E, e), dimm ), TRUE ) == BIGCUPD (v IN V, TUPLE (n IN vdt(v), v))" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d graph5.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 8" s2="NAMING (cs AS cyclespace, FORALL (D IN Ps(E), FORALL (v IN V, COUNT (d IN D, v ELT d) % 2 == 0) IFF D ELT cs))" -v i=minimal3.cfg allsets set

# Diestel Prop 1.9.2
$PTH/flagcalc -d graph4.fcg -a isp=graphlinearalgebra.dat s="NAMING (cs AS cutspace, edgesetspan(atomiccuts) == cs)" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph5.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 8" s2="NAMING (cs AS cutspace, edgesetspan(atomiccuts) == cs)" -v i=minimal3.cfg allsets set

# Diestel Thm 1.9.4
$PTH/flagcalc -d graph4.fcg -a isp=graphlinearalgebra.dat s="NAMING (cuts AS cutspace, NAMING (cycles AS cyclespace, cuts == edgesubspaceortho(cycles)))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph5.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 8" s2="NAMING (cuts AS cutspace, NAMING (cycles AS cyclespace, cuts == edgesubspaceortho(cycles)))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph5.fcg -a isp=graphlinearalgebra.dat s1="edgecm < 8" s2="NAMING (cuts AS cutspace, NAMING (cycles AS cyclespace, edgesubspaceortho(cuts) == cycles))" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d graph5.fcg -a ipy=pymeas isp=graphlinearalgebra.dat s1="edgecm < 8" s2="conn1c" s3="NAMING (T AS pyfindspanningtree(Nulls), NAMING (chords AS E SETMINUS T, edgesetspan(SETD (c IN chords, fundamentalcycle(T,c))) == cyclespace))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph5.fcg -a ipy=pymeas isp=graphlinearalgebra.dat s1="edgecm < 8" s2="conn1c" s3="NAMING (T AS pyfindspanningtree(Nulls), edgesetspan(SETD (e IN T, fundamentalcut(T,e))) == cutspace)" -v i=minimal3.cfg allsets set

# Verify chordal graphs
$PTH/flagcalc -d chordal4.fcg -a s="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d chordal5.fcg -a s="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d chordal6.fcg -a s="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph6c.fcg -a s="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d chordal7.fcg -a s="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" -v i=minimal3.cfg allsets set
$PTH/flagcalc -d graph7c.fcg -a s="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d chordal4.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg -c -d graph4c.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg allsets set
$PTH/flagcalc -d chordal5.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg -c -d graph5c.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg allsets set
$PTH/flagcalc -d chordal6.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg -c -d graph6c.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg allsets set
$PTH/flagcalc -d chordal7.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg -c -d graph7c.fcg -a isp=storedprocedures.dat s=chordal -v i=minimal3.cfg allsets set

$PTH/flagcalc -d graph5.fcg -a s="FORALL (n IN st(V), SETD (P IN Setpartition(V), st(P) == n+1, P) == NAMING (M AS Maps(n+1,st(V)), SETD (m IN M, SETD (i IN n+1, m[i]))))" -v i=minimal3.cfg allsets set

$PTH/flagcalc -d f="a b c d e f g h" -a s="Setnpartitions(V,5) == SETD (P IN Setpartition(V), st(P) == 5, P)" -v i=minimal3.cfg

$PTH/flagcalc -d f="a b c d e f g h i j" -a e="Setnpartitions(V,2)" -a e="SETD (P IN Setpartition(V), st(P) == 2, P)" -v i=minimal3.cfg
