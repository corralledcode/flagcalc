PTH=${PTH:-'../bin'}

# Request 10,000 random graphs on six vertices, sort by fingerprint, and save to file one representative per iso class
$PTH/flagcalc -r 6 p=0.375 10000 -f all -g o=out6vertices.dat sorted overwrite -v rt

# Read the data back in from the file above, and check that there are zero pairwise isomorphic
$PTH/flagcalc -d out6vertices.dat -f all -C all -v cmp rt

# See "-C" at work: check each pair for isomorphism (contrast with -R sample randomly-chosen pairs)
$PTH/flagcalc -r 8 p=0.5 300 -f all -C all -v cmp rt fp Fp fpnone

# new feature: see fingerprinting only those that pass the criteria "conn1c" (aka "connected graph")
$PTH/flagcalc -r 5 p=0.3 3000 -a s1="conn1c" all -f passed -v cmp rt fp Fp fpnone i=minimal3.cfg

# new feature: see pairwise check "C" for isomorphism only on those that pass the criteria "conn1c"
$PTH/flagcalc -r 8 p=0.5 30 -a s1="conn1c" all -C passed -v cmp rt fp Fp fpnone i=minimal3.cfg

# not a new feature, just not used much elsewhere: "sorted" to check conn1c once per fingerprint-equiv-class
$PTH/flagcalc -r 5 p=0.5 3000 -f all -a s1="conn1c" sorted -v crit rt min Fp fp fpnone nofpseq

# some fun with Kuratowski's theorem on graph planarity
$PTH/flagcalc -r 7 p=0.75 3000 -f all -a s1="NOT (hasminorc(\"abcde\") OR hasminorc(\"abc=def\"))" sorted \
s1="NOT (hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\"))" sorted -v crit rt min Fp fp fpnone nofpseq

# 20 planar graphs on connected 5 vertices graphs
$PTH/flagcalc -r 5 p=0.5 1500 -a s1=conn1c s2="NOT (hasminorc(\"abcde\") OR hasminorc(\"abc=def\"))" all -f passed -v crit rt min Fp fp fpnone nofpseq

# 99 planar graphs on connected 6 vertices graphs
$PTH/flagcalc -r 6 p=0.5 1500 -a s1=conn1c s2="NOT (hasminorc(\"abcde\") OR hasminorc(\"abc=def\"))" all -f passed -v crit rt min Fp fp fpnone nofpseq

# more elementary checks
$PTH/flagcalc -r 8 p=0.5 30000 -f all -a s1="embedsc(\"abc\") IFF NOT cr1" sorted -v crit rt min Fp fp fpnone nofpseq

# more elementary checks
$PTH/flagcalc -r 8 p=0.5 30000 -f all -a s1="embedsc(\"abc\") IFF NOT cr1" all -a s1="embedsc(\"abc\") IFF NOT cr1" sorted -v crit rt min Fp fp fpnone nofpseq

# 1044 graphs on 7 vertices; 853 of them connected (source: https://users.cecs.anu.edu.au/~bdm/data/graphs.html)
$PTH/flagcalc -r 7 p=0.5 25000 -a s=conn1c all -f passed -v fp Fp fpnone rt crit nofpseq min

# 11117 connected graphs on 8 vertices, according to internet source above
$PTH/flagcalc -r 8 p=0.5 10000 -a s=conn1c all -f passed -v fp Fp fpnone rt crit min nofpseq

# Eulerian graphs, max of 54 on 7 vertices
$PTH/flagcalc -r 7 p=0.5 50000 -a s="FORALL (v IN V, vdt(v) % 2 == 0)" all -f passed -v fp Fp fpnone rt crit min nofpseq
$PTH/flagcalc -r 7 p=0.1 20000 -r 7 p=0.5 20000 -r 7 p=0.9 20000 -a s="FORALL (v IN V, vdt(v) % 2 == 0)" all -f passed -v fp Fp fpnone rt crit min nofpseq

# Connected Eulerian graphs, max of 184 on 8 vertices
$PTH/flagcalc -r 8 p=0.5 50000 -a s1="FORALL (v IN V, vdt(v) % 2 == 0)" s2=conn1c all -f passed -v fp Fp fpnone rt crit min nofpseq

# connected chordal graphs: 58 on 6 vertices
$PTH/flagcalc -r 6 p=0.5 3000 -a s1=conn1c s2="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" all -f passed -v fp Fp fpnone rt crit min nofpseq

# connected chordal graphs: 272 on 7 vertices: stated several different ways
$PTH/flagcalc -r 7 p=0.5 1500 -a s1=conn1c s2="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" all -f passed -v fp Fp fpnone rt crit min nofpseq
$PTH/flagcalc -r 7 p=0.5 1500 -a s1=conn1c s2="(NOT embedsc(\"-abcda\") AND NOT embedsc(\"-abcdea\") AND NOT embedsc(\"-abcdefa\") AND NOT embedsc(\"-abcdefga\")) IFF FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" -f passed -v fp Fp fpnone rt crit min nofpseq
$PTH/flagcalc -r 7 p=0.5 1500 -a s1=conn1c \
s2="FORALL (c IN Cycless, st(c) > 3, EXISTS (n IN NN(st(c) - 1), m IN NN(st(c) - 1), n < m AND NOT (n == 0 AND m + 2 == st(c)), ac(c[n],c[m+1])))" all -f passed -v fp Fp fpnone rt crit min nofpseq
$PTH/flagcalc -r 7 p=0.5 1500 -a s1=conn1c s2="FORALL (c IN Cycless, st(c) > 3, EXISTS (n1 IN st(c)-1, n2 IN st(c)-1, n1 < n2 AND NOT (n1 == 0 AND n2 == st(c)-2), ac(c[n1],c[n2+1])))" all -f passed -v fp Fp fpnone rt crit min nofpseq
$PTH/flagcalc -r 7 p=0.5 1500 -a s1=conn1c \
s2="FORALL (c IN Cycless, st(c) > 3, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c)) IFF FORALL (c IN Cycless, st(c) > 3, EXISTS (n1 IN st(c)-1, n2 IN st(c)-1, n1 < n2 AND NOT (n1 == 0 AND n2 == st(c)-2), ac(c[n1],c[n2+1])))" all -f passed -v fp Fp fpnone rt crit min nofpseq

# Perfect graphs: 33 on 5 vertices; 148 on 6 vertices; 906 on 7 vertices
$PTH/flagcalc -r 5 p=0.5 1500 -a s1="FORALL (c IN Cycless, st(c) > 4 && st(c) % 2 == 1, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" \
s2="FORALL (c IN Cycless(Complementg), st(c) > 4 && st(c) % 2 == 1, COUNT (v1 IN c, v2 IN c, v1 < v2, NOT ac(v1,v2)) > st(c))" all -f passed -v fp Fp fpnone rt crit min nofpseq
$PTH/flagcalc -r 6 p=0.5 1500 -a s1="FORALL (c IN Cycless, st(c) > 4 && st(c) % 2 == 1, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" \
s2="FORALL (c IN Cycless(Complementg), st(c) > 4 && st(c) % 2 == 1, COUNT (v1 IN c, v2 IN c, v1 < v2, NOT ac(v1,v2)) > st(c))" all -f passed -v fp Fp fpnone rt crit min nofpseq
$PTH/flagcalc -r 7 p=0.5 1500 -a s1="FORALL (c IN Cycless, st(c) > 4 && st(c) % 2 == 1, COUNT (v1 IN c, v2 IN c, v1 < v2, ac(v1,v2)) > st(c))" \
s2="FORALL (c IN Cycless(Complementg), st(c) > 4 && st(c) % 2 == 1, COUNT (v1 IN c, v2 IN c, v1 < v2, NOT ac(v1,v2)) > st(c))" all -f passed -v fp Fp fpnone rt crit min nofpseq

# Self-complementary graphs: 2 on 5 vertices
$PTH/flagcalc -r 5 p=0.5 1500 -a s="embedsc(Complementg)" all -f passed -v fp Fp fpnone rt crit min nofpseq

# Self-complementary graphs: 10 on 8 vertices
$PTH/flagcalc -r 8 p=0.5 1500 -a s="embedsc(Complementg)" all -f passed -v fp Fp fpnone rt crit min nofpseq
