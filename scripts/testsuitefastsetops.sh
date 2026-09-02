PTH=${PTH:-'../bin'}

# This script is for debugging the myriad and somewhat convulated "abstractsubsetmaker" and such, which easily halve or better the
# output of queries via identifying a subset of an integer sequence using a boolean vector, and flattening 2d symmetric sets (i.e. graph edges)
# into one dimension to obtain same treatment as subsets of integers

$PTH/flagcalc -d f="abcd" -a e="BIGCUPD (F IN Ps(E), H IN Ps(E), NOT ({0,1} ELT F) AND NOT ({0,1} ELT H), F CUPD H)" -v i=minimal3.cfg allsets

# from 3.13 seconds to 0.014 seconds (bearing in mind the "D")
$PTH/flagcalc -d f="abcdefg" -a e="BIGCUPD (U IN Ps(V), W IN Ps(V), NOT (0 ELT U) AND NOT (0 ELT W), U CUPD W)" -v i=minimal3.cfg allsets

# from 0.02 seconds to 0.014
$PTH/flagcalc -d f="abcdefg" -a e="BIGCUP (U IN Ps(V), W IN Ps(V), NOT (0 ELT U) AND NOT (0 ELT W), U CUP W)" -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="abcdefg" -a e="BIGCUP (U IN Ps(V), U == {1,2,4}, BIGCUP (W IN Ps(U), W))" -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abcdefg" -a e="BIGCAP (U IN Ps(V), BIGCAP (W IN Ps(U), st(W) < 7, W))" -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abcdefg" -a e="SET (U IN Ps(V), BIGCUPD (W IN Sizedsubset(U,1), W != {2}, W))" -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="abc" -a p="BIGCUPD (U IN Ps(V), U != Nulls, BIGCUPD (W IN Perms(U), W != <<2>>, W))" -v i=minimal3.cfg allsets


# from 9 seconds to 4.7 seconds
$PTH/flagcalc -d f="abcde" -a e="BIGCUP (H IN Ps(E), K IN Ps(E), NOT ({0,1} ELT H) AND NOT ({0,1} ELT K), H CUP K)" -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="abcde" -a e="BIGCUP (H IN Ps(E), K IN Ps(E), NOT ({3,1} ELT H) AND NOT ({3,1} ELT K), H CUP K)" -v i=minimal3.cfg allsets

# Should result in a set of tuples (where CUP between tuples is appending one to the other)
$PTH/flagcalc -d f="abc -defgd" -a e="BIGCUP (H IN Cycless, K IN Cycless, H CUP K)" -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abc -defgd" -a e="SETD (H IN Cycless, K IN Cycless, H CAP K)" -v i=minimal3.cfg allsets

# should all be true
$PTH/flagcalc -d f="a" f="ab" f="abc" f="abcd" f="abcde" f="abcdef" f="abcdefg" -a s="BIGCUP (n IN dimm, Setnpartitions(V,n+1)) == Setpartition(V)" -v i=minimal3.cfg allsets

# mtareequalgenerous had a bug around recognizing that 10.0 == 10. Fixed 9/1/2026
$PTH/flagcalc -d f="a" -a e="BIGCUP (S IN Ps({1.5,2.5,3.5,4.6,5.9,6.0,7.45,8.01,9,10}), T IN Ps({1.2,2.8,3.4,4.6,5,6.1,7.5,8.1,9.5,10.0}), S CUP T)" -v i=minimal3.cfg allsets

# no bug here, however major speed-up from 4 seconds to 2.5 seconds; and of course replacing each powerset with P(V) gives 0.4 seconds, super fast (that is, it doesn't id literal constants as a set of integers yet
$PTH/flagcalc -d f="a" -a e="BIGCUP (S IN Ps({1,2,3,4,5,6,7,8,9,10}), T IN Ps({11,12,13,14,5,6,17,18,19,20}), S CUP T)" -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="a" -a e="BIGCAP (S IN Ps({1,2,3,4,5,6,7,8,9,10}), T IN Ps({11,12,13,14,5,6,17,18,19,20}), 5 ELT S AND 5 ELT T, S CAP T)" -v i=minimal3.cfg allsets
$PTH/flagcalc -d f="a" -a e="BIGCUP (S IN Ps({1,2,3, (-4),5, (-6),7, (-8),9, -10}), T IN Ps({11,12,13,14,5,6,17,18,19,20}), S CUP T)" -v i=minimal3.cfg allsets

$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg  i=minimal3.cfg
