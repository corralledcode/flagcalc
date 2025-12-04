# The aim is Kuratowski's theorem, e.g. to code ``embeds subdivision of of K_5"
# PTH='../cmake-build-debug'


# Embeds K_5 partition?



# exists subset of 10 paths such that each is internally disjoint pairwise, and each pair v1, v2 appears exactly once in the set
  
# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), EXISTS (partition IN Setpartition(V SETMINUS v), st(partition) <= 10, \\
EXISTS (perm IN Perms(NN(5)), FORALL (i IN NN(5), j IN NN(5), perm[i] < perm[j], EXISTS (p IN Pathss(v[perm[i]],v[perm[j]]), \\
NAMING (tri AS (9 - nchoosek(5 - i, 2) + (j - i)), tri >= st(partition) ? st(p) == 2 : st(p CAP partition[tri]) == (st(p) - 2)))))) OR FORALL (v1 IN v, v2 IN v, v1 < v2, ac(v1,v2)))" all -v crit allcrit i=minimal3.cfg

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2 AND FORALL (internalv IN NN(st(p) - 2), vdt(p[internalv+1]) == 2), p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2 >= 0 ? st(pathi) - 2 : 0), l IN NN(st(pathj) - 2 >= 0 ? st(pathj) - 2 : 0), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \\
all -v crit allcrit i=minimal3.cfg

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2, p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \\
all -v crit allcrit i=minimal3.cfg

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,6), EXISTS (vleft IN Sizedsubset(v,3), NAMING (vright AS (v SETMINUS vleft), NAMING (lcupr AS (vleft CUPD vright), NAMING (paths AS TUPLE (v1 IN vleft, v2 IN vright, TUPLE (p IN Pathss(v1,v2), st(p CAP lcupr) == 2,p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0))))))) " \\
all -v crit allcrit i=minimal3.cfg



# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2, p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \\
all -v crit allcrit i=minimal3.cfg



# not working, without Choices, K_5

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (vp IN Ps(V), st(vp) >= 5, EXISTS (v IN Sizedsubset(vp,5), \\
FORALL (v1 IN v, v2 IN v, v1 < v2, EXISTS (path IN Pathss(v1,v2), path <= vp AND st(path CAP v) == 2, FORALL (i IN NN(st(path) - 2), st(Ns({path[i+1]}) CAP vp) == 2)))))" \\
all -v crit allcrit i=minimal3.cfg

# not working, without Choices, K_3,3

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (vp IN Ps(V), st(vp) >= 6, EXISTS (v IN Sizedsubset(vp,6), EXISTS (vleft IN Sizedsubset(v,3), NAMING (vright AS v SETMINUS vleft, \\
FORALL (v1 IN vleft, v2 IN vright, EXISTS (path IN Pathss(v1,v2), path <= vp AND st(path CAP v) == 2, FORALL (i IN NN(st(path) - 2), st(Ns({path[i+1]}) CAP vp) == 2)))))))" \\
all -v crit allcrit i=minimal3.cfg


# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (es IN Ps(E), st(es) >= 10, NAMING (cupes AS BIGCUP (e IN es, e), EXISTS (v IN Sizedsubset(cupes, 5), \\
NAMING (esminusv AS SET (e IN es, NOT (e <= v), e), \\
FORALL (v1 IN v, v2 IN v, v1 < v2, EXISTS (path IN Pathss(v1,v2), path <= cupes, FORALL (i IN NN(st(path)-2), NOT path[i+1] ELT v) AND FORALL (e IN esminusv, (e[0] ELT path IFF e[1] ELT path) OR e[0] == v1 OR e[1] == v2 )))))))" \\
all -v crit allcrit i=minimal3.cfg

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (es IN Ps(E), st(es) >= 10, NAMING (cupes AS BIGCUP (e IN es, e), EXISTS (v IN Sizedsubset(cupes, 5), \\
NAMING (esminusv AS SET (e IN es, NOT (e <= v), e), \\
FORALL (v1 IN v, v2 IN v, v1 < v2, EXISTS (path IN Pathss(v1,v2), FORALL (e IN esminusv, e[0] ELT path IFF e[1] ELT path) AND FORALL (i IN NN(st(path)-1), {path[i], path[i+1]} ELT esminusv) OR st(path) == 2))))))" \\
all -v crit allcrit i=minimal3.cfg




 NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2, p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \\
all -v crit allcrit i=minimal3.cfg


# Working, K_5, no Choices

$PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (n IN NN(st(E) - 10 >= 0 ? st(E) - 10 : 0), EXISTS (es IN Sizedsubset(E,n+10), \\
NAMING (cupes AS BIGCUP (e IN es, e), \\
EXISTS (vs IN Sizedsubset(cupes,5), FORALL (v IN cupes SETMINUS vs, NOT EXISTS (e1 IN es, e2 IN es, e3 IN es, e1 != e2 AND e1 != e3 AND e2 != e3 AND v ELT e1 AND v ELT e2 AND v ELT e3)), \\
FORALL (v1 IN vs, v2 IN vs, v1 < v2, EXISTS (path IN Pathss(v1,v2), st(path CAP vs) == 2, FORALL (i IN NN(st(path) - 1), {path[i],path[i+1]} ELT es)))))))" \\
all -v crit allcrit i=minimal3.cfg

# Working, K_3,3, no choices

$PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (es IN Ps(E), st(es) >= 9, \\
NAMING (cupes AS BIGCUP (e IN es, e), \\
EXISTS (vs IN Sizedsubset(cupes,6), FORALL (v IN cupes SETMINUS vs, NOT EXISTS (e1 IN es, e2 IN es, e3 IN es, e1 != e2 AND e1 != e3 AND e2 != e3 AND v ELT e1 AND v ELT e2 AND v ELT e3)), 
EXISTS (vsleft IN Sizedsubset(vs,3), NAMING (vsright AS vs SETMINUS vsleft, \\
FORALL (v1 IN vsleft, v2 IN vsright, EXISTS (path IN Pathss(v1,v2), st(path CAP vs) == 2, FORALL (i IN NN(st(path)-1), {path[i],path[i+1]} ELT es))))))))" \\
all -v crit allcrit i=minimal3.cfg




# working K_3,3, fast

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,6), EXISTS (vleft IN Sizedsubset(v,3), NAMING (vright AS (v SETMINUS vleft), NAMING (lcupr AS (vleft CUPD vright), NAMING (paths AS TUPLE (v1 IN vleft, v2 IN vright, TUPLE (p IN Pathss(v1,v2), st(p CAP lcupr) == 2,p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0))))))) " \\
all -v crit allcrit i=minimal3.cfg

# working K_5, fast

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2, p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \\
all -v crit allcrit i=minimal3.cfg


# together working, fast

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,6), EXISTS (vleft IN Sizedsubset(v,3), NAMING (vright AS (v SETMINUS vleft), NAMING (lcupr AS (vleft CUPD vright), NAMING (paths AS TUPLE (v1 IN vleft, v2 IN vright, TUPLE (p IN Pathss(v1,v2), st(p CAP lcupr) == 2,p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0))))))) \\
OR EXISTS (v IN Sizedsubset(V,5), NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2, p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \\
all -v crit allcrit i=minimal3.cfg





# -d std::cin -a e="Choices(NAMING (paths AS TUPLE (v1 IN V, v2 IN V, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(TupletoSet(p) CAP V) == 2, p)), paths))" \\ all -v crit allsets i=minimal3.cfg


there is a subset containing five and a partition of its edges into ten such that some permutation of five vertices satisfies: non-equal partition elements meet at most at the five.

exists subset of v such that there is a five element subset of it, and a tuple of integers such that each vertex pair maps to a differently-colored set of edges

# Maps(Set1, Set2)



# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), EXISTS (ep IN Setpartition(E), st(ep) == 10, EXISTS (perm IN Perms(NN(10)), FORALL (i IN NN(5), j IN NN(5), i < j, connvsc({v[i],v[j]}, ep[perm[9 - nchoosek(5 - i,2) + (j - i)]])))))" all -v crit allcrit i=minimal3.cfg

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), EXISTS (ep IN Setpartition(E), st(ep) == 10, EXISTS (perm IN Perms(NN(5)), FORALL (i IN NN(5), j IN NN(5), perm[i] < perm[j],
EXISTS (pt IN Pathss(v[perm[i]],v[perm[j]]), TUPLE (m IN NN(st(pt)-2), pt[m+1]) <= ep[9 - nchoosek(5 - i,2) + (j - i)])))))" all -v crit allcrit i=minimal3.cfg



# EXISTS (paths IN Sizedsubset(BIGCUP (v1 IN s, v2 IN s, v1 < v2, Pathss(v1,v2)),5*4/2), \\
FORALL (v1 IN s, v2 IN s, v1 < v2, st(SET (p IN paths, p[0] == v1 AND p[st(p)-1] == v2)) == 1) AND \\
nwisec( SETD (p IN paths, SETD (n IN NN(st(p)-2), p[n+1])), \"DISJOINT\",2,1)))" all -v crit allcrit i=minimal3.cfg


tuple of 1...st - 1:

TUPLE (n IN NN(st(p)-2, p[n+1]))




EXISTS (paths IN Setpartition(V), st(paths) == 5*4/2), FORALL (v1 IN s, v2 IN s, v1 < v2, EXISTS (pv IN paths, p IN Pathss(v1,v2), p <= (pv CUP {v1,v2}) AND FORALL

 FORALL (p1 IN paths, p2 IN paths, FORALL (i IN st(p1)-2, j IN st(p2)-2, p1[i+1] != p2[j+1])) AND FORALL (v1 IN s, v2 IN s, v1 < v2, EXISTS (p IN paths, p[0] == v1 AND p[1] == v2))))" all -v crit allcrit i=minimal3.cfg


EXISTS (distinctpaths IN Choices(SET (v1 IN s, v2 IN s, v1 < v2, Pathss(v1,v2))), FORALL (p1 IN st(distinctpaths), p2 IN st(distinctpaths), p1 < p2, FORALL (n1 IN NN(st(distinctpaths[p1])-2), n2 IN NN(st(distinctpaths[p2])-2), (distinctpaths[p1])[n1+1] != (distinctpaths[p2])[n2+1]))))" all -v crit allcrit i=minimal3.cfg



$PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (s IN Sizedsubset(V,5), EXISTS (sp IN Setpartition(V), FORALL (v1 IN s, v2 IN s, v1 < v2, EXISTS (p IN Pathss(v1,v2), p <= (sp CUPD {v1,v2})

EXISTS (distinctpaths IN Ps(BIGCUP (w1 IN s, w2 IN s, w1 < w2, Pathss(w1,w2))), \\
FORALL (p1 IN st(distinctpaths), p2 IN st(distinctpaths), p1 < p2, FORALL (i IN st(distinctpaths[p1])-2, j IN st(distinctpaths[p2])-2, (distinctpaths[p1])[i+1] != (distinctpaths[p2])[j+1])) \\
AND FORALL (w1 IN s, w2 IN s, w1 < w2, EXISTS (p IN distinctpaths, p[0] == w1 AND p[st(p)-1] == w2))))" all -v crit allcrit i=minimal3.cfg



$PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (s IN Sizedsubset(V,5), EXISTS (distinctpaths IN Ps(BIGCUP (w1 IN s, w2 IN s, w1 < w2, Pathss(w1,w2))), \\
FORALL (p1 IN st(distinctpaths), p2 IN st(distinctpaths), p1 < p2, FORALL (i IN st(distinctpaths[p1])-2, j IN st(distinctpaths[p2])-2, (distinctpaths[p1])[i+1] != (distinctpaths[p2])[j+1])) \\
AND FORALL (w1 IN s, w2 IN s, w1 < w2, EXISTS (p IN distinctpaths, p[0] == w1 AND p[st(p)-1] == w2))))" all -v crit allcrit i=minimal3.cfg



$PTH/flagcalc -d testplanarshort.dat -a s="FORALL (s IN Sizedsubset(V,5), FORALL (sp IN Setpartition(V), st(sp) == 5*4/2, EXISTS (i IN NN(5), j IN NN(5), i < j, FORALL (p IN Pathss(s[i],s[j]), NOT (p <= ({s[i],s[j]} CUPD sp[TALLY (n IN j, n) + i] ))))))" all -v crit allcrit i=minimal3.cfg

EXISTS (v1 IN s, v2 IN s, v1 != v2 AND (NOT ac(v1,v2)) AND FORALL (p IN Pathss(v1,v2), TupletoSet(p) <= s )))" all -v crit allcrit i=minimal3.cfg

$PTH/flagcalc -d testplanar.dat -a s="FORALL (sp IN Setpartition(V), st(sp) == 5, EXISTS (i IN NN(5), j IN NN(5), i != j, FORALL (v1 IN sp[i], v2 IN sp[j], FORALL (p IN Pathss(v1,v2), NOT (p <= (sp[i] CAP sp[j]))))))" all -v crit allcrit i=minimal3.cfg


$PTH/flagcalc -d testplanar.dat -a s="FORALL (s IN Sizedsubset(V,5),
EXISTS (v1 IN s, v2 IN s, v1 != v2 AND (NOT ac(v1,v2)) AND FORALL (p IN Pathss(v1,v2), TupletoSet(p) <= s )))" all -v crit allcrit i=minimal3.cfg

# Embeds K_{3,3} partition?

$PTH/flagcalc -d testplanar.dat -a s="FORALL (sleft IN Sizedsubset(V,3), sright IN Sizedsubset(V,3), EXISTS (v1 IN sleftg, v2 IN sright, NOT connvc(v1,v2)))" all -v i=minimal3.cfg

# Altogether

$PTH/flagcalc -d testplanar.dat -a s="FORALL (s IN Sizedsubset(V,5), EXISTS (v1 IN s, v2 IN s, NOT connvc(v1,v2)))" s2="FORALL (sleft IN Sizedsubset(V,3), sright IN Sizedsubset(V,3), EXISTS (v1 IN sleft, v2 IN sright, NOT connvc(v1,v2)))" all -v i=minimal3.cfg
