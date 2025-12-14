# The aim is Kuratowski's theorem, e.g. to code ``embeds subdivision of of K_5"
PTH=${PTH:-'../bin'}


# Embeds K_5 partition?

# Intractable Working, K_5, not using Choices

#$PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (es IN Ps(E), st(es) >= 10, \
#NAMING (cupes AS BIGCUP (e IN es, e), \
#EXISTS (vs IN Sizedsubset(cupes,5), FORALL (v IN cupes SETMINUS vs, NOT EXISTS (e1 IN es, e2 IN es, e3 IN es, e1 != e2 AND e1 != e3 AND e2 != e3 AND v ELT e1 AND v ELT e2 AND v ELT e3)), \
#FORALL (v1 IN vs, v2 IN vs, v1 < v2, EXISTS (path IN Pathss(v1,v2), st(path CAP vs) == 2, FORALL (i IN NN(st(path) - 1), {path[i],path[i+1]} ELT es))))))" \
#all -v crit allcrit i=minimal3.cfg

# Intractable Working, K_3,3, not using choices

#$PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (es IN Ps(E), st(es) >= 9, \
#NAMING (cupes AS BIGCUP (e IN es, e), \
#EXISTS (vs IN Sizedsubset(cupes,6), FORALL (v IN cupes SETMINUS vs, NOT EXISTS (e1 IN es, e2 IN es, e3 IN es, e1 != e2 AND e1 != e3 AND e2 != e3 AND v ELT e1 AND v ELT e2 AND v ELT e3)), \
#EXISTS (vsleft IN Sizedsubset(vs,3), NAMING (vsright AS vs SETMINUS vsleft, \
#FORALL (v1 IN vsleft, v2 IN vsright, EXISTS (path IN Pathss(v1,v2), st(path CAP vs) == 2, FORALL (i IN NN(st(path)-1), {path[i],path[i+1]} ELT es))))))))" \
#all -v crit allcrit i=minimal3.cfg



# working K_3,3

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,6), EXISTS (vleft IN Sizedsubset(v,3), NAMING (vright AS (v SETMINUS vleft), NAMING (lcupr AS (vleft CUPD vright), NAMING (paths AS TUPLE (v1 IN vleft, v2 IN vright, TUPLE (p IN Pathss(v1,v2), st(p CAP lcupr) == 2,p)), \
# EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0))))))) " \
# all -v crit allcrit i=minimal3.cfg
 
# working K_5

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,5), NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2, p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], \
# NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \
# all -v crit allcrit i=minimal3.cfg

# ...together working

# $PTH/flagcalc -d testplanarshort.dat -a s="EXISTS (v IN Sizedsubset(V,6), EXISTS (vleft IN Sizedsubset(v,3), NAMING (vright AS (v SETMINUS vleft), NAMING (lcupr AS (vleft CUPD vright), NAMING (paths AS TUPLE (v1 IN vleft, v2 IN vright, TUPLE (p IN Pathss(v1,v2), st(p CAP lcupr) == 2,p)), \
# EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0))))))) \
# OR EXISTS (v IN Sizedsubset(V,5), NAMING (paths AS TUPLE (v1 IN v, v2 IN v, v1 < v2, TUPLE (p IN Pathss(v1,v2), st(p CAP v) == 2, p)), EXISTS (ch IN Choices(paths), FORALL (i IN NN(st(ch)), j IN NN(st(ch)), i < j , NAMING (pathi AS (paths[i])[ch[i]], NAMING (pathj AS (paths[j])[ch[j]], \
# FORALL (k IN NN(st(pathi) - 2), l IN NN(st(pathj) - 2), pathi[k+1] != pathj[l+1]))) AND ch[i] >= 0 AND ch[j] >= 0)))) " \
# all -v crit allcrit i=minimal3.cfg
 

$PTH/flagcalc -d testplanarshort.dat -d testplanarsmall.dat -a isp="../scripts/planarity.dat" s="planar" all -v crit allcrit i=minimal3.cfg

$PTH/flagcalc -d testplanarshort.dat -d testplanarsmall.dat -a isp="../scripts/planarity.dat" s="criticalnonplanar" all -v crit allcrit i=minimal3.cfg

$PTH/flagcalc -d testplanarshort.dat -d testplanarsmall.dat -a isp="../scripts/planarity.dat" s="apexgraph" all -v crit allcrit i=minimal3.cfg

