#
PTH=${PTH:-'../bin'}

$PTH/flagcalc -r 150 75 1 -a s="THREADED PARTITION (u,v IN V, connvc(u,v)) == PARTITION (u,v IN V, connvc(u,v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 120 70 1 -a s="st(THREADED PARTITION (u,v IN V, connvc(u,v))) == connm" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 120 70 1 -a s="st(PARTITION (u,v IN V, connvc(u,v))) == connm" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="ab cde fghi jklmn" -a e="THREADED PARTITION (u,v IN V, w AS u+10, w > 10, connvc(u,v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="ab cde fghi jklmn" -a e="THREADED PARTITION (u,v IN V, w AS u+10, connvc(u,v))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="a bc" -a e="THREADED PARTITION (a,b IN Ps(V), st(b) > 0, st(a) == st(b))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="ab cde" -a s="THREADED PARTITION (a,b IN Ps(V), st(b) > 0, st(a) == st(b)) == PARTITION (a,b IN Ps(V), st(b) > 0, st(a) == st(b))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="ab cde" -a e="NAMING (p AS THREADED PARTITION (u, v IN V, connvc(u,v)), BIGCUPD (e IN p, SET (a IN e, b IN e, a != b, {a,b})))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="ab -cde" -a gm="NAMING (p AS THREADED PARTITION (u, v IN V, connvc(u,v)), BIGCUPD (e IN p, SET (a IN e, b IN e, a != b, {a,b})))" all -v measg i=minimal3.cfg

# two out of three:

$PTH/flagcalc -d f="ab cde" f="abcd efg hi" f="abc -defgd" -a s="NAMING (p AS THREADED PARTITION (u, v IN V, connvc(u,v)), NAMING (c AS BIGCUPD (e IN p, SET (a IN e, b IN e, a != b, {a,b})), EXISTS (r IN Perms(NN(dimm)), SET (g IN c, {r[g[0]],r[g[1]]})  == E)))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="ab cde" f="abcd efg hi" f="abc -defgd" -a s="NAMING (p AS PARTITION (u, v IN V, connvc(u,v)), NAMING (c AS BIGCUPD (e IN p, SET (a IN e, b IN e, a != b, {a,b})), EXISTS (r IN Perms(NN(dimm)), SET (g IN c, {r[g[0]],r[g[1]]})  == E)))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="ab cde" f="abcd efg hi" f="abc -defgd" -a s="NAMING (p AS PARTITION (u, v IN V, connvc(u,v)), NAMING (b AS THREADED BIGCUPD (e IN p, SET (a IN e, b IN e, a != b, {a,b})), EXISTS (r IN Perms(NN(dimm)), SET (g IN b, {r[g[0]],r[g[1]]})  == E)))" all -v i=minimal3.cfg

# two out of four:
$PTH/flagcalc -d f="ab cde" f="abcd efg hi" f="abc -defgd" f="abcd=efghi" -a s="EXISTS (s IN Setpartition(V), FORALL (t IN s, FORALL (u IN t, v IN t, connvc(u,v))) AND FORALL (t1 IN s, t2 IN s, t1 != t2, FORALL (u IN t1, v IN t2, !connvc(u,v))) AND edgecm == SUM (a IN s, st(a)*(st(a)-1)/2))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="ab cde" f="abcd efg hi" f="abc -defgd" f="abcd=efghi" -a s="EXISTS (s IN Setpartition(V), FORALL (u IN V, v IN V, connvc(u,v) IFF EXISTS (t IN s, u ELT t AND v ELT t)) AND edgecm == SUM (a IN s, st(a)*(st(a)-1)/2))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="ab cde" f="abcd efg hi" f="abc -defgd" f="abcd=efghi" -a isp="../scripts/storedprocedures.dat" s="EachComponentComplete IFF EXISTS (s IN Setpartition(V), FORALL (u IN V, v IN V, connvc(u,v) IFF EXISTS (t IN s, u ELT t AND v ELT t)) AND edgecm == SUM (a IN s, st(a)*(st(a)-1)/2))" all -v i=minimal3.cfg

# four out of four
# 7 seconds versus 1.5 seconds based on placement of THREADED
$PTH/flagcalc -d f="ab cde" -a isp="../scripts/storedprocedures.dat" s="EachComponentComplete IFF NAMING (p AS PARTITION (u, v IN V, connvc(u,v)), NAMING (c AS BIGCUPD (e IN p, SET (a IN e, b IN e, a != b, {a,b})), THREADED EXISTS (r IN Perms(NN(dimm)), SET (g IN c, {r[g[0]],r[g[1]]})  == E)))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="ab cde" f="abcd efg hi" f="abc -defgd" f="defg=hijk" -a isp="../scripts/storedprocedures.dat" s="EachComponentComplete IFF NAMING (p AS PARTITION (u, v IN V, connvc(u,v)), NAMING (c AS BIGCUPD (e IN p, SET (a IN e, b IN e, a != b, {a,b})), THREADED EXISTS (r IN Perms(NN(dimm)), SET (g IN c, {r[g[0]],r[g[1]]})  == E)))" all -v i=minimal3.cfg


$PTH/flagcalc -d f="abcd efg hi" -a s="THREADED EXISTS (r IN Perms(NN(dimm)), SET (g IN E, {r[g[0]],r[g[1]]})  == E)" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abcd efg hi" -a s="THREADED FORALL (r IN Perms(NN(dimm)), st(SETD (g IN E, {r[g[0]],r[g[1]]})) == st(E))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abcdef" -a s="THREADED FORALL (r IN Ps(V), st(r) == 2 IMPLIES r ELT E)" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abcdef" -a s="THREADED SETD (r IN Ps(V), st(r) == 2, r) == E" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abcdefghi" -a s="THREADED SET (r IN Ps(V), st(r) == 2, {r[0],r[1]}) == E" all -v i=minimal3.cfg

# elementary number theory

$PTH/flagcalc -d f="abcdefghi" -a isp="../scripts/storedprocedures.dat" s="FORALL (v IN V, v > 0, ntphi(n) == phi(n))" all -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a isp="../scripts/storedprocedures.dat" z="SUM (d IN V, v AS dimm, d1 AS d+1, mod(v,d1) == 0, ntphi(d1))" all -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a z="SUM (d IN V, d > 1, phi(d))" all -a isp="../scripts/storedprocedures.dat" z="SUM (d IN V, d > 1, ntphi(d))" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a isp="../scripts/storedprocedures.dat" s="FORALL (d IN V, d > 1, phi(d) == ntphi(d))" all -v i=minimal3.cfg
