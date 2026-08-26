PTH=${PTH:-'../bin'}

# Hamiltonian cycles, in four distinct ways
$PTH/flagcalc -r 8 p=0.5 100 -a s="EXISTS (p IN Perms(V), FORALL (a IN dimm, ac(p[a],p[(a+1) % dimm])))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="EXISTS (c IN Cycless(0), st(c) == dimm)" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="EXISTS (p IN Perms(V), FORALL (a IN dimm, ac(p[a],p[(a+1) % dimm])))" s="EXISTS (c IN Cycless(0), st(c) == dimm)" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="EXISTS (p IN Perms(V), FORALL (a IN dimm, ac(p[a],p[(a+1) % dimm]))) IFF EXISTS (c IN Cycless(0), st(c) == dimm)" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="embedsgenerousc(\"-abcdefgha\")" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="embedsgenerousc(\"-abcdefgha\") IFF EXISTS (c IN Cycless(0), st(c) == dimm)" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="circm == dimm" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 100 -a s="circm == dimm IFF EXISTS (c IN Cycless(0), st(c) == dimm)" all -v i=minimal3.cfg

# ... a fifth way...
$PTH/flagcalc -r 8 p=0.5 100 -a s="EXISTS (S IN Sizedsubset(E,dimm), FORALL (s IN Ps(S), NAMING (sz AS st(s), 0 < sz AND sz < dimm), EXISTS (v IN V, EXISTS (e IN s, v ELT e) AND NOT EXISTSN (2, e IN s, v ELT e))) AND FORALL (e IN S, NOT EXISTSN (4, e2 IN S, e MEET e2))) IFF circm == dimm" all -v i=minimal3.cfg

# ... Some theorems on Hamiltonian cycles:
# ... Dirac's Theorem (1952)
$PTH/flagcalc -r 8 p=0.5 100 -a s="circm == dimm IF deltam >= dimm/2" all -v i=minimal3.cfg

# ... Ore's Theorem (1960)
$PTH/flagcalc -r 8 p=0.5 100 -a s="circm == dimm IF FORALL (u IN V, v IN V, u < v, NOT ac(u,v) IMPLIES vdt(u) + vdt(v) >= dimm)" all -v i=minimal3.cfg

# ... Bondy-Chvátal Theorem
$PTH/flagcalc -r 8 p=0.5 100 -a s="circm == dimm IFF circm(Closureg) == dimm" all -v i=minimal3.cfg

# ... Tutte's Theorem (1956)
$PTH/flagcalc -r 8 p=0.75 100 -a isp="planarity.dat" s1="kconnc(4)" s2="planarquick" s3="circm == dimm" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 p=0.5 10 -a isp="planarity.dat" s1="kconnc(4)" s2="planarquick" s3="circm == dimm" all -v i=minimal3.cfg

# ... Vertex cut rule
$PTH/flagcalc -r 8 p=0.5 100 -a s1="circm == dimm" s2="FORALL (s IN Ps(V), st(s) > 0, connm( SubgraphonUg( V SETMINUS s ) ) <= st(s))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 p=0.5 100 -a s1="circm == dimm" s2="FORALL (s IN Ps(V), st(s) > 0, connm( SubgraphonUg( V SETMINUS s ) ) <= st(s))" all -v i=minimal3.cfg

# embedsgenerousc
$PTH/flagcalc -d f="abcde" f="abcdef" f="abc -cdefg ghij" -a s="embedsgenerousc(\"abcd -defgh hi\")" s="embedsgenerousc(\"abcde\")" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcde" f="abcdef" f="abc -cdefg ghij" -a s="embedsc(\"abcd -defgh hi\")" s="embedsc(\"abcde\")" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 p=0.5 10 -a s="FORALL (S IN Ps(V), FORALL (P IN Perms(S), embedsgenerousc(SubgraphonUg(P))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 p=0.5 10 -a s="FORALL (S IN Ps(V), FORALL (P IN Perms(S), embedsc(SubgraphonUg(P))))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="ab ae ag bc bd bf bg cf de df dg eg fg" -a s="EXISTS (S IN Ps(V), THREADED EXISTS (P IN Perms(S), NOT embedsgenerousc(SubgraphonUg(P))))" all -v any i=minimal3.cfg

$PTH/flagcalc -r 6 p=0.5 10 -a s="FORALL (S IN Ps(V), FORALL (T IN Ps(Edgess(S)), embedsgenerousc(GraphonVEg(S,T))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 p=0.5 10 -a s="FORALL (H IN Ps(V), FORALL (F IN Ps(Edgess(H)), embedsgenerousc(SubgraphonVEg(H,F))))" all -v i=minimal3.cfg

$PTH/flagcalc -r 6 p=0.5 10 -a e="ANY (S IN Ps(V), ANY (T IN Ps(Edgess(S)), NOT embedsgenerousc(GraphonVEg(S,T)), {S,T}))" all -v i=minimal3.cfg set allsets
$PTH/flagcalc -d f="ae af bc be cd ce cf df ef" -a s="embedsgenerousc(GraphonVEg({0,1,2,4},{{0,4},{1,4},{2,4}}))" all -v graphs crit set allsets
$PTH/flagcalc -d f="ae bc be ce f" -a s="embedsgenerousc(GraphonVEg({0,1,2,4},{{0,4},{1,4},{2,4}}))" all -v graphs crit set allsets

$PTH/flagcalc -r 7 p=0.5 10 -a z="MAX (H IN Ps(V), MIN (F IN Ps(Edgess(H)), NOT embedsc(SubgraphonVEg(H,F)), st(Edgess(H)) - st(F)))" all -v i=minimal3.cfg

$PTH/flagcalc -r 7 p=0.5 10 -a s="FORALL (H IN Ps(V), SubgraphonUg(H) == SubgraphonVEg(H,Edgess(H)))" all -v i=minimal3.cfg

$PTH/flagcalc -r 6 p=0.5 10 -a s="FORALL (H IN Ps(V), FORALL (F IN Ps(Edgess(H)), embedsgenerousc(SubgraphonVEg(H,F))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 p=0.5 10 -a s="FORALL (H IN Ps(V), FORALL (P IN Perms(H), FORALL (F IN Ps(Edgess(H)), embedsgenerousc(SubgraphonVEg(H,SETD (i IN st(H), j IN st(H), {H[i],H[j]} ELT F, {P[i],P[j]} ))))))" all -v i=minimal3.cfg

$PTH/flagcalc -r 12 p=0.5 10 -a s="cr1 IFF NOT embedsgenerousc(\"abc\")" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 p=0.5 100 -a s="embedsgenerousc(\"abcde\") IFF NOT embedsc(\"abcde\")" all -v any i=minimal3.cfg

$PTH/flagcalc -d f="-adbca" -f -a s="embedsgenerousc(SubgraphonVEg(V,{{0,2}}))" all -v graphs crit fp FpMin

$PTH/flagcalc -d f="ac ad ae af ag ah bc bd be bf bh cg de df dg ef eg eh fg fh gh" -a s="embedsc(\"abcde\")" s="embedsgenerousc(\"abcde\")" all -v i=minimal3.cfg

# the below NOT elegant, rather just a sampling of useful queries, not adjusted for run time

$PTH/flagcalc -r 7 p=0.5 40 -a s="FORALL (H IN Ps(V), FORALL (F IN Ps(Edgess(H)), embedsgenerousc(Complementg(SubgraphonVEg(H,F)),Complementg(SubgraphonUg(H)))))" all -v i=minimal3.cfg
# $PTH/flagcalc -r 7 p=0.5 40 -a s="FORALL (H IN Ps(V), FORALL (F IN Ps(Edgess(H)), embedsgenerousc(SubgraphonUg(H),SubgraphonVEg(H,F))))" all -v i=minimal3.cfg
# $PTH/flagcalc -r 7 p=0.5 40 -a s="FORALL (H IN Ps(V), FORALL (F IN Ps(Edgess(H)), embedsgenerousc(SubgraphonVEg(H,F))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 p=0.5 40 -a s="embedsgenerousc(SubgraphonUg({0,1,2}))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="ac ag bc cg dg fg bh" -a s="NOT embedsgenerousc(SubgraphonUg({0,1,2,3,5,6}))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 40 -a s="FORALL (H IN Ps(V), st(H)+3 < dimm, FORALL (F IN Ps(Edgess(H)), embedsgenerousc(SubgraphonVEg(H,F))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 p=0.5 40 -a s="FORALL (H IN Ps(V), st(H)+4 < dimm, FORALL (F IN Ps(Edgess(H)), embedsgenerousc(SubgraphonVEg(H,F))))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd -efghie" -a s="embedsgenerousc(\"-abcde fg\")" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefghij bk" -a s="embedsgenerousc(\"-abcdefghij ik\")" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcdefghi" -a s="embedsgenerousc(\"abc=def\")" all -v set allsets i=minimal3.cfg

# longint is 64 bits on linux; above 12 vertices cannot compute the powerset of selfpaired as it requires 66 bits
$PTH/flagcalc -r 10 p=0.5 40 -a isp="storedprocedures.dat" s="FORALL (S IN Ps(selfpaired(V)), S <= E) IFF Knc(dimm,1)" all -v i=minimal3.cfg

$PTH/flagcalc -d out8.dat -r 8 p=0.5 1500000 -r 8 p=0.8 200000  -a s=conn1c all -f passed -g o=out8b.dat passed sorted overwrite -v fp Fp fpnone rt crit min nofpseq