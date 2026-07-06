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
$PTH/flagcalc -r 8 p=0.75 100 -a isp="../scripts/planarity.dat" s1="kconnc(4)" s2="planarquick" s3="circm == dimm" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 p=0.5 10 -a isp="../scripts/planarity.dat" s1="kconnc(4)" s2="planarquick" s3="circm == dimm" all -v i=minimal3.cfg

# ... Vertex cut rule
$PTH/flagcalc -r 8 p=0.5 100 -a s1="circm == dimm" s2="FORALL (s IN Ps(V), st(s) > 0, connm( SubgraphonUg( V SETMINUS s ) ) <= st(s))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 p=0.5 100 -a s1="circm == dimm" s2="FORALL (s IN Ps(V), st(s) > 0, connm( SubgraphonUg( V SETMINUS s ) ) <= st(s))" all -v i=minimal3.cfg

