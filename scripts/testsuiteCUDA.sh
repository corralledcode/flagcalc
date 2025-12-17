# CUDA-enabled flagcalc starts with two tracks: GPU <quantifier> or natively CUDA-coded measures

PTH=${PTH:-'../bin'}

# First some basic set theory: this should return True
$PTH/flagcalc -d f="abcde" -a s="FORALL (u IN Ps(V), v IN Ps(V), GPU EXISTS (x IN V, 1, x ELT (u CAP v) IFF (x ELT u AND x ELT v)))"

$PTH/flagcalc -d f="abcde" -a s="FORALL (v IN V, GPU EXISTS (s IN Ps(V), t IN Ps(V), 1, v ELT (s CAP t)))"

$PTH/flagcalc -d f="abcde" -a s="FORALL (s IN Ps(V), GPU EXISTS (t IN Ps(V), 1, t <= s))"

$PTH/flagcalc -d f="abcde" -a s="GPU FORALL (s IN Ps(V), t IN Ps(V), 1, s <= (s CUP t) AND (s CAP t) <= t)"

$PTH/flagcalc -d f="abcd" -a s="GPU FORALL (s IN Ps(V), t IN Ps(V), u IN Ps(V), 1, (s <= t AND s <= u) IMPLIES s <= (t CAP u))"

$PTH/flagcalc -d f="abcd" -a s="GPU FORALL (u IN V, v IN V, w IN V, x IN V, u1 AS u+1, v1 AS v+1, w1 AS w+1, x1 AS x+1, 1, log(u1*v1*w1*x1) == (log(u1) + log(v1) + log(w1) + log(x1)))"

$PTH/flagcalc -d f="abcd" -a s="FORALL (s IN Ps(V), GPU FORALL (u IN V, v IN V, w IN V, x IN V, u1 AS u+1, v1 AS v+1, w1 AS w+1, x1 AS x+1, 1, (u ELT s AND v ELT s AND w ELT s AND x ELT s) IMPLIES log(u1*v1*w1*x1) == (log(u1) + log(v1) + log(w1) + log(x1))))"

$PTH/flagcalc -d f="abcd" -a s="GPU SUM (u IN V, v IN V, w IN V, x IN V, u1 AS u+1, v1 AS v+1, w1 AS w+1, x1 AS x+1, log(u1*v1*w1*x1) - ((log(u1) + log(v1) + log(w1) + log(x1))))"

$PTH/flagcalc -d massivegraph.dat -a s="GPU FORALL (v IN V, 1, mod(v,2) == 0 OR mod(v,2) == 1)"

$PTH/flagcalc -d massivegraph.dat -a a="GPU SUM (v IN V, w IN V, 1, (log(v+1) + log(v + 1) + log(v+1) + log(v+1) + log(v+1) + log(v+1) + log(v+1) + log(v+1)) - log(exp(8*log(v+1))))"

$PTH/flagcalc -d f="abc def ghi jkl mno pqr stu vwx yz" -a p="nwalksbetweenp(30)" all -a p="CUDAnwalksbetweenp(30)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a p="nwalksbetweenp(30)" all -a p="CUDAnwalksbetweenp(30)" all -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a p="nwalksbetweenp(30) == CUDAnwalksbetweenp(30)" all -v i=minimal3.cfg

$PTH/flagcalc -d f="a b c d e f g h i j" -a s="GPU EXISTS (u IN Ps(V), 1, 9 <= st(u))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="a b c d e f g h i j k" -a z="GPU SUM (u IN Ps(V), 1, st(u))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="a b c d e f g h i j k" -a z="GPU SUM (u IN Ps(V), 1, st(u))" all -a z="GPU SUM (u IN Ps(V), 1, st(u))" all -a z="SUM (u IN Ps(V), 1, st(u))" all -v i=minimal3.cfg

$PTH/flagcalc -d massivegraph.dat -a z="GPU SUM (n IN NN(dimm), 1, phi(n))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="a" -a z="GPU SUM (n IN NN(dimm), 1, phi(15))" all -v i=minimal3.cfg

$PTH/flagcalc -d massivegraph.dat -a a="GPU SUM (w IN V, v AS dimm, 1, (mod(v*v+1,w+1) == 0)*(phi(w+1) + phi((v*v+1)/(w+1))))" all -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a a="GPU SUM (w IN V, v AS dimm, mod(v*v+1,w+1) == 0, phi(w+1) + phi((v*v+1)/(w+1)))" all -v i=minimal3.cfg

$PTH/flagcalc -d massivegraph.dat -a a="GPU SUM (w IN V, v AS dimm, mod(v*v+1,w+1) == 0, phi(w+1) + phi((v*v+1)/(w+1)))" all -a a="GPU SUM (w IN V, v AS dimm, mod(v*v+1,w+1) == 0, phi(w+1) + phi((v*v+1)/(w+1)))" all -a a="SUM (w IN V, v AS dimm, mod(v*v+1,w+1) == 0, phi(w+1) + phi((v*v+1)/(w+1)))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="a b c d e f g h i j k" -a z="GPU SUM (u IN V, v IN V, w IN V, 1 <= u*v*w, phi(u*v*w))" -a z="GPU SUM (u IN V, v IN V, w IN V, 1 <= u*v*w, phi(u*v*w))" -a z="SUM (u IN V, v IN V, w IN V, 1 <= u*v*w, phi(u*v*w))" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefgh" -a z="GPU SUM (u IN V, v IN V, 1 <= u*v AND u + 1 <= v, nchoosek(v,u))" -a z="GPU SUM (u IN V, v IN V, 1 <= u*v AND u + 1 <= v, nchoosek(v,u))" -a z="SUM (u IN V, v IN V, 1 <= u*v AND u + 1 <= v, nchoosek(v,u))" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefghijklm" -a s="GPU FORALL (u IN V, v IN V, 1 <= u*v AND u + 1 <= v, nchoosek(v,u) == nchoosek(v, v - u))" -a s="GPU FORALL (u IN V, v IN V, 1 <= u*v AND u + 1 <= v, nchoosek(v,u) == nchoosek(v, v - u))" -a s="FORALL (u IN V, v IN V, 1 <= u*v AND u + 1 <= v, nchoosek(v,u) == nchoosek(v, v - u))" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefghijklm" -a s="GPU FORALL (u IN V, v IN V, 1 <= u*v, gamma(u*v/v))" -a s="GPU FORALL (u IN V, v IN V, 1 <= u*v, gamma(u*v/v))" -a s="FORALL (u IN V, v IN V, 1 <= u*v, gamma(u*v/v) > 0)" -v i=minimal3.cfg

$PTH/flagcalc -d f="abcdefghijklmnopq" -a s="SUM (a IN SET (n IN V, 1, GPU SUM (k IN NN(n + 1), 1, nchoosek(n,k))), a) == GPU SUM (n IN V, 1, exp(log(2)*n))" -a a="SUM (a IN SET (n IN V, 1, GPU SUM (k IN NN(n + 1), 1, nchoosek(n,k))), a)" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefghijklmnopq" -a z="SUM (a IN SET (n IN V, 1, SUM (k IN NN(n + 1), 1, nchoosek(n,k))), a)" -a a="SUM (n IN V, 1, exp(log(2)*n))"  -v i=minimal3.cfg

$PTH/flagcalc -d f="abcdefghijklmnopqrstuvwxyz" -a z="GPU SUM (n IN NN(dimm), k IN NN(dimm+1), k <= n, nchoosek(n,k))" -a z="GPU SUM (n IN NN(dimm), k IN NN(dimm+1), k <= n, nchoosek(n,k))" -a z="SUM (n IN NN(dimm), k IN NN(dimm+1), k <= n, nchoosek(n,k))"

$PTH/flagcalc -d f="abcde" -a z="SUM (n IN NN(dimm+1), GPU SUM (a IN Sizedsubset(V,n), 1, st(a)))" -a z="SUM (n IN NN(dimm+1), GPU SUM (a IN Sizedsubset(V,n), 1, st(a)))" -a z="SUM (n IN NN(dimm+1), SUM (a IN Sizedsubset(V,n), 1, st(a)))"  -v i=minimal3.cfg

$PTH/flagcalc -d massivegraph.dat -a p="GPU TUPLE (n IN NN(dimm), m IN NN(10), k IN NN(10), phi(n*m + k) )" -a p="GPU TUPLE (n IN NN(dimm), m IN NN(10), k IN NN(10), phi(n*m + k) )"  -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a p="TUPLE (n IN NN(dimm), phi(n) )" -a e="TUPLE (n IN NN(dimm), phi(n) )"  -v i=minimal3.cfg

$PTH/flagcalc -d massivegraph.dat -a e="GPU SETD (n IN NN(dimm*dimm), sin(n) )" -a e="GPU SETD (n IN NN(dimm*dimm), sin(n) )" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a e="SETD (n IN NN(dimm*dimm), sin(n) )" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a s="GPU SETD (n IN NN(dimm*dimm), phi(n) ) == SETD (n IN NN(dimm*dimm), toInt(phi(n)) )" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a s="GPU SET (n IN NN(dimm*dimm), phi(n) ) == SET (n IN NN(dimm*dimm), toInt(phi(n)) )" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a s="GPU TUPLE (n IN NN(dimm*dimm), phi(n) ) == TUPLE (n IN NN(dimm*dimm), toInt(phi(n)) )" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a s="GPU TUPLE (n IN NN(dimm*dimm), sin(n) ) == TUPLE (n IN NN(dimm*dimm), sin(n) )" -v i=minimal3.cfg

# .. no support for the next line
$PTH/flagcalc -d f="abc def ghi" -a s="FORALL (e IN E, GPU EXISTS (u IN V, v IN V, u ELT e AND v ELT e))" -v i=minimal3.cfg

# ... but can be done like this
$PTH/flagcalc -d f="abcdefg" -a s="NAMING (Ename AS E, NAMING (d AS dimm, GPU FORALL (a IN V, b IN V, a < b, (a * d + b) ELT Ename)))" -v i=minimal3.cfg

$PTH/flagcalc -d f="abcd" -a s="GPU FORALL (s IN Ps(V), t IN Ps(V), u IN Ps(V), st(s) + st(t) + st(u) >= st(s CUP t CUP u))" -a s="GPU FORALL (s IN Ps(V), t IN Ps(V), u IN Ps(V), st(s) + st(t) + st(u) >= st(s CUP t CUP u))" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefg" -a s="FORALL (s IN Ps(V), t IN Ps(V), st(s CUP t) <= st(s) + st(t) AND st(s CAP t) <= st(s) AND st(s CAP t) <= st(t))" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefghijklmnopqrstuvwxyz" -a s="GPU FORALL (s IN V, t IN V, u IN V, s+t+u >= s AND s+t+u >= t AND s+t+u >= u)" -a s="GPU FORALL (s IN V, t IN V, u IN V, s+t+u >= s AND s+t+u >= t AND s+t+u >= u)" -v i=minimal3.cfg

# Below is the new cudafast option: using v1 IN V, v2 IN V, v3 IN V, and the device manages the ranging itself, not passed by the host. So significantly faster here (by contrast to all the above examples)
$PTH/flagcalc -d f="abcdefghijklmnopqrstuvwxyz" -a s="GPU FORALL (v1 IN V, v2 IN V, v3 IN V, v1+v2+v3 >= v1 AND v1+v2+v3 >= v2 AND v1+v2+v3 >= v3)" -a s="GPU FORALL (v1 IN V, v2 IN V, v3 IN V, v1+v2+v3 >= v1 AND v1+v2+v3 >= v2 AND v1+v2+v3 >= v3)" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefghijklmnopqrstuvwxyz" -a s="FORALL (v1 IN V, v2 IN V, v3 IN V, v1+v2+v3 >= v1 AND v1+v2+v3 >= v2 AND v1+v2+v3 >= v3)" -v i=minimal3.cfg

$PTH/flagcalc -d massivegraph.dat -a s="GPU FORALL (v1 IN V, v2 IN V, v3 IN V, v1+v2+v3 >= v1 AND v1+v2+v3 >= v2 AND v1+v2+v3 >= v3)" -a s="GPU FORALL (v1 IN V, v2 IN V, v3 IN V, v1+v2+v3 >= v1 AND v1+v2+v3 >= v2 AND v1+v2+v3 >= v3)" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a s="FORALL (v1 IN V, v2 IN V, v3 IN V, v1+v2+v3 >= v1 AND v1+v2+v3 >= v2 AND v1+v2+v3 >= v3)" -v i=minimal3.cfg

$PTH/flagcalc -d massivegraph.dat -a z="GPU SUM (v1 IN V, v2 IN V, v3 IN V, v1 < v2 AND v2 < v3, v1+v2+v3)" -a z="GPU SUM (v1 IN V, v2 IN V, v3 IN V, v1 < v2 AND v2 < v3, v1+v2+v3)" -v i=minimal3.cfg
$PTH/flagcalc -d massivegraph.dat -a z="SUM (v1 IN V, v2 IN V, v3 IN V, v1 < v2 AND v2 < v3, v1+v2+v3)" -v i=minimal3.cfg

# runtime of 1:31 on an i9 with a 4050 GPU

$PTH/flagcalc -d f="abcdefghijklmnopqrstuvwxyza1b1c1d1e1f1g1h1i1j1k1l1m1n1o1p1q1r1s1t1u1v1w1x1y1z1" -a s="GPU FORALL (v1 IN V, v2 IN V, v1 != v2, ac(v1,v2))" -a s="GPU FORALL (v1 IN V, v2 IN V, v1 != v2, ac(v1,v2))" -v i=minimal3.cfg
$PTH/flagcalc -d f="abcdefghijklmnopqrstuvwxyza1b1c1d1e1f1g1h1i1j1k1l1m1n1o1p1q1r1s1t1u1v1w1x1y1z1" -a s="FORALL (v1 IN V, v2 IN V, v1 != v2, ac(v1,v2))" -v i=minimal3.cfg


# 1:18 5/13/2025

$PTH/flagcalc -d f="abcd efghi ijkl mnop" -a p="GPU TUPLE (v1 IN V, v2 IN V, connvc(v1,v2))" -a p="GPU TUPLE (v1 IN V, v2 IN V, connvc(v1,v2))" -a p="TUPLE (v1 IN V, v2 IN V, connvc(v1,v2))" -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc d efg hi ij kl mno p" -a s="GPU TUPLE (v1 IN V, v2 IN V, connvc(v1,v2)) == TUPLE (v1 IN V, v2 IN V, connvc(v1,v2))" -v i=minimal3.cfg

$PTH/flagcalc -d f="-abcd -efghi -ijkl -mnop -qrst -uvwx yz" -a p="GPU TUPLE (v1 IN V, v2 IN V, connvc(v1,v2))" -a p="GPU TUPLE (v1 IN V, v2 IN V, connvc(v1,v2))" -a p="TUPLE (v1 IN V, v2 IN V, connvc(v1,v2))" -v set allsets i=minimal3.cfg

# 0:46 5/14/2025

$PTH/flagcalc -d f="-abc -defgd ah" -a p="TUPLE (v1 IN V, TUPLE( v2 IN V, connvc(v1,v2)))" -a p="Connv" -v set allsets i=minimal3.cfg