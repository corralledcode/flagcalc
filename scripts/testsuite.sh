PTH='../cmake-build-debug'
$PTH/flagcalc -L 10 18 1000
$PTH/flagcalc -d testbip12.dat -f all -i sorted
$PTH/flagcalc -d testgraph23.dat -i all
$PTH/flagcalc -r 6 8 200 -f all -i sortedverify -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -g o=out4.dat overwrite all -v i=minimal3.cfg
$PTH/flagcalc -d out4.dat -a s="conn1c" s2="forestc AND NOT treec" all -g o=out5.dat passed overwrite -v i=minimal3.cfg
$PTH/flagcalc -d testgraph5.dat testgraph4.dat testgraph30.dat -f all -i sorted
$PTH/flagcalc -r 10 18 100 -a a="dm * dimm == 2 * edgecm" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 18 2000 -a c=cr1 s2="edgecm > dimm^2/4" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 15 3000 -a ft="abcd" s="cliquem < 4" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 15 20000 -a s="Deltam > 2" s2="dimm < Deltam/(Deltam-2) * (Deltam - 1)^radiusm" all -g o=out6.dat overwrite passed -v i=minimal3.cfg
$PTH/flagcalc -d out6.dat -a nf="embeddings.dat" m2=girthm m2=circm all -v i=minimal3.cfg
$PTH/flagcalc -d out6.dat -a is="sentence.dat" s2="radiusm <= diamm AND diamm <= 2*radiusm" all -v i=minimal3.cfg
$PTH/flagcalc -d out4.dat -a ia="sentence.dat" s2="diamc(2)" all -v i=minimal3.cfg
$PTH/flagcalc -d out4.dat -a s="Knc(cliquem,1)" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 18 100 -a s1="NOT isinf(radiusm)" s2="connc(radiusm)" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 18 100 -a a="radiusm" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 12 50000 -a c=cr1 s2="cyclet(5) <= 2^5" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 20 5000 -a nc=forestc s2="girthm <= (2*diamm + 1)" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 20 50000 -a s="deltam >= 3" s2="girthm < 2*log(dimm)" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 25 50000 -a c=conn1c s2="treec == (edgecm == dimm-1)" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip12.dat -u GRAPH0 n="" 'r=rs1(5,100000)' -a sub t='cyclet(4)' -v i=minimal3.cfg
$PTH/flagcalc -r 12 20 15 -v i=minimal3.cfg
$PTH/flagcalc -r 20 25 1 -u GRAPH0 n="a b c" 'r=rs1(10,10000)' -a m=girthm c=conn1c c2=forestc c2=treec sub -v i=minimal3.cfg
$PTH/flagcalc -d testbip10.dat -a "c=kconnc(6)" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip10.dat -a "c=kconnc(5)" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 12 1000 -a s="kappat <= deltam" all -v i=minimal3.cfg min subobj srm
$PTH/flagcalc -r 10 12 10000 -a s="(kappat > 0) == conn1c" all -v i=minimal3.cfg min subobj srm
$PTH/flagcalc -r 8 7 1000 -a is=quantforestcrit.dat all -v i=minimal3.cfg
$PTH/flagcalc -d testbip10.dat -a is=bipartitecrit2.dat all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd=defg" -a is=bipartitecrit2.dat all -v i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a is=bipartitecrit2.dat all -v i=minimal3.cfg
$PTH/flagcalc -r 10 12 1000 -a c=cr1 s2="FORALL (x IN V, FORALL (y IN V, FORALL (z IN V, NOT (ac(x,y) AND ac(x,z) AND ac(y,z)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 12 1000 -a s="cr1 != FORALL (x IN V, FORALL (y IN V, FORALL (z IN V, NOT (ac(x,y) AND ac(x,z) AND ac(y,z)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 15 10 200 -a s="FORALL (s IN Sizedsubset(V,dimm), s == V)" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 4 100 -a s="EXISTS (y IN Sizedsubset(V,3), EXISTS (v IN V, EXISTS (u IN V, EXISTS (t IN V, EXISTS (w IN V, t ELT y AND u ELT y AND v ELT y AND w ELT y AND t != u AND t != v AND u != v AND t != w AND u != w AND v != w)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 4 100 -a s="EXISTS (y IN Sizedsubset(V,4), EXISTS (v IN V, EXISTS (u IN V, EXISTS (t IN V, EXISTS (w IN V, t ELT y AND u ELT y AND v ELT y AND w ELT y AND t != u AND t != v AND u != v AND t != w AND u != w AND v != w)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 10 10 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), (s CUP t) == V IMPLIES FORALL (x IN V, x ELT (s CUP t))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 10 5 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), (s CUP t) != V IMPLIES EXISTS (x IN V, NOT (x ELT (s CUP t)))))" all -v i=minimal3.cfg






