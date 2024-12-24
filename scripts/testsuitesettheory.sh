PTH='../cmake-build-debug'
$PTH/flagcalc -r 9 2 50 -a s="EXISTS (t IN [Ps]([V]), EXISTS (s IN [Ps]([V]), (s CUP t) == [V] AND [st](s) == ceil([dimm]/2) AND [st](t) == floor([dimm]/2)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a s="FORALL (n IN [NN]([dimm]), EXISTS (s IN [Ps]([V]), EXISTS (t IN [Ps]([V]), (s CUP t) == [V] AND [st](s) == ceil([dimm]/(n+1)) AND [st](t) == floor(n*[dimm]/(n+1)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 10 100 -a s="FORALL (e IN [E], EXISTS (a IN [V], EXISTS (b IN [V], a ELT e AND b ELT e AND a != b)))" all -v i=minimal3.cfg
$PTH/flagcalc -d halfconnectedgraph.dat -a s="EXISTS (s IN [Ps]([V]), [st](s) >= [dimm]/2 AND FORALL (a IN s, FORALL (b IN s, a == b OR EXISTS (e IN [E], a ELT e AND b ELT e))))" all
$PTH/flagcalc -r 10 20 10 -a s="FORALL (n IN [NN]([dimm]+1), [Knc](n,1) IFF EXISTS (s IN [Sizedsubset]([V],n), FORALL (b IN s, FORALL (a IN s, a == b OR EXISTS (e IN [E], a ELT e AND b ELT e)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 15 1 -a s="FORALL (s IN [Ps]([V]), s == [V] OR EXISTS (t IN [Ps]([V]), EXISTS (u IN [Ps]([V]), (NOT (u <= s)) AND (NOT (t <= s)) AND ((s CAP t) <= s))))" all
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (s IN [Ps]([V]), EXISTS (t IN [Ps]([V]), ((s CAP t) == [Nulls]) AND ((s CUP t) == [V]) AND FORALL (a IN s, FORALL (b IN s, NOT [ac](a,b))) AND FORALL (d IN t, FORALL (c IN t, NOT [ac](c,d)))))" all
$PTH/flagcalc -r 4 5 50 -a s="EXISTS (s IN [Ps]([Ps]([V])), s == [E])" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 2 -a s="FORALL (e IN [E], EXISTS (a IN [Sizedsubset]([V],2), FORALL (b IN a, FORALL (c IN a, (b ELT e) AND (c ELT e)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 2 -a s="FORALL (e IN [E], e ELT [Sizedsubset]([V],2))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (s IN [Ps]([Sizedsubset]([V],2)), FORALL (a IN s, FORALL (b IN a, FORALL (c IN a, EXISTS (e IN [E], c ELT e AND b ELT e)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (s IN [Ps]([Sizedsubset]([V],2)), FORALL (a IN s, a ELT [E]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 4 3 1 -a s="EXISTS (s IN [Ps]([Ps]([V])), [st](s) == 2^[dimm])" all
$PTH/flagcalc -r 4 3 1 -a s="FORALL (s IN [Ps]([Ps]([V])), EXISTS (t IN [Ps]([V]), t ELT s) OR [st](s) == 0)" all
$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s IN [Sizedsubset]([Sizedsubset]([V],2),[st]([E])), s == [E])" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s IN [Ps]([Sizedsubset]([V],2)), s == [E])" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip12.dat -a s="EXISTS (n IN [NN]([dimm]+1), EXISTS (l IN [Sizedsubset]([V],n), EXISTS (r IN [Sizedsubset]([V],[dimm] - n), [st](r CUP l) == [dimm] AND FORALL (a IN l, FORALL (b IN l, NOT [ac](a,b))) AND FORALL (c IN r, FORALL (d IN r, NOT [ac](c,d))))))" all
$PTH/flagcalc -r 4 3 10 -a s="FORALL (n IN [NN](2^[dimm]+1), EXISTS (s IN [Sizedsubset]([Ps]([V]),n), EXISTS (t IN [Sizedsubset]([Ps]([V]),2^[dimm] - n), ((s CUP t) == [Ps]([V])) AND ((s CAP t) == [Nulls]))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 20 100 -a s="FORALL (e IN [E], [ac]([idxt](e,0),[idxt](e,1)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 20 100 -a s="FORALL (n IN [NN]([st]([E])), [ac]([idxt]([idxs]([E],n),0),[idxt]([idxs]([E],n),1)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 12 10 -a s="FORALL (b IN [V], FORALL (a IN [V], [st]([Pathss](a,b)) == [pct](a,b)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 10 -a s="FORALL (b IN [V], FORALL (a IN [V], FORALL (p IN [Pathss](a,b), FORALL (m IN [NN]([st](p)-1), [ac]([idxt](p,m),[idxt](p,m+1))))))" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip8.dat -a s="FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), mod([st](c),2) == 0))" all
$PTH/flagcalc -r 8 10 1000 -a s="[cr1] IFF FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), [st](c) > 3))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 8 100 -a s="[treec] IFF FORALL (a IN [V], FORALL (b IN [V], [st]([Pathss](a,b)) == 1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 12 100 -a s="[forestc] IFF FORALL (a IN [V], FORALL (b IN [V], [st]([Pathss](a,b)) <= 1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 12 100 -a s="[forestc] IFF FORALL (a IN [V], [st]([Cyclesvs](a)) == 0)" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 6 100 -a s="[forestc] IFF FORALL (a IN [V], [st]([Cyclesvs](a)) == 0)" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 15 50 -a s="[conn1c] IFF FORALL (a IN [V], FORALL (b IN [V], [Pathss](a,b) != [Nulls]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), FORALL (w IN c, [TupletoSet](c) ELT [Cyclesvs](w))))" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (r IN [Ps]([V]), EXISTS (l IN [Ps]([V]), (l CUP r) == [V] AND [bipc](l,r)))" all
$PTH/flagcalc -r 9 10 100 -a s="(EXISTS (r IN [Ps]([V]), EXISTS (l IN [Ps]([V]), (l CUP r) == [V] AND [bipc](l,r)))) IFF FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), mod([st](c),2) == 0))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM (n IN [NN]([dimm]+1), [cyclet](n)) == SUM (v IN [V], SUM (c IN [Cyclesvs](v), 1/[st](c)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 1000 -a s="SUM (v IN [V], [vdt](v))/2 == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM (e IN [E], 1) == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 10 100 -a s="FORALL (v IN [V], [st]([Cyclesvs](v)) == [cyclesvt](v))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 1000 -a s="FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), SUM (a IN c, SUM (b IN c, [ac](b,a)))/2 >= [lt](c)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), FORALL (w IN c, EXISTS (d IN [Cyclesvs](w), [TupletoSet](d) == [TupletoSet](c)))))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="EXISTS (c IN [Cyclesvs](0), EXISTS (d IN [Cyclesvs](0), c != d AND [TupletoSet](c) == [TupletoSet](d)))" all
$PTH/flagcalc -d f="abc" -a s="EXISTS (c IN [Cyclesvs](0), EXISTS (d IN [Cyclesvs](0), c != d AND [TupletoSet](c) == [TupletoSet](d)))" all
$PTH/flagcalc -r 20 95 100 -a s="TALLY (v IN [V], [vdt](v))/2 == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 15 52.5 10000 -a s="COUNT (v IN [V], [vdt](v) % 2 == 1) % 2 == 0" all -v i=minimal3.cfg
$PTH/flagcalc -r 15 52.5 10000 -a i="COUNT (v IN [V], [vdt](v) % 2 == 1)" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 7.5 1 -a e="[Cyclesvs](0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a p="[idxs]([Pathss](0,1),0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcdef" -a p="[idxs]([Pathss](0,5),0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 22.5 1 -a e="[V] CUP [E]" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="[Pathss](0,2) CUP [Pathss](0,3)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="[V] CUP [E]" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="[Pathss](0,1) CUP [Pathss](0,2)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUPD (v IN [V], [Cyclesvs](v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -g o=out.dat all overwrite -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUPD (v IN [V], [Cyclesvs](v))" all -v set i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUP (v IN [V], [Cyclesvs](v))" all -v set i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUP (v IN [V], BIGCUP (c IN [Cyclesvs](v), [TupletoSet](c)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cyclesvs](v), [TupletoSet](c)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc d" -a e="SET (x IN [V], [st](SET (e IN [E], x ELT e, e)) > 0, x)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUPD (v IN [V], SET (c IN [Cyclesvs](v), [st](c) % 2 == 1, [TupletoSet](c)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUPD (v IN [V], SET (c IN [Cyclesvs](v), [st](c) % 2 == 0, [TupletoSet](c)))" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUP (v IN [V], SET (c IN [Cyclesvs](v), [st](c) % 2 == 0, [TupletoSet](c)))" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a a="SUM (v IN [V], [vdt](v) > [dimm]/2, 1)" all -v i=minimal3.cfg
$PTH/flagcalc -r 16 60 100 -a s="SUM (v IN [V], [vdt](v) > [dimm]/4, 1) == COUNT (v IN [V], [vdt](v) > [dimm]/4)" all -v i=minimal3.cfg
$PTH/flagcalc -r 16 60 1000 -a s="FORALL (v IN [V], [vdt](v) > 0, EXISTS (u IN [V], [ac](u,v)))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="SET (v IN [V], SET (u IN [V], SET (t IN [V], t)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="SETD (v IN [V], SETD (u IN [V], SETD (t IN [V], t)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cyclesvs](v), SET (e IN [E], EXISTS (a IN c, EXISTS (b IN c, a ELT e AND b ELT e AND a != b)), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cyclesvs](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), [idxt](c,n) ELT e AND [idxt](c,(n + 1) % [st](c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -a s="[cr1]" e2="BIGCUP (v IN [V], [Cyclesvs](v))" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cyclesvs](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), c(n) ELT e AND c((n+1)%[st](c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdea" -a e="BIGCUP (v IN [V], SET (c IN [Cyclesvs](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), c(n) ELT e AND c((n+1)%[st](c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="BIGCUP (v IN [V], SET (c IN [Cyclesvs](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), c(n) ELT e AND c((n+1)%[st](c)) ELT e), e)))" all -v set i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -a f="abcd" a2="[cliquem]>3" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e)))" all -v set i=minimal3.cfg
$PTH/flagcalc -r 4 3 1 -a s="FORALL (s IN Ps(Ps(V)), EXISTS (t IN Ps(V), t ELT s) OR st(s) == 0)" all
$PTH/flagcalc -d f="abc" -a e="SET (v IN V, SET (u IN V, SET (t IN V, t)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM (n IN NN(dimm+1), cyclet(n)) == st(BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e))))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a s="SUM (n IN NN(dimm+1), cyclet(n)) == st(BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e))))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="{3+2,SUM (v IN V, v), {1}}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a e="BIGCUP (v IN V, {v+1})" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a e="SET (v IN V, {v+1})" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a p="<<0,<<<<1>>,5,6>>,2,3>>" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=ef=ghi" -a e="{SET (v IN V, FORALL (c IN Cyclesvs(v), st(c) % 2 == 0), v)}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efghi" -a e="{SET (v IN V, FORALL (c IN Cyclesvs(v), (st(c) % 2) == 0), v)}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefdgha" -a e="Cyclesvs(0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefdgha" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="SET (es IN Ps(E), st(es) != 0, es(0)(0))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcdef" -a a="MAX (n IN NN(st(Cyclesvs(0)(0))), Cyclesvs(0)(0)(n))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a a="<<0,2, SUM (v IN Ps(V), st(v) > 0, st(Cyclesvs(v(0))))>>(1)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a a="<<0,2, SUM (v IN Ps(V), st(v) > 0, st(Cyclesvs(v(0))))>>(2)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a e="SETD (n IN st(V), SETD (i IN n, {i}))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="SET (s IN Setpartition(V), s)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 7 20 100 -a s="conn1c" s2="Knc(dimm,1)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg

# Diestel 5.2.4 (Brooks 1941)

$PTH/flagcalc -r 7 10.5 100 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cyclesvs(0)) == 1 IMPLIES FORALL (c IN Cyclesvs(0), st(c) % 2 == 0 AND st(c) == dimm)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 100 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cyclesvs(0)) == 1 IMPLIES FORALL (c IN Cyclesvs(0), st(c) % 2 == 0 AND st(c) == dimm)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 1000 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cyclesvs(0)) == 1 IMPLIES FORALL (c IN Cyclesvs(0), st(c) % 2 == 0  AND st(c) == dimm)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 100 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cyclesvs(0)) == 1 IMPLIES FORALL (c IN Cyclesvs(0), st(c) % 2 == 0  AND st(c) == dimm)" s3="Chit <= Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 200 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cyclesvs(0)) == 1 IMPLIES FORALL (c IN Cyclesvs(0), st(c) % 2 == 0  AND st(c) == dimm)" s3="Chigreedyt <= Deltam" all -v i=minimal3.cfg

# Diestel 5.2.1

$PTH/flagcalc -r 12 36 1000 -a s="Chit <= (1/2 + sqrt(2*edgecm + 1/4))" all -v i=minimal3.cfg
$PTH/flagcalc -r 17 68 1000 -a s="Chigreedyt <= (1/2 + sqrt(2*edgecm + 1/4))" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip12.dat -a p=Chip all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefga" -a p=Chigreedyp all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefga" -a a=Chigreedyt all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 17 68 100000 -a a="(1/2 + sqrt(2*edgecm + 1/4)) - Chigreedyt" all -v i=minimal3.cfg
$PTH/flagcalc -r 17 68 1000 -a a="(1/2 + sqrt(2*edgecm + 1/4)) - Chit" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 1000 -a s="MIN (p IN Setpartition(V), FORALL (s IN p, FORALL (v1 IN s, FORALL (v2 IN s, NOT ac(v1,v2)))), st(p)) <= Chigreedyt" all -v i=minimal3.cfg

$PTH/flagcalc -r 17 68 1000 -a s="Chit <= Chigreedyt" all -v i=minimal3.cfg

# Diestel 5.2.5 (Erdos)

$PTH/flagcalc -r 32 60 100000 -a s="girthm > 3 AND Chigreedyt > 3" all -v i=minimal3.cfg
$PTH/flagcalc -r 38 50 100000 -a s="girthm > 4 AND Chigreedyt > 4" all -v i=minimal3.cfg

# Diestel 1.3.1

$PTH/flagcalc -r 12 30 100 -a s="deltam >= 2" s2="MAX (v IN V, MAX (c IN Cyclesvs(v), st(c))) >= deltam + 1" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -a s="deltam >= 2" s2="EXISTS (v IN V, EXISTS (c IN Cyclesvs(v), st(c) >= deltam + 1))" all -v i=minimal3.cfg

# Diestel 1.3.1

$PTH/flagcalc -r 12 30 100 -a s="MAX (v1 IN V, MAX (v2 IN V, MAX (p IN Pathss(v1,v2), st(p)))) >= deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -a s="EXISTS (v1 IN V, EXISTS (v2 IN V, EXISTS (p IN Pathss(v1,v2), st(p) >= deltam)))" all -v i=minimal3.cfg

# Diestel 5.3.2 (Vizing 1964)

$PTH/flagcalc -r 8 14 100 -a s="Deltam <= Chiprimet && Chiprimet <= Deltam + 1" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abc=defg" f="abcd=efgh" f="abc=de" -a s="FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), mod([st](c),2) == 0))" s2="Chiprimet == Deltam" all -v i=minimal3.cfg

# Diestel 5.3.1 (Konig 1916)

$PTH/flagcalc -r 8 14 1000 -a s="FORALL (v IN [V], FORALL (c IN [Cyclesvs](v), mod([st](c),2) == 0))" s2="Chiprimet == Deltam" all -v i=minimal3.cfg

# Diestel 2.1.1

$PTH/flagcalc -r 8 14 200 -a s="EXISTS (p IN Setpartition(V), st(p) == 2, bipc(p(0),p(1)))" s2="MAX (es IN Ps(E), FORALL (e1 IN es, FORALL (e2 IN es, NOT eadjc(e1,e2))), st(es)) == MIN (vs IN Ps(V), FORALL (e IN E, vs CAP e != Nulls), st(vs))" all -v i=minimal3.cfg

# Diestel 2.1.2 (Hall 1935)

# $PTH/flagcalc -d testbip8.dat -a s="FORALL (p IN Setpartition(V), (bipc(p(0),p(1)) && st(p) == 2) IMPLIES (EXISTS (es IN Ps(E), FORALL (e1 IN es, FORALL (e2 IN es, NOT eadjc(e1,e2))) AND FORALL (a IN p(0), EXISTS (e IN es, a ELT e))) IFF FORALL (S IN Ps(p(0)), Nt(S) >= st(S))))" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 14 2000 -a s="FORALL (p IN Setpartition(V), (EXISTS (es IN Ps(E), FORALL (e1 IN es, FORALL (e2 IN es, NOT eadjc(e1,e2))) AND FORALL (a IN p(0), EXISTS (e IN es, a ELT e))) IFF FORALL (S IN Ps(p(0)), Nt(S) >= st(S))) IF (bipc(p(0),p(1)) && st(p) == 2))" all -v i=minimal3.cfg

# Diestel 5.3.3 (Csaba et al 2016)

$PTH/flagcalc -r 5 4 10 -r 6 7.5 10 -r 7 10.5 10 -r 8 14 10 -a s="FORALL (n IN NN(9), FORALL (d IN NN(9), Chiprimet == Deltam IF (FORALL (v IN V, (vdt(v) == d) AND dimm == n AND d >= n/2 AND n >= 4 AND n % 2 == 0))))" all -v i=minimal3.cfg

$PTH/flagcalc -r 6 7.5 100 -a z="Chiprimegreedyt - Chiprimet" all -v i=minimal3.cfg

# Diestel Conjecture p. 184 (Hadwiger 1943), Corollary 7.3.3

$PTH/flagcalc -r 8 14 10000 -a s="Chit >= 5" s2="EXISTS (so IN Ps(V), EXISTS (p IN Setpartition(so), st(p) >= 5, FORALL (q IN p, FORALL (r2 IN q, FORALL (s2 IN q, r2 != s2, EXISTS (z IN Pathss(r2,s2), z <= q)))) AND FORALL (s IN p, FORALL (t IN p, s != t, EXISTS (u IN s, EXISTS (v IN t, ac(u,v)))))))" all -v i=minimal3.cfg

# Diestel Corollary 7.3.2

$PTH/flagcalc -r 8 14 300 -a s="FORALL (vs IN Ps(V), NOT EXISTS (o IN Setpartition(vs), st(o) == 4, FORALL (p IN o, FORALL (q IN p, FORALL (r IN p, q != r, EXISTS (z IN Pathss(q,r), z <= p)))) AND FORALL (p1 IN o, FORALL (q1 IN o, q1 != p1, EXISTS (r1 IN p1, EXISTS (s1 IN q1, ac(r1,s1)))))))" \
s2="FORALL (v1 IN V, FORALL (v2 IN V, v1 != v2 AND NOT ac(v1,v2), EXISTS (vs IN Ps(V), EXISTS (m IN Setpartition(vs), st(m) == 4, FORALL (p IN m, FORALL (q IN p, FORALL (r IN p, EXISTS (z IN Pathss(q,r), z <= p))) OR (v1 ELT p AND v2 ELT p AND FORALL (u IN p, FORALL (v IN p, (EXISTS (z IN Pathss(u,v1), z <= p) AND EXISTS (z2 IN Pathss(v2,v), z2 <= p)) OR (EXISTS (z3 IN Pathss(v,v1), z3 <= p) AND EXISTS (z4 IN Pathss(u,v2), z4 <= p)))))) AND FORALL (p2 IN m, FORALL (q2 IN m, p2 != q2, EXISTS (r2 IN p2, EXISTS (s2 IN q2, ac(r2,s2))) OR (v1 ELT p2 AND v2 ELT q2) OR (v2 ELT p2 AND v1 ELT q2) ))))))" \
z3="edgecm == 2*dimm - 3" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 14 300 -a s="FORALL (vs IN Ps(V), NOT EXISTS (o IN Setpartition(vs), st(o) == 4, FORALL (p IN o, FORALL (q IN p, FORALL (r IN p, q != r, EXISTS (z IN Pathss(q,r), z <= p)))) AND FORALL (p1 IN o, FORALL (q1 IN o, q1 != p1, Nssc(p1,q1)))))" \
s2="FORALL (ne IN nE, EXISTS (vs IN Ps(V), EXISTS (m IN Setpartition(vs), st(m) == 4, FORALL (p IN m, FORALL (q IN p, FORALL (r IN p, EXISTS (z IN Pathss(q,r), z <= p))) OR (ne(0) ELT p AND ne(1) ELT p AND FORALL (u IN p, FORALL (v IN p, (EXISTS (z IN Pathss(u,ne(0)), z <= p) AND EXISTS (z2 IN Pathss(ne(1),v), z2 <= p)) OR (EXISTS (z3 IN Pathss(v,ne(0)), z3 <= p) AND EXISTS (z4 IN Pathss(u,ne(1)), z4 <= p)))))) AND FORALL (p2 IN m, FORALL (q2 IN m, p2 != q2, Nssc(p2,q2) OR (ne(0) ELT p2 AND ne(1) ELT q2) OR (ne(1) ELT p2 AND ne(0) ELT q2) )) )))" \
z3="edgecm == 2*dimm - 3" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 14 1500 -a s="FORALL (vs IN Ps(V), NOT EXISTS (o IN Setpartition(vs), st(o) == 4, FORALL (p IN o, connvsc(p,E)) AND FORALL (p1 IN o, FORALL (q1 IN o, q1 != p1, Nssc(p1,q1)))))" \
s2="FORALL (ne IN nE, EXISTS (vs IN Ps(V), EXISTS (m IN Setpartition(vs), st(m) == 4, FORALL (p IN m, connvsc(p, E CUP {ne})) AND FORALL (p2 IN m, FORALL (q2 IN m, p2 != q2, Nssc(p2,q2) OR (ne(0) ELT p2 AND ne(1) ELT q2) OR (ne(1) ELT p2 AND ne(0) ELT q2) )) )))" \
z3="edgecm == 2*dimm - 3" all -v i=minimal3.cfg


# Diestel Theorem 3.3.6(i) (Global version of Menger's Theorem) Note that the native kconnc is much faster than the count of independent paths, and here that speedup is a matter of the native code being to the left of the IFF

$PTH/flagcalc -r 5 5 300 -a s="FORALL (k IN dimm, kconnc(k) IFF FORALL (v1 IN V, FORALL (v2 IN V, v1 != v2, EXISTS (ps IN Ps(Pathss(v1,v2)), st(ps) >= k, FORALL (p1 IN ps, FORALL (p2 IN ps, p1 != p2, FORALL (v3 IN p1, v3 != v1 AND v3 != v2, FORALL (v4 IN p2, v4 != v1 AND v4 != v2, v3 != v4))))))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 300 -a s="FORALL (k IN dimm, kconnc(k) IFF FORALL (v1 IN V, FORALL (v2 IN V, v1 != v2, EXISTS (ps IN Ps(Pathss(v1,v2)), st(ps) >= k, FORALL (p1 IN ps, FORALL (p2 IN ps, p1 != p2, p1 CAP p2 <= {v1,v2}))))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 300 -a s="FORALL (k IN dimm, FORALL (v1 IN V, FORALL (v2 IN V, v1 != v2, EXISTS (ps IN Ps(Pathss(v1,v2)), st(ps) >= k, FORALL (p1 IN ps, FORALL (p2 IN ps, p1 != p2, p1 CAP p2 <= {v1,v2}))))) IFF kconnc(k))" all -v i=minimal3.cfg

# Diestel Theorem 3.3.1 (Menger 1927)

$PTH/flagcalc -r 5 5 50 -a isp=storedprocedures.dat s="FORALL (A IN Ps(V), FORALL  (B IN Ps(V), MIN (X IN Ps(V), Separatesc(A,B,X), st(X)) == DisjointABpaths( A, B ) ))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 50 -a isp=storedprocedures.dat s="FORALL (A IN Ps(V), FORALL  (B IN Ps(V), MinXSeparates(A,B) == DisjointABpaths( A, B ) ))" all -v i=minimal3.cfg


# Diestel Theorem 7.4.1 (Regularity lemma), specifically "admits an epsilon-regular partition with partition size k <= M"

# $PTH/flagcalc -r 8 14 300 -a z="EXISTS (vp IN Setpartition(V), st(vp(0)) <= 0.1*st(vp(1)) AND st(vp) <= 5 AND FORALL (n IN st(vp)-1, n > 0, vp(n) == vp(n+1)), FORALL (i IN st(vp), FORALL (j IN st(vp), i != j, Nsst(vp(i),vp(j))/(st(vp(i)*st(vp(j)) "

# Diestel Exercise 1.7 (p 10)

$PTH/flagcalc -r 8 14 1000 -a isp=storedprocedures.dat s="n_0(deltam, girthm) <= dimm" all -v i=minimal3.cfg

# Diestel Theorem 1.3.4 (Alon et al 2002)

$PTH/flagcalc -r 8 14 100 -a isp=storedprocedures.dat s="NOT isinf(girthm)" s2="FORALL (d IN dm + 1, d >= 2, FORALL (g IN girthm, n_0(d,g) <= dimm ))" all -v i=minimal3.cfg

# Diestel Prop 1.4.2 (p. 12)

$PTH/flagcalc -d octahedron.dat -a z="lambdat" z="kappat" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 10000 -a s="dimm>1" s2="kappat <= lambdat && lambdat <= deltam" all -v i=minimal3.cfg

# Diestel Theorem 1.4.3 (p. 13) (Mader 1972)

$PTH/flagcalc -r 12 50 1000 -a s="FORALL (k IN dimm, k > 0 AND dm >= 4*k, EXISTS (U IN Ps(V), kconnc(SubgraphonUg(U),k+1) AND dm(SubgraphonUg(U))/2 > dm/2 - k))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 50 1000 -a s="FORALL (k IN dimm, k > 0 AND dm >= 4*k, EXISTS (U IN Ps(V), NAMING (SU AS SubgraphonUg(U), kconnc(SU,k+1) AND dm(SU)/2 > dm/2 - k)))" all -v i=minimal3.cfg

$PTH/flagcalc -r 12 30.5 100 -a a="st(Cyclesvs(0)) + st(Cyclesvs(0)) + st(Cyclesvs(0))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 30.5 100 -a a="NAMING (C AS Cyclesvs(0), st(C) + st(C) + st(C))" all -v i=minimal3.cfg


# Diestel Proposition 1.4.1 (p 10)

$PTH/flagcalc -r 8 14 1000 -a s1="conn1c" s2="EXISTS (p IN Perms(V), FORALL (n IN dimm+1, conn1c(SubgraphonUg(Sp(p,n)))))" all -v i=minimal3.cfg

$PTH/flagcalc -r 30 100 10000 -a s="connm == st(Componentss)" all -v i=minimal3.cfg
$PTH/flagcalc -r 30 100 10000 -a a="connm" all -v i=minimal3.cfg
$PTH/flagcalc -r 30 100 10000 -a a="st(Componentss)" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 10 1000 -a s="MAX (p IN Setpartition(V), FORALL (a IN p, FORALL (b IN p, a != b, NOT connvssc(a,b))), st(p)) == connm" all -v i=minimal3.cfg

$PTH/flagcalc -r 10 10 1 -a e="Componentss(Gg(\"abc de fgh ijkl\"))" all -v set allsets i=minimal3.cfg

# Diestel Theorem 3.4.1 (Mader 1978)

# $PTH/flagcalc -r 6 7.5 10 -a isp=storedprocedures.dat a="MAX (H IN Ps(V), M_H(H))" all -v i=minimal3.cfg
# $PTH/flagcalc -r 6 7.5 10 -a isp=storedprocedures.dat s="\
# FORALL (H IN Ps(V), NAMING (MHH AS M_H(H), NAMING (VminusH AS V SETMINUS H, EXISTS (p IN Setpartition(VminusH), st(p) >= MHH, \
#  FORALL (s IN p, EXISTS (v1 IN H, EXISTS (v2 IN H, v1 != v2, NAMING (v1v2 AS Pathss(v1,v2), EXISTS (path IN v1v2, path <= (s CUPD {v1,v2}) ))))) ))))" all -v i=minimal3.cfg
