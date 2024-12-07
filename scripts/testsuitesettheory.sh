PTH='../cmake-build-debug'
$PTH/flagcalc -r 9 2 50 -a s="EXISTS (t IN [P]([V]), EXISTS (s IN [P]([V]), (s CUP t) == [V] AND [st](s) == ceil([dimm]/2) AND [st](t) == floor([dimm]/2)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a s="FORALL (n IN [NN]([dimm]), EXISTS (s IN [P]([V]), EXISTS (t IN [P]([V]), (s CUP t) == [V] AND [st](s) == ceil([dimm]/(n+1)) AND [st](t) == floor(n*[dimm]/(n+1)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 10 100 -a s="FORALL (e IN [E], EXISTS (a IN [V], EXISTS (b IN [V], a ELT e AND b ELT e AND a != b)))" all -v i=minimal3.cfg
$PTH/flagcalc -d halfconnectedgraph.dat -a s="EXISTS (s IN [P]([V]), [st](s) >= [dimm]/2 AND FORALL (a IN s, FORALL (b IN s, a == b OR EXISTS (e IN [E], a ELT e AND b ELT e))))" all
$PTH/flagcalc -r 10 20 10 -a s="FORALL (n IN [NN]([dimm]+1), [Knc](n,1) IFF EXISTS (s IN [Sizedsubset]([V],n), FORALL (b IN s, FORALL (a IN s, a == b OR EXISTS (e IN [E], a ELT e AND b ELT e)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 15 1 -a s="FORALL (s IN [P]([V]), s == [V] OR EXISTS (t IN [P]([V]), EXISTS (u IN [P]([V]), (NOT (u <= s)) AND (NOT (t <= s)) AND ((s CAP t) <= s))))" all
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (s IN [P]([V]), EXISTS (t IN [P]([V]), ((s CAP t) == [Nulls]) AND ((s CUP t) == [V]) AND FORALL (a IN s, FORALL (b IN s, NOT [ac](a,b))) AND FORALL (d IN t, FORALL (c IN t, NOT [ac](c,d)))))" all
$PTH/flagcalc -r 4 5 50 -a s="EXISTS (s IN [P]([P]([V])), s == [E])" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 2 -a s="FORALL (e IN [E], EXISTS (a IN [Sizedsubset]([V],2), FORALL (b IN a, FORALL (c IN a, (b ELT e) AND (c ELT e)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 2 -a s="FORALL (e IN [E], e ELT [Sizedsubset]([V],2))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (s IN [P]([Sizedsubset]([V],2)), FORALL (a IN s, FORALL (b IN a, FORALL (c IN a, EXISTS (e IN [E], c ELT e AND b ELT e)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (s IN [P]([Sizedsubset]([V],2)), FORALL (a IN s, a ELT [E]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 4 3 1 -a s="EXISTS (s IN [P]([P]([V])), [st](s) == 2^[dimm])" all
$PTH/flagcalc -r 4 3 1 -a s="FORALL (s IN [P]([P]([V])), EXISTS (t IN [P]([V]), t ELT s) OR [st](s) == 0)" all
$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s IN [Sizedsubset]([Sizedsubset]([V],2),[st]([E])), s == [E])" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s IN [P]([Sizedsubset]([V],2)), s == [E])" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip12.dat -a s="EXISTS (n IN [NN]([dimm]+1), EXISTS (l IN [Sizedsubset]([V],n), EXISTS (r IN [Sizedsubset]([V],[dimm] - n), [st](r CUP l) == [dimm] AND FORALL (a IN l, FORALL (b IN l, NOT [ac](a,b))) AND FORALL (c IN r, FORALL (d IN r, NOT [ac](c,d))))))" all
$PTH/flagcalc -r 4 3 10 -a s="FORALL (n IN [NN](2^[dimm]+1), EXISTS (s IN [Sizedsubset]([P]([V]),n), EXISTS (t IN [Sizedsubset]([P]([V]),2^[dimm] - n), ((s CUP t) == [P]([V])) AND ((s CAP t) == [Nulls]))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 20 100 -a s="FORALL (e IN [E], [ac]([idxt](e,0),[idxt](e,1)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 20 100 -a s="FORALL (n IN [NN]([st]([E])), [ac]([idxt]([idxs]([E],n),0),[idxt]([idxs]([E],n),1)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 12 10 -a s="FORALL (b IN [V], FORALL (a IN [V], [st]([Pathss](a,b)) == [pct](a,b)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 10 -a s="FORALL (b IN [V], FORALL (a IN [V], FORALL (p IN [Pathss](a,b), FORALL (m IN [NN]([st](p)-1), [ac]([idxt](p,m),[idxt](p,m+1))))))" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip8.dat -a s="FORALL (v IN [V], FORALL (c IN [Cycless](v), mod([st](c),2) == 0))" all
$PTH/flagcalc -r 8 10 1000 -a s="[cr1] IFF FORALL (v IN [V], FORALL (c IN [Cycless](v), [st](c) > 3))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 8 100 -a s="[treec] IFF FORALL (a IN [V], FORALL (b IN [V], [st]([Pathss](a,b)) == 1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 12 100 -a s="[forestc] IFF FORALL (a IN [V], FORALL (b IN [V], [st]([Pathss](a,b)) <= 1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 12 100 -a s="[forestc] IFF FORALL (a IN [V], [st]([Cycless](a)) == 0)" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 6 100 -a s="[forestc] IFF FORALL (a IN [V], [st]([Cycless](a)) == 0)" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 15 50 -a s="[conn1c] IFF FORALL (a IN [V], FORALL (b IN [V], [Pathss](a,b) != [Nulls]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (v IN [V], FORALL (c IN [Cycless](v), FORALL (w IN c, [TupletoSet](c) ELT [Cycless](w))))" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (r IN [P]([V]), EXISTS (l IN [P]([V]), (l CUP r) == [V] AND [bipc](l,r)))" all
$PTH/flagcalc -r 9 10 100 -a s="(EXISTS (r IN [P]([V]), EXISTS (l IN [P]([V]), (l CUP r) == [V] AND [bipc](l,r)))) IFF FORALL (v IN [V], FORALL (c IN [Cycless](v), mod([st](c),2) == 0))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM (n IN [NN]([dimm]+1), [cyclet](n)) == SUM (v IN [V], SUM (c IN [Cycless](v), 1/[st](c)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 1000 -a s="SUM (v IN [V], [vdt](v))/2 == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM (e IN [E], 1) == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 10 100 -a s="FORALL (v IN [V], [st]([Cycless](v)) == [cyclest](v))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 1000 -a s="FORALL (v IN [V], FORALL (c IN [Cycless](v), SUM (a IN c, SUM (b IN c, [ac](b,a)))/2 >= [lt](c)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (v IN [V], FORALL (c IN [Cycless](v), FORALL (w IN c, EXISTS (d IN [Cycless](w), [TupletoSet](d) == [TupletoSet](c)))))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="EXISTS (c IN [Cycless](0), EXISTS (d IN [Cycless](0), c != d AND [TupletoSet](c) == [TupletoSet](d)))" all
$PTH/flagcalc -d f="abc" -a s="EXISTS (c IN [Cycless](0), EXISTS (d IN [Cycless](0), c != d AND [TupletoSet](c) == [TupletoSet](d)))" all
$PTH/flagcalc -r 20 95 100 -a s="TALLY (v IN [V], [vdt](v))/2 == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 15 52.5 10000 -a s="COUNT (v IN [V], [vdt](v) % 2 == 1) % 2 == 0" all -v i=minimal3.cfg
$PTH/flagcalc -r 15 52.5 10000 -a i="COUNT (v IN [V], [vdt](v) % 2 == 1)" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 7.5 1 -a e="[Cycless](0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a p="[idxs]([Pathss](0,1),0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcdef" -a p="[idxs]([Pathss](0,5),0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 22.5 1 -a e="[V] CUP [E]" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="[Pathss](0,2) CUP [Pathss](0,3)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="[V] CUP [E]" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="[Pathss](0,1) CUP [Pathss](0,2)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUPD (v IN [V], [Cycless](v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -g o=out.dat all overwrite -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUPD (v IN [V], [Cycless](v))" all -v set i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUP (v IN [V], [Cycless](v))" all -v set i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUP (v IN [V], BIGCUP (c IN [Cycless](v), [TupletoSet](c)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cycless](v), [TupletoSet](c)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc d" -a e="SET (x IN [V], [st](SET (e IN [E], x ELT e, e)) > 0, x)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUPD (v IN [V], SET (c IN [Cycless](v), [st](c) % 2 == 1, [TupletoSet](c)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUPD (v IN [V], SET (c IN [Cycless](v), [st](c) % 2 == 0, [TupletoSet](c)))" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUP (v IN [V], SET (c IN [Cycless](v), [st](c) % 2 == 0, [TupletoSet](c)))" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a a="SUM (v IN [V], [vdt](v) > [dimm]/2, 1)" all -v i=minimal3.cfg
$PTH/flagcalc -r 16 60 100 -a s="SUM (v IN [V], [vdt](v) > [dimm]/4, 1) == COUNT (v IN [V], [vdt](v) > [dimm]/4)" all -v i=minimal3.cfg
$PTH/flagcalc -r 16 60 1000 -a s="FORALL (v IN [V], [vdt](v) > 0, EXISTS (u IN [V], [ac](u,v)))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="SET (v IN [V], SET (u IN [V], SET (t IN [V], t)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="SETD (v IN [V], SETD (u IN [V], SETD (t IN [V], t)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cycless](v), SET (e IN [E], EXISTS (a IN c, EXISTS (b IN c, a ELT e AND b ELT e AND a != b)), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cycless](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), [idxt](c,n) ELT e AND [idxt](c,(n + 1) % [st](c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -a s="[cr1]" e2="BIGCUP (v IN [V], [Cycless](v))" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN [V], SET (c IN [Cycless](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), c(n) ELT e AND c((n+1)%[st](c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdea" -a e="BIGCUP (v IN [V], SET (c IN [Cycless](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), c(n) ELT e AND c((n+1)%[st](c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="BIGCUP (v IN [V], SET (c IN [Cycless](v), SET (e IN [E], EXISTS (n IN [NN]([st](c)), c(n) ELT e AND c((n+1)%[st](c)) ELT e), e)))" all -v set i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -a f="abcd" a2="[cliquem]>3" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="BIGCUP (v IN V, SET (c IN Cycless(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e)))" all -v set i=minimal3.cfg
$PTH/flagcalc -r 4 3 1 -a s="FORALL (s IN P(P(V)), EXISTS (t IN P(V), t ELT s) OR st(s) == 0)" all
$PTH/flagcalc -d f="abc" -a e="SET (v IN V, SET (u IN V, SET (t IN V, t)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM (n IN NN(dimm+1), cyclet(n)) == st(BIGCUP (v IN V, SET (c IN Cycless(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e))))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a s="SUM (n IN NN(dimm+1), cyclet(n)) == st(BIGCUP (v IN V, SET (c IN Cycless(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e))))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="{3+2,SUM (v IN V, v), {1}}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a e="BIGCUP (v IN V, {v+1})" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a e="SET (v IN V, {v+1})" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a p="<<0,<<<<1>>,5,6>>,2,3>>" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=ef=ghi" -a e="{SET (v IN V, FORALL (c IN Cycless(v), st(c) % 2 == 0), v)}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efghi" -a e="{SET (v IN V, FORALL (c IN Cycless(v), (st(c) % 2) == 0), v)}" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefdgha" -a e="Cycless(0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefdgha" -a e="BIGCUP (v IN V, SET (c IN Cycless(v), SET (e IN E, EXISTS (n IN NN(st(c)), c(n) ELT e AND c((n+1)%st(c)) ELT e), e)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="SET (es IN P(E), st(es) != 0, es(0)(0))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcdef" -a a="MAX (n IN NN(st(Cycless(0)(0))), Cycless(0)(0)(n))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a p="<<0,2, SUM (v IN P(V), st(v) > 0, st(Cycless(v(0))))>>(1)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a p="<<0,2, SUM (v IN P(V), st(v) > 0, st(Cycless(v(0))))>>(2)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc=def=ghi" -a e="SETD (n IN st(V), SETD (i IN n, {i}))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcdefg" -a e="SET (s IN Setpartition(V), s)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 7 20 100 -a s="conn1c" s2="Knc(dimm,1)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 100 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cycless(0)) == 1 IMPLIES FORALL (c IN Cycless(0), st(c) % 2 == 0 AND st(c) == dimm)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 100 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cycless(0)) == 1 IMPLIES FORALL (c IN Cycless(0), st(c) % 2 == 0 AND st(c) == dimm)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 1000 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cycless(0)) == 1 IMPLIES FORALL (c IN Cycless(0), st(c) % 2 == 0  AND st(c) == dimm)" s3="EXISTS (c IN Setpartition(V), st(c) == Deltam AND FORALL (s IN c, FORALL (u1 IN s, FORALL (u2 IN s, NOT ac(u1,u2)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 100 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cycless(0)) == 1 IMPLIES FORALL (c IN Cycless(0), st(c) % 2 == 0  AND st(c) == dimm)" s3="Chit <= Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 200 -a s="conn1c" s2="NOT Knc(dimm,1) AND st(Cycless(0)) == 1 IMPLIES FORALL (c IN Cycless(0), st(c) % 2 == 0  AND st(c) == dimm)" s3="Chigreedyt <= Deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 17 68 100000 -a s="Chit <= (1/2 + sqrt(2*edgecm + 1/4))" all -v i=minimal3.cfg
$PTH/flagcalc -r 17 68 100000 -a s="Chigreedyt <= (1/2 + sqrt(2*edgecm + 1/4))" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip12.dat -a p=Chip all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefga" -a p=Chigreedyp all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdefga" -a a=Chigreedyt all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 17 68 100000 -a a="(1/2 + sqrt(2*edgecm + 1/4)) - Chigreedyt" all -v i=minimal3.cfg
$PTH/flagcalc -r 17 68 100000 -a a="(1/2 + sqrt(2*edgecm + 1/4)) - Chit" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 1000 -a s="MIN (p IN Setpartition(V), FORALL (s IN p, FORALL (v1 IN s, FORALL (v2 IN s, NOT ac(v1,v2)))), st(p)) <= Chigreedyt" all -v i=minimal3.cfg
$PTH/flagcalc -r 17 68 10000 -a s="Chit <= Chigreedyt" all -v i=minimal3.cfg
$PTH/flagcalc -r 32 60 100000 -a s="girthm > 3 AND Chigreedyt > 3" all -v i=minimal3.cfg
$PTH/flagcalc -r 38 50 1000000 -a s="girthm > 4 AND Chigreedyt > 4" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -a s="deltam >= 2" s2="MAX (v IN V, MAX (c IN Cycless(v), st(c))) >= deltam + 1" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -a s="deltam >= 2" s2="EXISTS (v IN V, EXISTS (c IN Cycless(v), st(c) >= deltam + 1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -a s="MAX (v1 IN V, MAX (v2 IN V, MAX (p IN Pathss(v1,v2), st(p)))) >= deltam" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -a s="EXISTS (v1 IN V, EXISTS (v2 IN V, EXISTS (p IN Pathss(v1,v2), st(p) >= deltam)))" all -v i=minimal3.cfg
