PTH='../cmake-build-debug'
$PTH/flagcalc -r 9 2 50 -a s="EXISTS (EXISTS ((s CUP t) == [V] AND [st](s) == ceil([dimm]/2) AND [st](t) == floor([dimm]/2), t IN [P]([V])), s IN [P]([V]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 10 1 -a s="FORALL (EXISTS (EXISTS ((s CUP t) == [V] AND [st](s) == ceil([dimm]/(n+1)) AND [st](t) == floor(n*[dimm]/(n+1)), t IN [P]([V])), s IN [P]([V])), n IN [NN]([dimm]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 10 100 -a s="FORALL (EXISTS (EXISTS (a ELT e AND b ELT e AND a != b, b IN [V]), a IN [V]), e IN [E])" all -v i=minimal3.cfg
$PTH/flagcalc -d halfconnectedgraph.dat -a s="EXISTS ([st](s) >= [dimm]/2 AND FORALL (FORALL (a == b OR EXISTS (a ELT e AND b ELT e, e IN [E]), b IN s), a IN s), s IN [P]([V]))" all
$PTH/flagcalc -r 10 20 10 -a s="FORALL ([Knc](n,1) IFF EXISTS (FORALL (FORALL (a == b OR EXISTS (a ELT e AND b ELT e, e IN [E]) , a IN s), b IN s), s IN [Sizedsubset]([V],n)), n IN [NN]([dimm]+1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 15 1 -a s="FORALL (s == [V] OR EXISTS (EXISTS ((NOT (u <= s)) AND (NOT (t <= s)) AND ((s CAP t) <= s), u IN [P]([V])), t IN [P]([V])), s IN [P]([V]))" all
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (EXISTS (((s CAP t) == [Nulls]) AND ((s CUP t) == [V]) AND FORALL (FORALL (NOT [ac](a,b), b IN s), a IN s) AND FORALL (FORALL (NOT [ac](c,d), d IN t), c IN t), t IN [P]([V])), s IN [P]([V]))" all
$PTH/flagcalc -r 4 5 50 -a s="EXISTS (s == [E], s IN [P]([P]([V])))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 2 -a s="FORALL (EXISTS (FORALL (FORALL ((b ELT e) AND (c ELT e), c IN a), b IN a), a IN [Sizedsubset]([V],2)), e IN [E])" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 2 -a s="FORALL (e ELT [Sizedsubset]([V],2), e IN [E])" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (FORALL (FORALL (FORALL (EXISTS (c ELT e AND b ELT e, e IN [E]), c IN a), b IN a), a IN s), s IN [P]([Sizedsubset]([V],2)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (FORALL (a ELT [E], a IN s), s IN [P]([Sizedsubset]([V],2)))" all -v i=minimal3.cfg
$PTH/flagcalc -r 4 3 1 -a s="EXISTS ([st](s) == 2^[dimm], s IN [P]([P]([V])))" all
$PTH/flagcalc -r 4 3 1 -a s="FORALL (EXISTS (t ELT s, t IN [P]([V])) OR [st](s) == 0, s IN [P]([P]([V])))" all
$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s == [E], s IN [Sizedsubset]([Sizedsubset]([V],2),[st]([E])))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s == [E], s IN [P]([Sizedsubset]([V],2)))" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip12.dat -a s="EXISTS (EXISTS (EXISTS ([st](r CUP l) == [dimm] AND FORALL (FORALL (NOT [ac](a,b), b IN l), a IN l) AND FORALL (FORALL (NOT [ac](c,d), d IN r), c IN r), r IN [Sizedsubset]([V],[dimm] - n)), l IN [Sizedsubset]([V],n)) , n IN [NN]([dimm]+1))" all
$PTH/flagcalc -r 4 3 10 -a s="FORALL (EXISTS (EXISTS (((s CUP t) == [P]([V])) AND ((s CAP t) == [Nulls]), t IN [Sizedsubset]([P]([V]),2^[dimm] - n)), s IN [Sizedsubset]([P]([V]),n)), n IN [NN](2^[dimm]+1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 20 100 -a s="FORALL ([ac]([idxt](e,0),[idxt](e,1)), e IN [E])" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 20 100 -a s="FORALL ([ac]([idxt]([idxs]([E],n),0),[idxt]([idxs]([E],n),1)), n IN [NN]([st]([E])))" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 12 10 -a s="FORALL (FORALL ([st]([Pathss](a,b)) == [pct](a,b), a IN [V]), b IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 10 10 -a s="FORALL (FORALL (FORALL (FORALL ([ac]([idxt](p,m),[idxt](p,m+1)), m IN [NN]([st](p)-1)), p IN [Pathss](a,b)), a IN [V]), b IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip8.dat -a s="FORALL (FORALL (mod([st](c),2) == 0, c IN [Cycless](v)), v IN [V])" all
$PTH/flagcalc -r 8 10 1000 -a s="[cr1] IFF FORALL (FORALL ([st](c) > 3, c IN [Cycless](v)), v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 8 100 -a s="[treec] IFF FORALL (FORALL ([st]([Pathss](a,b)) == 1, b IN [V]), a IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 12 100 -a s="[forestc] IFF FORALL (FORALL ([st]([Pathss](a,b)) <= 1, b IN [V]), a IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 12 100 -a s="[forestc] IFF FORALL ([st]([Cycless](a)) == 0, a IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 6 100 -a s="[forestc] IFF FORALL ([st]([Cycless](a)) == 0, a IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 15 50 -a s="[conn1c] IFF FORALL (FORALL ([Pathss](a,b) != [Nulls], b IN [V]), a IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (FORALL (FORALL ([TupletoSet](c) ELT [Cycless](w), w IN c), c IN [Cycless](v)), v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (EXISTS ((l CUP r) == [V] AND [bipc](l,r), l IN [P]([V])), r IN [P]([V]))" all
$PTH/flagcalc -r 9 10 100 -a s="(EXISTS (EXISTS ((l CUP r) == [V] AND [bipc](l,r), l IN [P]([V])), r IN [P]([V]))) IFF FORALL (FORALL (mod([st](c),2) == 0, c IN [Cycless](v)), v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM ([cyclet](n), n IN [NN]([dimm]+1)) == SUM (SUM (1/[st](c), c IN [Cycless](v)), v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 1000 -a s="SUM ([vdt](v), v IN [V])/2 == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 100 -a s="SUM (1, e IN [E]) == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 10 100 -a s="FORALL ([st]([Cycless](v)) == [cyclest](v), v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10 1000 -a s="FORALL (FORALL (SUM (SUM ([ac](b,a), b IN c), a IN c)/2 >= [lt](c), c IN [Cycless](v)), v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (FORALL (FORALL (EXISTS ([TupletoSet](d) == [TupletoSet](c), d IN [Cycless](w)), w IN c), c IN [Cycless](v)), v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a s="EXISTS (EXISTS (c != d AND [TupletoSet](c) == [TupletoSet](d), d IN [Cycless](0)), c IN [Cycless](0))" all
$PTH/flagcalc -d f="abc" -a s="EXISTS (EXISTS (c != d AND [TupletoSet](c) == [TupletoSet](d), d IN [Cycless](0)), c IN [Cycless](0))" all
$PTH/flagcalc -r 20 95 100 -a s="TALLY ([vdt](v), v IN [V])/2 == [edgecm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 15 52.5 10000 -a s="COUNT ([vdt](v) % 2 == 1, v IN [V]) % 2 == 0" all -v i=minimal3.cfg
$PTH/flagcalc -r 15 52.5 10000 -a i="COUNT ([vdt](v) % 2 == 1, v IN [V])" all -v i=minimal3.cfg
$PTH/flagcalc -r 6 7.5 1 -a e="[Cycless](0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a p="[idxs]([Pathss](0,1),0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcdef" -a p="[idxs]([Pathss](0,5),0)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 10 22.5 1 -a e="[V] CUP [E]" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="[Pathss](0,2) CUP [Pathss](0,3)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcde" -a e="[V] CUP [E]" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc" -a e="[Pathss](0,1) CUP [Pathss](0,2)" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUPD ([Cycless](v), v IN [V])" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 8 14 100 -g o=out.dat all overwrite -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUPD ([Cycless](v), v IN [V])" all -v set i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUP ([Cycless](v), v IN [V])" all -v set i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUP (BIGCUP ([TupletoSet](c), c IN [Cycless](v)), v IN [V])" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d out.dat -a e="BIGCUP (SET (a, a IN c), c IN [Cylcess](v)), v IN [V])" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (SET ([TupletoSet](c), c IN [Cycless](v)), v IN [V])" all -v set allsets i=minimal3.cfg
