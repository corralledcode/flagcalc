PTH='../cmake-build-debug'
$PTH/flagcalc -r 9 2 50 -a s="EXISTS (EXISTS ((s CUP t) == [V] AND [st](s) == ceil([dimm]/2) AND [st](t) == floor([dimm]/2), t IN [P]([V])), s IN [P]([V]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 10 50 -a s="FORALL (EXISTS (EXISTS ((s CUP t) == [V] AND [st](s) == ceil([dimm]/(n+1)) AND [st](t) == floor(n*[dimm]/(n+1)), t IN [P]([V])), s IN [P]([V])), n IN [NN]([dimm]))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 10 1000 -a s="FORALL (EXISTS (EXISTS (a ELT e AND b ELT e AND a != b, b IN [V]), a IN [V]), e IN [E])" all -v i=minimal3.cfg
$PTH/flagcalc -d halfconnectedgraph.dat -a s="EXISTS ([st](s) >= [dimm]/2 AND FORALL (FORALL (a == b OR EXISTS (a ELT e AND b ELT e, e IN [E]), b IN s), a IN s), s IN [P]([V]))" all
$PTH/flagcalc -r 10 45 100 -a s="FORALL ([Knc](n,1) IFF EXISTS (FORALL (FORALL (a == b OR EXISTS (a ELT e AND b ELT e, e IN [E]) , a IN s), b IN s), s IN [Sizedsubset]([V],n)), n IN [NN]([dimm]+1))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 15 1 -a s="FORALL (s == [V] OR EXISTS (EXISTS ((NOT (u <= s)) AND (NOT (t <= s)) AND ((s CAP t) <= s), u IN [P]([V])), t IN [P]([V])), s IN [P]([V]))" all
$PTH/flagcalc -r 9 15 1 -a s="EXISTS (EXISTS (((s CAP t) == [Nulls]) AND ((s CUP t) == [V]) AND FORALL (FORALL (NOT [ac](a,b), b IN s), a IN s) AND FORALL (FORALL (NOT [ac](c,d), d IN t), c IN t), t IN [P]([V])), s IN [P]([V]))" all