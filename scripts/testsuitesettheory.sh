# This is a comprehensive script running in the ones or tens of minutes depending on the hardware
# It covers some very pertinent theorems from graph theory, and leans heavily on set-theoretic notions and FOL

PTH=${PTH:-'../bin'}

# This is just fun with ceiling and floor, to effect a partition of V into two disjoint subsets.
# Note the measure "st" returns the size (cardinality) of the set or tuple passed to it as an argument
# This query should return all 50 "true"
$PTH/flagcalc -r 9 2 50 -a s="EXISTS (t IN Ps(V), EXISTS (s IN Ps(V), (s CUP t) == V AND st(s) == ceil(dimm/2) AND st(t) == floor(dimm/2)))" all -v i=minimal3.cfg

# This again is just set theory and arithmetic. It should return all 1 true: for each n < graph dim, the correspondingly-sized sets form a partition of V
$PTH/flagcalc -r 10 10 1 -a s="FORALL (n IN NN(dimm), EXISTS (s IN Ps(V), EXISTS (t IN Ps(V), (s CUP t) == V AND st(s) == ceil(dimm/(n+1)) AND st(t) == floor(n*dimm/(n+1)))))" all -v i=minimal3.cfg

# This is more set theory around the definition of the set of edges E: each e IN E consists of exactly two distinct items. Note that e IN E is not a tuple but a set
$PTH/flagcalc -r 9 10 100 -a s="FORALL (e IN E, EXISTS (a IN V, EXISTS (b IN V, a ELT e AND b ELT e AND a != b)))" all -v i=minimal3.cfg

# "There is a subset of V having size at least half of graph dim, such that the subset is completely connected.
# Inspecting the data file halfconnectedgraph.dat, we see this should return "True" for four out of the five graphs in that file
$PTH/flagcalc -d halfconnectedgraph.dat -a s="EXISTS (s IN Ps(V), st(s) >= dimm/2 AND FORALL (a IN s, FORALL (b IN s, a == b OR EXISTS (e IN E, a ELT e AND b ELT e))))" all -v i=minimal3.cfg

# These last few have all used an EXISTS to test what can more succinctly be tested by invoking measure "ac", taking two inputs and returning True precisely if the two vertices are connected
# In the present case, we are testing that the measure "Knc" is correct: Knc takes two arguments n and mincnt and returns true if there are at least mincnt embeddings of K_n into the graph
# Here: all of the ten random graphs should return True
$PTH/flagcalc -r 10 20 10 -a s="FORALL (n IN NN(dimm+1), Knc(n,1) IFF EXISTS (s IN Sizedsubset(V,n), FORALL (b IN s, FORALL (a IN s, a == b OR EXISTS (e IN E, a ELT e AND b ELT e)))))" all -v i=minimal3.cfg

# This is half a thought: if a subset s is not all of V, then there are two subsets of V not subsets of the subset s such that the meet of s and (any) one of them is contained in s
# called "half a thought" because it isn't invoking u meaningfully; one can find inspiration, though, and try several variations on this query
$PTH/flagcalc -r 9 15 1 -a s="FORALL (s IN Ps(V), s == V OR EXISTS (t IN Ps(V), EXISTS (u IN Ps(V), (NOT (u <= s)) AND (NOT (t <= s)) AND ((s CAP t) <= s))))" all

# This is a way to test a graph for being complete bipartite: it should return true for the duplicate two graphs in the named data file
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (s IN Ps(V), EXISTS (t IN Ps(V), ((s CAP t) == Nulls) AND ((s CUP t) == V) AND FORALL (a IN s, FORALL (b IN s, NOT ac(a,b))) AND FORALL (d IN t, FORALL (c IN t, NOT ac(c,d)))))" all

# This is elementary set theory: it states that E is a subset of the powerset of V: it should return true for all fifty graphs
$PTH/flagcalc -r 4 5 50 -a s="EXISTS (s IN Ps(Ps(V)), s == E)" all -v i=minimal3.cfg

# This is more elementary set theory: for every edge e there is a set a amongst subsets of V sized 2, such that e is the set consisting of two elements from a
# Please note one can also just say "a == e"
$PTH/flagcalc -r 6 10 2 -a s="FORALL (e IN E, EXISTS (a IN Sizedsubset(V,2), FORALL (b IN a, FORALL (c IN a, (b ELT e) AND (c ELT e)))))" all -v i=minimal3.cfg

# This is another way to state the previous more succinctly
$PTH/flagcalc -r 6 10 2 -a s="FORALL (e IN E, e ELT Sizedsubset(V,2))" all -v i=minimal3.cfg

# This states that there is a set of sized-two subsets of V that essentially equals E: all 8 should return True
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (s IN Ps(Sizedsubset(V,2)), FORALL (a IN s, FORALL (b IN a, FORALL (c IN a, EXISTS (e IN E, c ELT e AND b ELT e)))))" all -v i=minimal3.cfg

# Again, the previous but more succinctly
$PTH/flagcalc -r 6 10 8 -a s="EXISTS (s IN Ps(Sizedsubset(V,2)), FORALL (a IN s, a ELT E))" all -v i=minimal3.cfg

# Some subset of the set of subsets of V has size 2^dimm: since the size of Ps(V) is 2^dimm, this is
# pointing out that one of the subsets of Ps(V) is indeed all of Ps(V); it should return True
$PTH/flagcalc -r 4 3 1 -a s="EXISTS (s IN Ps(Ps(V)), st(s) == 2^dimm)" all

# This simply states (by way of continuing to affirm the language and its parser) that every subset of subsets of V
# either has some element, or is sized 0. Note using "ELT" inline, but "IN" for quantifying;
# this is a departure from received notation, where the epsilon is used for both; it helps with syntax and parsing
$PTH/flagcalc -r 4 3 1 -a s="FORALL (s IN Ps(Ps(V)), EXISTS (t IN Ps(V), t ELT s) OR st(s) == 0)" all

# This again is more affirmation from elementary set theory: some subset of size equal to the size of the edges set E
# of the set of 2-sized subsets of V, is equal to the edge set E; it should return True for all ten
$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s IN Sizedsubset(Sizedsubset(V,2),st(E)), s == E)" all -v i=minimal3.cfg

$PTH/flagcalc -r 7 8 10 -a s="EXISTS (s IN Ps(Sizedsubset(V,2)), s == E)" all -v i=minimal3.cfg

$PTH/flagcalc -d testbip12.dat -a s="EXISTS (n IN NN(dimm+1), EXISTS (l IN Sizedsubset(V,n), EXISTS (r IN Sizedsubset(V,dimm - n), st(r CUP l) == dimm AND FORALL (a IN l, FORALL (b IN l, NOT ac(a,b))) AND FORALL (c IN r, FORALL (d IN r, NOT ac(c,d))))))" all

$PTH/flagcalc -r 4 3 10 -a s="FORALL (n IN NN(2^dimm+1), EXISTS (s IN Sizedsubset(Ps(V),n), EXISTS (t IN Sizedsubset(Ps(V),2^dimm - n), ((s CUP t) == Ps(V)) AND ((s CAP t) == Nulls))))" all -v i=minimal3.cfg

# This is an obsolete way to index into a set; now we use square brackets as in e[0], and e[2]
$PTH/flagcalc -r 10 20 100 -a s="FORALL (e IN E, ac(idxt(e,0),idxt(e,1)))" all -v i=minimal3.cfg

$PTH/flagcalc -r 10 20 100 -a s="FORALL (n IN NN(st(E)), ac(idxt(idxs(E,n),0),idxt(idxs(E,n),1)))" all -v i=minimal3.cfg

# here st(Pathss(a,b)) counts the size of that set, and checks it's equal to the measure pct
$PTH/flagcalc -r 7 12 10 -a s="FORALL (b IN V, FORALL (a IN V, st(Pathss(a,b)) == pct(a,b)))" all -v i=minimal3.cfg

# this again uses obsolete dereferencing to check that adjacent vertices in a path are edge-connected
$PTH/flagcalc -r 6 10 10 -a s="FORALL (b IN V, FORALL (a IN V, FORALL (p IN Pathss(a,b), FORALL (m IN NN(st(p)-1), ac(idxt(p,m),idxt(p,m+1))))))" all -v i=minimal3.cfg

# this uses a standard complete bipartite graph to check all cycles are even in length; also note there is mod built-in as the percent symbol (%), not shown here
$PTH/flagcalc -d testbip8.dat -a s="FORALL (v IN V, FORALL (c IN Cyclesvs(v), mod(st(c),2) == 0))" all

# this checks a graph is triangle free if and only if every cycle starting at every vertex is at least 4 in length
$PTH/flagcalc -r 8 10 1000 -a s="cr1 IFF FORALL (v IN V, FORALL (c IN Cyclesvs(v), st(c) > 3))" all -v i=minimal3.cfg

# a graph is a tree if and only if every two vertices have exactly one path between them
# (note it is correct that a vertex has exactly one path, of length zero, with itself (see defn in Diestel p. 6)
$PTH/flagcalc -r 9 8 100 -a s="treec IFF FORALL (a IN V, FORALL (b IN V, st(Pathss(a,b)) == 1))" all -v i=minimal3.cfg

# Again, now the forestc is whether all pairs have at most one path between them
$PTH/flagcalc -r 9 12 100 -a s="forestc IFF FORALL (a IN V, FORALL (b IN V, st(Pathss(a,b)) <= 1))" all -v i=minimal3.cfg

# A variant: now the test is that a graph is a forest if the cycles containing any given vertex a are zero in number
$PTH/flagcalc -r 9 12 100 -a s="forestc IFF FORALL (a IN V, st(Cyclesvs(a)) == 0)" all -v i=minimal3.cfg

# this is repeated, just showing how we feel out good numbers for the randomizer, to produce a few actual forests
$PTH/flagcalc -r 9 6 100 -a s="forestc IFF FORALL (a IN V, st(Cyclesvs(a)) == 0)" all -v i=minimal3.cfg

# Here we first encounter Pathss(a,b) being the empty set: a graph that isn't 1-connected (defn Diestel p. 11)
$PTH/flagcalc -r 10 15 50 -a s="conn1c IFF FORALL (a IN V, FORALL (b IN V, Pathss(a,b) != Nulls))" all -v i=minimal3.cfg

# This checks that for any given cycle c, and for any vertex w in that cycle, the cycles containing w contain some permutation of c
# (achieved not by using a quantifier over Permss (the set of permutations), but simply by "forgetting" and treating the tuple as an (unordered) set
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (v IN V, FORALL (c IN Cyclesvs(v), FORALL (w IN c, TupletoSet(c) ELT Cyclesvs(w))))" all -v i=minimal3.cfg

# Here we are just testing a bit more the previous query
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (v IN V, FORALL (c IN Cyclesvs(v), FORALL (w IN c, TupletoSet(c) ELT (SET (d IN Cyclesvs(w), TupletoSet(d))))))" all -v i=minimal3.cfg

# Testing the built-in criteria bipc, more antiquated but still functional: it takes as input two purported bipartite sets comprising V
# (i.e. not directly using the later-added "Setpartition(V)" with st(...) == 2).
# Note also per received definition, that complete bipartite is a stronger statement than bipartite (see defn Diestel p 17)
$PTH/flagcalc -d testbip10.dat -a s="EXISTS (r IN Ps(V), EXISTS (l IN Ps(V), (l CUP r) == V AND bipc(l,r)))" all

# This is again using the criteria of no odd cycles for bipartite
$PTH/flagcalc -r 9 10 100 -a s="(EXISTS (r IN Ps(V), EXISTS (l IN Ps(V), (l CUP r) == V AND bipc(l,r)))) IFF FORALL (v IN V, FORALL (c IN Cyclesvs(v), mod(st(c),2) == 0))" all -v i=minimal3.cfg

# This is for fun, again, using some arithmetic: the number of cycles of a given length n is equal to the double sum
# over first vertices, then over each cycle around that vertex, finally taking the summand to be 1/"the size of the cycle".
$PTH/flagcalc -r 7 10 100 -a s="SUM (n IN NN(dimm+1), cyclet(n)) == SUM (v IN V, SUM (c IN Cyclesvs(v), 1/st(c)))" all -v i=minimal3.cfg

# This is the handshake lemma
$PTH/flagcalc -r 12 33 1000 -a s="SUM (v IN V, vdt(v))/2 == edgecm" all -v i=minimal3.cfg

# This is a way to express the quantifier "COUNT" using "SUM", and introduces again edgecm (these early measures
# were incorrectly given suffix of "m" even though they are discrete (should end in "t")
$PTH/flagcalc -r 7 10 100 -a s="SUM (e IN E, 1) == edgecm" all -v i=minimal3.cfg

# This is literally just the definition of the new measure cyclesvt; worth checking, however, the speed of the two
# ways of expressing the quantity in this query (not done here, but a simple exercise)
$PTH/flagcalc -r 8 10 100 -a s="FORALL (v IN V, st(Cyclesvs(v)) == cyclesvt(v))" all -v i=minimal3.cfg

# Again here using an antiquated tally, "lt" (tuple length tally) (just use st). But the arithmetic is fun
# (note: "ac" is the widely-used adjacency criterion)
$PTH/flagcalc -r 7 10 1000 -a s="FORALL (v IN V, FORALL (c IN Cyclesvs(v), SUM (a IN c, SUM (b IN c, ac(b,a)))/2 >= lt(c)))" all -v i=minimal3.cfg

#  This is a repeat of an earlier query in this file, but showing we can directly use equality now between sets
$PTH/flagcalc -r 7 10.5 50 -a s="FORALL (v IN V, FORALL (c IN Cyclesvs(v), FORALL (w IN c, EXISTS (d IN Cyclesvs(w), TupletoSet(d) == TupletoSet(c)))))" all -v i=minimal3.cfg

# Here relatively meaningless, except to verify that a complete K_4 graph has at least two cycles around the first vertex
$PTH/flagcalc -d f="abcd" -a s="EXISTS (c IN Cyclesvs(0), EXISTS (d IN Cyclesvs(0), c != d AND TupletoSet(c) == TupletoSet(d)))" all

# Now we see the reasoning: this way, it is no longer true (in reference to the previous query)
$PTH/flagcalc -d f="abc" -a s="EXISTS (c IN Cyclesvs(0), EXISTS (d IN Cyclesvs(0), c != d AND TupletoSet(c) == TupletoSet(d)))" all

# This is just seeing a first application of quantifier "TALLY", like in the earlier demonstration of this handshake lemma
# Please note that three of the following four queries returns all True (well, all four if extremely lucky)
$PTH/flagcalc -r 20 95 100 -a s="TALLY (v IN V, vdt(v))/2 == edgecm" all -v i=minimal3.cfg
$PTH/flagcalc -r 20 95 100 -a s="TALLY (v IN V, vdt(v)/2) == edgecm" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 1000 -a s="SUM (v IN V, vdt(v))/2 == edgecm" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 33 1000 -a s="SUM (v IN V, vdt(v)/2) == edgecm" all -v i=minimal3.cfg

# Here the number of vertices having odd degree has to be even
# (Diestel prop 1.2.1)
$PTH/flagcalc -r 15 52.5 10000 -a s="COUNT (v IN V, vdt(v) % 2 == 1) % 2 == 0" all -v i=minimal3.cfg

# Here is the same query, but with the results categorized with the (necessarily even) number between zero and 15
# of how many out of the ten-thousand have this sum of their respective vertices' degrees
# (please note earlier versions of this script obtained the same result replacing documented
# feature "z" with undocumented "i")
$PTH/flagcalc -r 15 52.5 10000 -a z="COUNT (v IN V, vdt(v) % 2 == 1)" all -v i=minimal3.cfg

# This just is a first foray into using "e=" under "-a": ensemble aka set valued measure
# It should return a set of tuples, not repeating the reverse of each one (per definition)
$PTH/flagcalc -r 6 7.5 1 -a e="Cyclesvs(0)" all -v set allsets i=minimal3.cfg

# This shows again the antiquated method of dereferencing a set
$PTH/flagcalc -d f="abcd" -a p="idxs(Pathss(0,1),0)" all -v set allsets i=minimal3.cfg

# ibid
$PTH/flagcalc -d f="abcdef" -a p="idxs(Pathss(0,5),0)" all -v set allsets i=minimal3.cfg

# This just shows the dynamic nature of the set theory in flagcalc (the set-theoretic union of V and E), on ten random graphs
$PTH/flagcalc -r 10 22.5 1 -a e="V CUP E" all -v set allsets i=minimal3.cfg

# this again just shows some effective set theoretic operation
$PTH/flagcalc -d f="abcd" -a e="Pathss(0,2) CUP Pathss(0,3)" all -v set allsets i=minimal3.cfg

# more random ideas
$PTH/flagcalc -d f="abcde" -a e="V CUP E" all -v set allsets i=minimal3.cfg

# again, non-overlapping sets, but still can talk about their union
$PTH/flagcalc -d f="abc" -a e="Pathss(0,1) CUP Pathss(0,2)" all -v set allsets i=minimal3.cfg

# here we use the "D" as in BIGCUPD to save the effort of checking the set-theoretic Union for duplicates
$PTH/flagcalc -d f="abcd" -a e="BIGCUPD (v IN V, Cyclesvs(v))" all -v set allsets i=minimal3.cfg

# Why not check that the BIGCUPD above returns the same as BIGCUP (should return True)
$PTH/flagcalc -d f="abcd" -a s="BIGCUPD (v IN V, Cyclesvs(v)) == BIGCUP (v IN V, Cyclesvs(v))" all -v set allsets i=minimal3.cfg

# Here we populate out.dat with 100 random graphs
$PTH/flagcalc -r 8 14 100 -g o=out.dat all overwrite -v i=minimal3.cfg

# Now for the next three queries we use that same set of 100 (useful if you want honest numbers, not due to random
# differences of which graphs were chosen, especially with say 10 instead of 100)
$PTH/flagcalc -d out.dat -a e="BIGCUPD (v IN V, Cyclesvs(v))" all -v set i=minimal3.cfg

# Wildly slower, but the same output
$PTH/flagcalc -d out.dat -a e="BIGCUP (v IN V, Cyclesvs(v))" all -v set i=minimal3.cfg

# This is a bit different, and quite verbose (due to "allsets")
$PTH/flagcalc -d out.dat -a e="BIGCUP (v IN V, BIGCUP (c IN Cyclesvs(v), TupletoSet(c)))" all -v set allsets i=minimal3.cfg

# This repackages every vertices' set of cycles around it, into sets rather than tuples, and omitting the "D" it has the effect
# of forgetting the order of the vertices in the cycle. So in a K_4, for example, -abdca is equal to -adbca
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), TupletoSet(c)))" all -v set allsets i=minimal3.cfg

# This should contain all vertices with non-zero degree (namely, the first three)
$PTH/flagcalc -d f="abc d" -a e="SET (x IN V, st(SET (e IN E, x ELT e, e)) > 0, x)" all -v set allsets i=minimal3.cfg

# More with bipartite graphs and odd cycles
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUPD (v IN V, SET (c IN Cyclesvs(v), st(c) % 2 == 1, TupletoSet(c)))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUPD (v IN V, SET (c IN Cyclesvs(v), st(c) % 2 == 0, TupletoSet(c)))" all -v set i=minimal3.cfg
$PTH/flagcalc -d f="abcd=efgh" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), st(c) % 2 == 0, TupletoSet(c)))" all -v set i=minimal3.cfg

# Here using SUM rather than COUNT, but to the same end (should output zero)
$PTH/flagcalc -d f="abcd=efgh" -a a="SUM (v IN V, vdt(v) > dimm/2, 1)" all -v i=minimal3.cfg

# To wit: here just checking (new) feature "COUNT" is the same as the appropriate "SUM"
$PTH/flagcalc -r 16 60 100 -a s="SUM (v IN V, vdt(v) > dimm/4, 1) == COUNT (v IN V, vdt(v) > dimm/4)" all -v i=minimal3.cfg

# More plain ideas about vertex degree tally ("vdt") and adjacency criterion ("ac")
$PTH/flagcalc -r 16 60 1000 -a s="FORALL (v IN V, vdt(v) > 0, EXISTS (u IN V, ac(u,v)))" all -v i=minimal3.cfg

# Exercise: write out this three-level-deep nested set of sets of sets
# Answer: it should be just {{{0,1,2}}}
$PTH/flagcalc -d f="abc" -a e="SET (v IN V, SET (u IN V, SET (t IN V, t)))" all -v set allsets i=minimal3.cfg

# Now the same thing with fast "D" ("SETD") option
# Answer: it should be three sets of three sets of {0,1,2} (that is, not technically a "set" to a purist, since a set
# can't repeat items, and here it is being repeated at two levels in fact)
$PTH/flagcalc -d f="abc" -a e="SETD (v IN V, SETD (u IN V, SETD (t IN V, t)))" all -v set allsets i=minimal3.cfg

# Something to puzzle through
$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (a IN c, EXISTS (b IN c, a ELT e AND b ELT e AND a != b)), e)))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), idxt(c,n) ELT e AND idxt(c,(n + 1) % st(c)) ELT e), e)))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -r 8 14 100 -a s="cr1" e2="BIGCUP (v IN V, Cyclesvs(v))" all -v set i=minimal3.cfg

$PTH/flagcalc -d f="abcd" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c[n] ELT e AND c[(n+1)%st(c)] ELT e), e)))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="-abcdea" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c[n] ELT e AND c[(n+1)%st(c)] ELT e), e)))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcde" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c[n] ELT e AND c[(n+1)%st(c)] ELT e), e)))" all -v set i=minimal3.cfg

$PTH/flagcalc -r 8 14 100 -a f="abcd" a2="cliquem>3" all -v set i=minimal3.cfg

$PTH/flagcalc -d f="abcde" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c[n] ELT e AND c[(n+1)%st(c)] ELT e), e)))" all -v set i=minimal3.cfg

$PTH/flagcalc -r 4 3 1 -a s="FORALL (s IN Ps(Ps(V)), EXISTS (t IN Ps(V), t ELT s) OR st(s) == 0)" all

$PTH/flagcalc -d f="abc" -a e="SET (v IN V, SET (u IN V, SET (t IN V, t)))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -r 7 10 100 -a s="SUM (n IN NN(dimm+1), cyclet(n)) == st(BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c[n] ELT e AND c[(n+1)%st(c)] ELT e), e))))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abcde" -a s="SUM (n IN NN(dimm+1), cyclet(n)) == st(BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c[n] ELT e AND c[(n+1)%st(c)] ELT e), e))))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abcde" -a e="{3+2,SUM (v IN V, v), {1}}" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abc=def=ghi" -a e="BIGCUP (v IN V, {v+1})" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abc=def=ghi" -a e="SET (v IN V, {v+1})" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abc=def=ghi" -a p="<<0,<<<<1>>,5,6>>,2,3>>" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcd=ef=ghi" -a e="{SET (v IN V, FORALL (c IN Cyclesvs(v), st(c) % 2 == 0), v)}" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcd=efghi" -a e="{SET (v IN V, FORALL (c IN Cyclesvs(v), (st(c) % 2) == 0), v)}" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="-abcdefdgha" -a e="Cyclesvs(0)" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="-abcdefdgha" -a e="BIGCUP (v IN V, SET (c IN Cyclesvs(v), SET (e IN E, EXISTS (n IN NN(st(c)), c[n] ELT e AND c[(n+1)%st(c)] ELT e), e)))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcde" -a e="SET (es IN Ps(E), st(es) != 0, (es[0])[0])" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcdef" -a a="MAX (n IN NN(st(Cyclesvs(0)[0])), (Cyclesvs(0)[0])[n])" all -v i=minimal3.cfg

$PTH/flagcalc -d f="abc=def=ghi" -a a="<<0,2, SUM (v IN Ps(V), st(v) > 0, st(Cyclesvs(v[0])))>>[1]" all -v set allsets i=minimal3.cfg

# Experiementing with tuple constants or literals; one can also add tuples pointwise with "+" or append them with CUPD or CUP (either works)
$PTH/flagcalc -d f="abc=def=ghi" -a a="<<0,2, SUM (v IN Ps(V), st(v) > 0, st(Cyclesvs(v[0])))>>[2]" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abc=def=ghi" -a e="SETD (n IN st(V), SETD (i IN n, {i}))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcd" -a e="SET (s IN Setpartition(V), s)" all -v set allsets i=minimal3.cfg

# This one says: amongst the (first iteration) 1-connected sets, then amongst those the ones (second iteration) that embed a complete graph of dimension dimm,
# there is a partition of V into \Delta (graph's max degree) partitions, such that each partition within itself is disconnected.
# This isn't annotated because the second iteration's criteria implies the first iteration's criteria, so it is some test
# perhaps nonetheless inspired by a meaningful proposition
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
$PTH/flagcalc -d f="abc=defg" f="abcd=efgh" f="abc=de" -a s="FORALL (v IN V, FORALL (c IN Cyclesvs(v), mod(st(c),2) == 0))" s2="Chiprimet == Deltam" all -v i=minimal3.cfg

# Diestel 5.3.1 (Konig 1916)

$PTH/flagcalc -r 8 14 1000 -a s="FORALL (v IN V, FORALL (c IN Cyclesvs(v), mod(st(c),2) == 0))" s2="Chiprimet == Deltam" all -v i=minimal3.cfg

# Diestel 2.1.1

$PTH/flagcalc -r 8 14 200 -a s="EXISTS (p IN Setpartition(V), st(p) == 2, bipc(p[0],p[1]))" s2="MAX (es IN Ps(E), FORALL (e1 IN es, FORALL (e2 IN es, NOT eadjc(e1,e2))), st(es)) == MIN (vs IN Ps(V), FORALL (e IN E, vs CAP e != Nulls), st(vs))" all -v i=minimal3.cfg

# Diestel 2.1.2 (Hall 1935)

# $PTH/flagcalc -d testbip8.dat -a s="FORALL (p IN Setpartition(V), (bipc(p(0),p(1)) && st(p) == 2) IMPLIES (EXISTS (es IN Ps(E), FORALL (e1 IN es, FORALL (e2 IN es, NOT eadjc(e1,e2))) AND FORALL (a IN p(0), EXISTS (e IN es, a ELT e))) IFF FORALL (S IN Ps(p(0)), Nt(S) >= st(S))))" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 14 2000 -a s="FORALL (p IN Setpartition(V), (EXISTS (es IN Ps(E), FORALL (e1 IN es, FORALL (e2 IN es, NOT eadjc(e1,e2))) AND FORALL (a IN p[0], EXISTS (e IN es, a ELT e))) IFF FORALL (S IN Ps(p[0]), Nt(S) >= st(S))) IF (bipc(p[0],p[1]) && st(p) == 2))" all -v i=minimal3.cfg

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
s2="FORALL (ne IN nE, EXISTS (vs IN Ps(V), EXISTS (m IN Setpartition(vs), st(m) == 4, FORALL (p IN m, FORALL (q IN p, FORALL (r IN p, EXISTS (z IN Pathss(q,r), z <= p))) OR (ne[0] ELT p AND ne[1] ELT p AND FORALL (u IN p, FORALL (v IN p, (EXISTS (z IN Pathss(u,ne[0]), z <= p) AND EXISTS (z2 IN Pathss(ne[1],v), z2 <= p)) OR (EXISTS (z3 IN Pathss(v,ne[0]), z3 <= p) AND EXISTS (z4 IN Pathss(u,ne[1]), z4 <= p)))))) AND FORALL (p2 IN m, FORALL (q2 IN m, p2 != q2, Nssc(p2,q2) OR (ne[0] ELT p2 AND ne[1] ELT q2) OR (ne[1] ELT p2 AND ne[0] ELT q2) )) )))" \
z3="edgecm == 2*dimm - 3" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 14 1500 -a s="FORALL (vs IN Ps(V), NOT EXISTS (o IN Setpartition(vs), st(o) == 4, FORALL (p IN o, connvsc(p,E)) AND FORALL (p1 IN o, FORALL (q1 IN o, q1 != p1, Nssc(p1,q1)))))" \
s2="FORALL (ne IN nE, EXISTS (vs IN Ps(V), EXISTS (m IN Setpartition(vs), st(m) == 4, FORALL (p IN m, connvsc(p, E CUP {ne})) AND FORALL (p2 IN m, FORALL (q2 IN m, p2 != q2, Nssc(p2,q2) OR (ne[0] ELT p2 AND ne[1] ELT q2) OR (ne[1] ELT p2 AND ne[0] ELT q2) )) )))" \
z3="edgecm == 2*dimm - 3" all -v i=minimal3.cfg


# Diestel Theorem 3.3.6(i) (Global version of Menger's Theorem) Note that the native kconnc is much faster than the count of independent paths, and here that speedup is a matter of the native code being to the left of the IFF

$PTH/flagcalc -r 5 5 300 -a s="FORALL (k IN dimm, kconnc(k) IFF FORALL (v1 IN V, FORALL (v2 IN V, v1 != v2, EXISTS (ps IN Ps(Pathss(v1,v2)), st(ps) >= k, FORALL (p1 IN ps, FORALL (p2 IN ps, p1 != p2, FORALL (v3 IN p1, v3 != v1 AND v3 != v2, FORALL (v4 IN p2, v4 != v1 AND v4 != v2, v3 != v4))))))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 300 -a s="FORALL (k IN dimm, kconnc(k) IFF FORALL (v1 IN V, FORALL (v2 IN V, v1 != v2, EXISTS (ps IN Ps(Pathss(v1,v2)), st(ps) >= k, FORALL (p1 IN ps, FORALL (p2 IN ps, p1 != p2, p1 CAP p2 <= {v1,v2}))))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 300 -a s="FORALL (k IN dimm, FORALL (v1 IN V, FORALL (v2 IN V, v1 != v2, EXISTS (ps IN Ps(Pathss(v1,v2)), st(ps) >= k, FORALL (p1 IN ps, FORALL (p2 IN ps, p1 != p2, p1 CAP p2 <= {v1,v2}))))) IFF kconnc(k))" all -v i=minimal3.cfg

# Diestel Theorem 3.3.1 (Menger 1927)

$PTH/flagcalc -r 5 5 50 -a isp="../scripts/storedprocedures.dat" s="FORALL (A IN Ps(V), FORALL  (B IN Ps(V), MIN (X IN Ps(V), Separatesc(A,B,X), st(X)) == DisjointABpaths( A, B ) ))" all -v i=minimal3.cfg
$PTH/flagcalc -r 5 5 50 -a isp="../scripts/storedprocedures.dat" s="FORALL (A IN Ps(V), FORALL  (B IN Ps(V), MinXSeparates(A,B) == DisjointABpaths( A, B ) ))" all -v i=minimal3.cfg


# Diestel Theorem 7.4.1 (Regularity lemma), specifically "admits an epsilon-regular partition with partition size k <= M"

# $PTH/flagcalc -r 8 14 300 -a z="EXISTS (vp IN Setpartition(V), st(vp(0)) <= 0.1*st(vp(1)) AND st(vp) <= 5 AND FORALL (n IN st(vp)-1, n > 0, vp(n) == vp(n+1)), FORALL (i IN st(vp), FORALL (j IN st(vp), i != j, Nsst(vp(i),vp(j))/(st(vp(i)*st(vp(j)) "

# Diestel Exercise 1.7 (p 10)

$PTH/flagcalc -r 8 14 1000 -a isp="../scripts/storedprocedures.dat" s="n_0(deltam, girthm) <= dimm" all -v i=minimal3.cfg

# Diestel Theorem 1.3.4 (Alon et al 2002)

$PTH/flagcalc -r 8 14 100 -a isp="../scripts/storedprocedures.dat" s="NOT isinf(girthm)" s2="FORALL (d IN dm + 1, d >= 2, FORALL (g IN girthm, n_0(d,g) <= dimm ))" all -v i=minimal3.cfg

# Diestel Prop 1.4.2 (p. 12)

$PTH/flagcalc -d octahedron.dat -a z="lambdat" z="kappat" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 14 10000 -a s="dimm>1" s2="kappat <= lambdat && lambdat <= deltam" all -v i=minimal3.cfg

# Diestel Theorem 1.4.3 (p. 13) (Mader 1972)

$PTH/flagcalc -r 12 50 1000 -a s="FORALL (k IN dimm, k > 0 AND dm >= 4*k, EXISTS (U IN Ps(V), kconnc(SubgraphonUg(U),k+1) AND dm(SubgraphonUg(U))/2 > dm/2 - k))" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 50 1000 -a s="FORALL (k IN dimm, k > 0 AND dm >= 4*k, EXISTS (U IN Ps(V), NAMING (SU AS SubgraphonUg(U), kconnc(SU,k+1) AND dm(SU)/2 > dm/2 - k)))" all -v i=minimal3.cfg

$PTH/flagcalc -r 12 p=0.4 1000 -a s="FORALL (k IN dimm, k > 0 AND dm >= 4*k, EXISTS (U IN Ps(V), NAMING (SU AS SubgraphonUg(U), kconnc(SU,k+1) AND dm(SU)/2 > dm/2 - k)))" all -v i=minimal3.cfg

# SLOW (by being specific about max k-connectedness rather than just a threshold: uncomment for a run-time magnitudes slower than the above
# $PTH/flagcalc -r 12 p=0.4 100 -a s="FORALL (k IN NN(dimm), k > 0, dm >= 4*k IMPLIES EXISTS (S IN Ps(V), NAMING (H AS SubgraphonUg(S), kappat(H) > k AND  edgecm(H)/dimm(H) > edgecm/dimm - k))) "  all -v i=minimal3.cfg allsets

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

# $PTH/flagcalc -r 6 7.5 10 -a isp="../scripts/storedprocedures.dat" a="MAX (H IN Ps(V), M_H(H))" all -v i=minimal3.cfg
# $PTH/flagcalc -r 6 7.5 10 -a isp="../scripts/storedprocedures.dat" s="\
# FORALL (H IN Ps(V), NAMING (MHH AS M_H(H), NAMING (VminusH AS V SETMINUS H, EXISTS (p IN Setpartition(VminusH), st(p) >= MHH, \
#  FORALL (s IN p, EXISTS (v1 IN H, EXISTS (v2 IN H, v1 != v2, NAMING (v1v2 AS Pathss(v1,v2), EXISTS (path IN v1v2, path <= (s CUPD {v1,v2}) ))))) ))))" all -v i=minimal3.cfg

$PTH/flagcalc -d platonicsolids.dat -a e="NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), SET (c IN C, st(c) == m, c)))" all -v set allsets i=minimal3.cfg

# intractable: $PTH/flagcalc -d platonicsolids.dat -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), BIGCUP (h IN Setpartition(hs), FORALL (h1 IN h, FORALL (h2 IN h, EXISTS (a IN A, FORALL (v IN V, a(h1(v)) == h2(v))))), h)))))" -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcd" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), SET (h1 IN hs, SET (h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))), h2 ))))))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcda -efghe ae bf cg dh" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), SET (h1 IN hs, SET (h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))), h2 ))))))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc def abd bce caf" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), SET (h1 IN hs, SET (h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))), h2 ))))))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdea -afghb -bhijc -cjkld -dlmne -enofa -pqrstp op gq ir ks mt" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), SET (h1 IN hs, SET (h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))), h2 ))))))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="-abcdea -afghb -bhijc -cjkld -dlmne -enofa -pqrstp op gq ir ks mt" -a gm="SET (v1 IN V, v2 IN V, {v1,v2} ELT E, {v1,v2})" all -v measg set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc def abd bce caf" -a gm="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), SET (h1 IN hs, h2 IN hs, h1 != h2 AND EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR  FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))), {h1,h2} )))))" all -v measg set allsets i=minimal3.cfg
$PTH/flagcalc -d f="-abcdea -afghb -bhijc -cjkld -dlmne -enofa -pqrstp op gq ir ks mt" -a gm="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), SET (h1 IN hs, h2 IN hs, h1 != h2 AND EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))), {h1,h2} )))))" all -v measg set allsets i=minimal3.cfg

$PTH/flagcalc -r 8 5 10 -a s="EXISTS (y IN Sizedsubset(V,3), EXISTS (a IN V, b IN V, c IN V, d IN V, a!=b AND a!=c AND a!=d AND b!=c AND b!=d AND c!=d AND y == {a,b,c,d}))" all -v i=minimal3.cfg

$PTH/flagcalc -d f="-abcdea -afghb -bhijc -cjkld -dlmne -enofa -pqrstp op gq ir ks mt" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg set allsets i=minimal3.cfg

# THREADED PARTITION

$PTH/flagcalc -d f="-abcdea -afghb -bhijc -cjkld -dlmne -enofa -pqrstp op gq ir ks mt" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg set allsets i=minimal3.cfg
$PTH/flagcalc -d f="abc def abd bce caf" -a e="NAMING (A AS Automs, NAMING (C AS Cycless, NAMING (m AS MAX (c IN C, st(c)), NAMING (hs AS SET (c IN C, st(c) == m, c), THREADED PARTITION (h1, h2 IN hs, EXISTS (a IN A, EXISTS (j IN NN(m), FORALL (i IN NN(m), a[h1[m - i - 1]] == h2[mod(i+j,m)]) OR FORALL (i IN NN(m), a[h1[i]] == h2[mod(i+j,m)]))) )))))" all -v measg set allsets i=minimal3.cfg


$PTH/flagcalc -d f="abc" -a p="SORT (s, t IN Ps(V), st(s) > st(t))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abc abd bcf cah ahi aid bde bef cfg cgh jkl dij dej efk fgk ghl hil" -a p="SORT (s, t IN Ps(V), st(s) < st(t))" all -v i=minimal3.cfg
$PTH/flagcalc -d f="a bc def ghij" -a p="SORT (s,t IN Ps(V), st(s) == cliquem(SubgraphonUg(s)) && st(t) == cliquem(SubgraphonUg(t)), st(s) < st(t))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="abcd" -a e="THREADED PARTITION (u,v IN Ps(V), st(u) % 2 == 0, st(u)==st(v)) " all -v measg set allsets i=minimal3.cfg

# a few chosen from earlier in the script, now THREADED
$PTH/flagcalc -r 11 10 1 -a s="THREADED FORALL (n IN NN(dimm), EXISTS (s IN Ps(V), EXISTS (t IN Ps(V), (s CUP t) == V AND st(s) == ceil(dimm/(n+1)) AND st(t) == floor(n*dimm/(n+1)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 11 10 1 -a s="FORALL (n IN NN(dimm), THREADED EXISTS (s IN Ps(V), EXISTS (t IN Ps(V), (s CUP t) == V AND st(s) == ceil(dimm/(n+1)) AND st(t) == floor(n*dimm/(n+1)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 15 1 -a s="THREADED FORALL (s IN Ps(V), s == V OR EXISTS (t IN Ps(V), EXISTS (u IN Ps(V), (NOT (u <= s)) AND (NOT (t <= s)) AND ((s CAP t) <= s))))" all
$PTH/flagcalc -d testbip12.dat -a s="THREADED EXISTS (n IN NN(dimm+1), EXISTS (l IN Sizedsubset(V,n), EXISTS (r IN Sizedsubset(V,dimm - n), st(r CUP l) == dimm AND FORALL (a IN l, FORALL (b IN l, NOT ac(a,b))) AND FORALL (c IN r, FORALL (d IN r, NOT ac(c,d))))))" all

$PTH/flagcalc -r 11 10 1 -a e="THREADED SET (n IN dimm, SET (s IN Ps(V), t IN Ps(V), (s CUP t) == V AND st(s) == ceil(dimm/(n+1)) AND st(t) == floor(n*dimm/(n+1)), {s,t}))" all -v i=minimal3.cfg
$PTH/flagcalc -r 11 10 1 -a e="THREADED SET (n IN dimm, SET (s IN Ps(V), t IN Ps(V), (s CUP t) == V AND st(s) == n AND st(t) == dimm-n, {s,t}))" all -v i=minimal3.cfg
$PTH/flagcalc -r 11 10 1 -a e="SET (n IN dimm, THREADED SET (s IN Ps(V), SET (t IN Ps(V), (s CUP t) == V AND st(s) == ceil(dimm/(n+1)) AND st(t) == floor(n*dimm/(n+1)), {s,t})))" all -v i=minimal3.cfg
$PTH/flagcalc -r 11 10 1 -a e="SET (n IN dimm, THREADED SETD (s IN Ps(V), SETD (t IN Ps(V), (s CUPD t) == V AND st(s) == n AND st(t) == dimm-n, {s,t})))" all -v i=minimal3.cfg

$PTH/flagcalc -r 150 75 1 -a e="THREADED PARTITION (u,v IN V, connvc(u,v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 50 20 1 -a e="PARTITION (u,v IN V, connvc(u,v))" all -v set allsets i=minimal3.cfg

# 4:42 runtime 4/30/2025 on i9 13900hx

# 5:36 runtime 5/1/2025 on same

# 5:57 runtime 5/1/2025 on same

# 5:23 runtime 5/1/2025 after idealizeset

# 4:20 on a Threadripper 24 core Shimada Peak series 12/12/2025: update: 4:00 on 12/18/2025

# Diestel Cor. 1.5.2
$PTH/flagcalc -r 9 p=0.2 100 -a s="treec" s2="EXISTS (P IN Perms(V), FORALL (v IN V, P[v] >= 1, EXISTS (n IN NN(dimm), P[n] < P[v] AND ac(n,v), FORALL (m IN NN(dimm), (P[m] < P[v] AND ac(m,v)) IMPLIES m == n))))" all -v set allsets i=minimal3.cfg

