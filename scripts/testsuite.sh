PTH=${PTH:-'../bin'}

# A dinosaur of a feature: chance of two randomly-chosen graphs from the workspace being isomorphic
$PTH/flagcalc -L 10 18 1000

# show that the three copies of K_6,6 are isomorphic (have same fingerprint)
$PTH/flagcalc -d testbip12.dat -f all -i sorted

# The isomorphisms of a cube (48, counting mirror images)
$PTH/flagcalc -d testgraph23.dat -i all

# Fingerprint 200 random graphs on 6 vertices, sort them by fingerprint (isomorphism) class,
# and verify fingerprint-equivalent graphs are isomorphic ("sortedverify")
$PTH/flagcalc -r 6 8 200 -f all -i sortedverify -v i=minimal3.cfg

# populate out4.dat with 100 random graphs on 12 vertices
$PTH/flagcalc -r 12 30 100 -g o=out4.dat overwrite all -v i=minimal3.cfg

# read back in what was just output, then test for the graph being 1-connected, and among those that pass, test
# for it being a forest that is not a tree (i.e. should return 0 out of however many passed the first test
$PTH/flagcalc -d out4.dat -a s="conn1c" s2="forestc AND NOT treec" all -g o=out5.dat passed overwrite -v i=minimal3.cfg

# read in from the three files stated, and fingerprint and output isomorphism classes
# result should be: GRAPH3 < GRAPH1 < GRAPH1 == GRAPH0 < GRAPH3 == GRAPH2, and 12,8,32,2 automorphism counts
$PTH/flagcalc -d testgraph5.dat testgraph4.dat testgraph30.dat -f all -i sorted

# this is the handshake lemma
$PTH/flagcalc -r 10 18 100 -a a="dm * dimm == 2 * edgecm" all -v i=minimal3.cfg

# Mantel's Theorem: should return 0 out of 2000
$PTH/flagcalc -r 10 18 2000 -a c=cr1 s2="edgecm > dimm^2/4" all -v i=minimal3.cfg

# abcd is a K_4; "ft" tests how many ways K_4 embeds in the graph; cliquem returns the highest clique found in the graph
# altogether, the cliquem < 4 should be true for exactly the number of ft's returning result==0
# Note as well: this run is on 3000 random graphs on 10 vertices
$PTH/flagcalc -r 10 15 3000 -a ft="abcd" s="cliquem < 4" all -v i=minimal3.cfg

# Diestel prop 1.3.3. Note we output to out6.dat all the evidentiary graphs found
# (that is, those that past the first "iteration" Deltam > 2 ("max degree > 2")
# and that pass the seccond fact, which should be ALL of them that passed first round)
$PTH/flagcalc -r 10 15 20000 -a s="Deltam > 2" s2="dimm < Deltam/(Deltam-2) * (Deltam - 1)^radiusm" all -g o=out6.dat overwrite passed -v i=minimal3.cfg

# This starts with the output of the previous query, then loads embeddings.dat and for those that
# do NOT embed a 3-cycle, a 4-cycle, or a 5-cycle, it checkes their girth and circumference. No real point, just a random query
$PTH/flagcalc -d out6.dat -a nf="embeddings.dat" m2=girthm m2=circm all -v i=minimal3.cfg

# Diestel p. 9
$PTH/flagcalc -d out6.dat -a is="sentence.dat" s2="radiusm <= diamm AND diamm <= 2*radiusm" all -v i=minimal3.cfg

# This runs three measures (the three found in sentence.dat) against the contents of out4.dat populated by an above query
# If any are trees, it checks if their diameter is greater than 2. No real point, just a random query
$PTH/flagcalc -d out4.dat -a ia="sentence.dat" s2="diamc(2)" all -v i=minimal3.cfg

# This should return all true, since it feeds cliquem right back into Knc
$PTH/flagcalc -d out4.dat -a s="Knc(cliquem,1)" all -v i=minimal3.cfg

# This is also rather meaningless: the first criterion is that the graph have finite radius (i.e. it be 1-connected)
# the second criterion asks if it has less than radiusm many connected components:
# almost always true except when radiusm = 1 (some sort of radial graph)
$PTH/flagcalc -r 10 18 100 -a s1="NOT isinf(radiusm)" s2="connc(radiusm)" all -v i=minimal3.cfg

# this checks the radius (defined in Diestel p. 9) of 100 random graphs on 10 vertices
# by choice of verbosity level, it only outputs the aggregate totals, min, max, and average
$PTH/flagcalc -r 10 18 100 -a a="radiusm" all -v i=minimal3.cfg

# This is a curious fact, that Google confirms but I forget where in Diestel I saw it... to look up later
$PTH/flagcalc -r 10 12 50000 -a c=cr1 s2="cyclet(5) <= 2^5" all -v i=minimal3.cfg

# Diestel 1.3.2: if the negation of forestc is true, then the "s2=" inequality holds
$PTH/flagcalc -r 12 20 5000 -a nc=forestc s2="girthm <= (2*diamm + 1)" all -v i=minimal3.cfg

# Diestel Cor 1.3.5 (corollary to Thm 1.3.4: Alon, Hoory, and Linial (2002))
$PTH/flagcalc -r 12 20 50000 -a s="deltam >= 3" s2="girthm < 2*log(dimm)" all -v i=minimal3.cfg

# This is a fact from the study of trees: that being a tree is equivalent to having edgecount equal to dimension minus 1
# IF the graph is connected.
$PTH/flagcalc -r 12 25 50000 -a c=conn1c s2="treec == (edgecm == dimm-1)" all -v i=minimal3.cfg

# the feature -u is only briefly studied but potentially very useful to those studying the three obvious papers
# on "Flag Algebras", and were coded in effort to understand the steps of these papers
# Please see the -h output for a rundown of how to use -u.
$PTH/flagcalc -d testbip12.dat -u GRAPH0 n="" 'r=rs1(5,100000)' -a sub t='cyclet(4)' -v i=minimal3.cfg

# This is just checking what minimal3.cfg does to 15 random graphs on 12 vertices, namely, print nothing except runtime
$PTH/flagcalc -r 12 20 15 -v i=minimal3.cfg

# again, the reader is referred to the comment on the previous invocation of -u a few lines above this line
$PTH/flagcalc -r 20 25 1 -u GRAPH0 n="a b c" 'r=rs1(10,10000)' -a m=girthm c=conn1c c2=forestc c2=treec sub -v i=minimal3.cfg

# This asks: if a K_5,5 (complete bipartite graph on 5,5 vertices) is "6-connected" (Defined on Diestel page 11): false
$PTH/flagcalc -d testbip10.dat -a s="kconnc(6)" all -v i=minimal3.cfg

# In reference to the line immediately preceding, now the answer is True
$PTH/flagcalc -d testbip10.dat -a s="kconnc(5)" all -v i=minimal3.cfg

# Diestel Prop 1.4.2
$PTH/flagcalc -r 10 12 1000 -a s="kappat <= deltam" all -v i=minimal3.cfg min subobj srm

# Diestel comment at the top of page 12
$PTH/flagcalc -r 10 12 10000 -a s="(kappat > 0) == conn1c" all -v i=minimal3.cfg min subobj srm

# This is the fact that a graph is a forest if the path count between any two vertices is at most 1
$PTH/flagcalc -r 8 7 1000 -a is=quantforestcrit.dat all -v i=minimal3.cfg

# This is showing an involved query (many nested quantifiers) spaced out with the luxury of an "is" (input sentence file)
# The query in question asks if a graph is "complete" bipartite (technically a bipartite graph need not equal the
# full K_n,m (see bipartite defn on Diestel p. 17_; it is passed testbip10, so the answer should be yes
$PTH/flagcalc -d testbip10.dat -a is=bipartitecrit2.dat all -v i=minimal3.cfg

# This is a non-bipartite graph, so the answer should be no
$PTH/flagcalc -d f="abcd=defg" -a is=bipartitecrit2.dat all -v i=minimal3.cfg

# This is a tripartite graph, so the answer should be no
$PTH/flagcalc -d f="abc=def=ghi" -a is=bipartitecrit2.dat all -v i=minimal3.cfg

# This just recodes cr1 ("triangle-free") using the first order logic;
# it also demonstrates iterated criteria: IF cr1 passes (triangle free) THEN apply the first-order query labelled "s2";
# clearly, therefore, all that pass criteria 1 should pass criteria 2
$PTH/flagcalc -r 10 12 1000 -a c=cr1 s2="FORALL (x IN V, FORALL (y IN V, FORALL (z IN V, NOT (ac(x,y) AND ac(x,z) AND ac(y,z)))))" all -v i=minimal3.cfg

# This should be false; the two notions should be equal, not unequal
$PTH/flagcalc -r 10 12 1000 -a s="cr1 != FORALL (x IN V, FORALL (y IN V, FORALL (z IN V, NOT (ac(x,y) AND ac(x,z) AND ac(y,z)))))" all -v i=minimal3.cfg

# This is a most elementary test that Sizedsubset returns V itself when asked for subsets of dimension dimm
$PTH/flagcalc -r 15 10 200 -a s="FORALL (s IN Sizedsubset(V,dimm), s == V)" all -v i=minimal3.cfg

# This illustrates a FALSE claim on the line one (a 3-sized subset cannot have 4 distinct elements) and a TRUE claim on line 2
# (Please note these are verging into more set-theoretic points, since not using graph theory per se)
$PTH/flagcalc -r 8 4 100 -a s="EXISTS (y IN Sizedsubset(V,3), EXISTS (v IN V, EXISTS (u IN V, EXISTS (t IN V, EXISTS (w IN V, t ELT y AND u ELT y AND v ELT y AND w ELT y AND t != u AND t != v AND u != v AND t != w AND u != w AND v != w)))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 4 100 -a s="EXISTS (y IN Sizedsubset(V,4), EXISTS (v IN V, EXISTS (u IN V, EXISTS (t IN V, EXISTS (w IN V, t ELT y AND u ELT y AND v ELT y AND w ELT y AND t != u AND t != v AND u != v AND t != w AND u != w AND v != w)))))" all -v i=minimal3.cfg

# Again, just basic set theory; line one should return all TRUE, and line two should return all TRUE as well
$PTH/flagcalc -r 8 10 10 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), (s CUP t) == V IMPLIES FORALL (x IN V, x ELT (s CUP t))))" all -v i=minimal3.cfg
$PTH/flagcalc -r 8 10 5 -a s="FORALL (s IN Ps(V), FORALL (t IN Ps(V), (s CUP t) != V IMPLIES EXISTS (x IN V, NOT (x ELT (s CUP t)))))" all -v i=minimal3.cfg

# runtime 19 seconds on 5/12/2025 on an i9




