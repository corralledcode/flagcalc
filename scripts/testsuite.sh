/home/peterglenn/CLionProjects/flagcalc/cmake-build-debug
PTH='../cmake-build-debug'
$PTH/flagcalc -R 10 18 1000
$PTH/flagcalc -d testbip12.dat -f all -i sorted
$PTH/flagcalc -d testgraph23.dat -i all
$PTH/flagcalc -r 6 8 200 -f all -i sortedverify -v i=minimal3.cfg
$PTH/flagcalc -r 12 30 100 -g o=out4.dat overwrite all -v i=minimal3.cfg
$PTH/flagcalc -d out4.dat -a s="[conn1c]" s2="[forestc] AND NOT [treec]" all -g o=out5.dat passed overwrite -v i=minimal3.cfg
$PTH/flagcalc -d testgraph5.dat testgraph4.dat testgraph30.dat -f all -i sorted
$PTH/flagcalc -r 10 18 100 -a a="[dm] * [dimm] == 2*18" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 18 2000 -a c=cr1 s2="[edgecm] > [dimm]^2/4" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 15 3000 -a ft="abcd" s="[cliquem] < 4" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 15 2000 -a s="[Deltam] > 2" s2="[dimm] < [Deltam]/([Deltam]-2) * ([Deltam] - 1)^[radiusm]" all -g o=out6.dat overwrite passed -v i=minimal3.cfg
$PTH/flagcalc -d out6.dat -a nf="embeddings.dat" m2=girthm m2=circm all -v i=minimal3.cfg
$PTH/flagcalc -d out6.dat -a is="sentence.dat" s2="[radiusm] <= [diamm] AND [diamm] <= 2*[radiusm]" all -v i=minimal3.cfg
$PTH/flagcalc -d out4.dat -a ia="sentence.dat" s2="[diamc](2)" all -v i=minimal3.cfg
$PTH/flagcalc -d out4.dat -a s="[Knc]([cliquem])" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 18 100 -a a="[radiusm]" s="[connc]([radiusm])" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 18 100 -a a="[radiusm]" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 12 50000 -a c=cr1 s2="[cyclet](5) <= 2^5" all -v i=minimal3.cfg

