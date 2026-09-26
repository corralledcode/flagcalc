PTH=${PTH:-'../bin'}

$PTH/flagcalc -r 4 p=0.5 2 -a isp=storedprocedures.dat e="SETD (g1 IN BIN(\"r0\"), g2 IN BIN(\"r0\"), <<g1, g2, strongproduct(g1,g2)>>)" all -v set allsets graphs crit

$PTH/flagcalc -r 10 p=0.5 2 -a isp=storedprocedures.dat e="SETD (g1 IN BIN(\"r0\"), g2 IN BIN(\"r0\"), <<g1, g2, strongproduct(g1,g2), Lovaszthetam(g1), Lovaszthetam(g2), Lovaszthetam(strongproduct(g1,g2))>>)" once -v i=minimal3.cfg set allsets
$PTH/flagcalc -r 10 p=0.5 4 -a isp=storedprocedures.dat e="THREADED SETD (g1 IN BIN(\"r0\"), g2 IN BIN(\"r0\"), <<Lovaszthetam(g1), Lovaszthetam(g2), Lovaszthetam(strongproduct(g1,g2))>>)" once -v i=minimal3.cfg set allsets

$PTH/flagcalc -r 8 p=0.5 2 -a isp=storedprocedures.dat e="BIN(\"r0\")" once -a e="BIN(\"a0\")" once -v i=minimal3.cfg set allsets
$PTH/flagcalc -r 8 p=0.5 2 -a isp=storedprocedures.dat e="SETD (g1 IN BIN(\"r0\"), g2 IN BIN(\"r0\"), strongproduct(g1,g2))" once -a e="SETD (g IN BIN(\"a0\"), Lovaszthetam(g))" once -v i=minimal3.cfg set allsets

$PTH/flagcalc -r 10 p=0.5 3 -a isp=storedprocedures.dat e="SETD (g1 IN BIN(\"r0\"), g2 IN BIN(\"r0\"), strongproduct(g1,g2))" once -a e="SETD (g IN BIN(\"a0\"), Lovaszthetam(g))" once -v i=minimal3.cfg set allsets

# $PTH/flagcalc -r 10 p=0.6 8 -a isp=storedprocedures.dat e="THREADED SETD (g1 IN BIN(\"r0\"), g2 IN BIN(\"r0\"), <<g1, g2, strongproduct(g1,g2)>>)" once -a s="FORALL (triple IN BIN(\"a0\"), Lovaszthetam(triple[0])*Lovaszthetam(triple[1]) == Lovaszthetam(triple[2]))" all -v i=minimal3.cfg
