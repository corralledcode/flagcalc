PTH=${PTH:-'../bin'}

# Should be sqrt(5) == 2.3607
$PTH/flagcalc -d f="-abcdea" -a a="Lovaszthetam"

# Should be 3
$PTH/flagcalc -d f="-abcdefa" -a a="Lovaszthetam"

# Should be 2.0
$PTH/flagcalc -d f="-abcda" -a a="Lovaszthetam"

# Lovasz's Sandwich Theorem (1979)

$PTH/flagcalc -r 10 p=0.5 100 -a s="NAMING (theta AS Lovaszthetam(Complementg), cliquem <= theta AND theta <= Chit)" -v i=minimal3.cfg

$PTH/flagcalc -r 12 p=0.5 1000 -a a="Lovaszthetam" -v i=minimal3.cfg

$PTH/flagcalc -d testplanar.dat -a z="cliquem(Complementg)" a="Lovaszthetam" z="Chigreedyt(Complementg)" -v allmeas crit alltally

$PTH/flagcalc -d testplanar.dat -a z="cliquem" a="Lovaszthetam(Complementg)" z="Chigreedyt" -v allmeas crit alltally

$PTH/flagcalc -d testplanar.dat -a s="NAMING (theta AS Lovaszthetam(Complementg), cliquem <= theta AND theta <= Chit)" -v allmeas crit alltally rt

# Vertex transitive graphs

$PTH/flagcalc -d testplanar.dat -a isp=storedprocedures.dat s="vertextransitive" -v allmeas crit alltally
$PTH/flagcalc -d f="a" -a isp=storedprocedures.dat s="FORALL (n IN NN(47), vertextransitive(Cg(n+3)))" -v i=minimal3.cfg
$PTH/flagcalc -d f="a" -a isp=storedprocedures.dat s="FORALL (n IN NN(47), Lovaszthetam(Cg(n+3))*Lovaszthetam(Complementg(Cg(n+3))) == n+3)" -v i=minimal3.cfg
$PTH/flagcalc -d f="a" -a isp=storedprocedures.dat s="FORALL (n IN NN(49), Lovaszthetam(Kg(n+1))*Lovaszthetam(Complementg(Kg(n+1))) == n+1)" -v i=minimal3.cfg
$PTH/flagcalc -d ./testgraph/hypercube/hypercube1.fcg -d ./testgraph/hypercube/hypercube2.fcg -d ./testgraph/hypercube/hypercube3.fcg -d ./testgraph/hypercube/hypercube4.fcg -a isp=storedprocedures.dat s="vertextransitive" -v i=minimal3.cfg
# -d ./testgraph/hypercube/hypercube5.fcg -d ./testgraph/hypercube/hypercube6.fcg -d ./testgraph/hypercube/hypercube7.fcg -d ./testgraph/hypercube/hypercube8.fcg
# doesn't compute for hypercuben.fcg n > 6, probably due to limitations of the semidefinite solver
$PTH/flagcalc -d ./testgraph/hypercube/hypercube1.fcg -d ./testgraph/hypercube/hypercube2.fcg -d ./testgraph/hypercube/hypercube3.fcg -d ./testgraph/hypercube/hypercube4.fcg -d ./testgraph/hypercube/hypercube5.fcg -d ./testgraph/hypercube/hypercube6.fcg -a isp=storedprocedures.dat s="Lovaszthetam*Lovaszthetam(Complementg) == dimm" -v i=minimal3.cfg

$PTH/flagcalc -d testplanar.dat -a isp=storedprocedures.dat s="vertextransitive" s2="Lovaszthetam*Lovaszthetam(Complementg) == dimm" -v allmeas crit alltally rt
$PTH/flagcalc -r 6 p=0.5 100 -a isp=storedprocedures.dat s="vertextransitive" s2="Lovaszthetam*Lovaszthetam(Complementg) == dimm" -v i=minimal3.cfg

