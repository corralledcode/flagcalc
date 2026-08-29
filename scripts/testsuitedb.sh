PTH=${PTH:-'../bin'}

# A suite of Flagcalc invocations that produce reusable databases

$PTH/flagcalc -d graph4c.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip4c.fcg overwrite passed
$PTH/flagcalc -d graph5c.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip5c.fcg overwrite passed
$PTH/flagcalc -d graph6c.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip6c.fcg overwrite passed
$PTH/flagcalc -d graph7c.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip7c.fcg overwrite passed
$PTH/flagcalc -d graph8c.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip8c.fcg overwrite passed
$PTH/flagcalc -d graph9c.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip9c.fcg overwrite passed

$PTH/flagcalc -d graph4.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip4.fcg overwrite passed
$PTH/flagcalc -d graph5.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip5.fcg overwrite passed
$PTH/flagcalc -d graph6.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip6.fcg overwrite passed
$PTH/flagcalc -d graph7.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip7.fcg overwrite passed
$PTH/flagcalc -d graph8.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip8.fcg overwrite passed
$PTH/flagcalc -d graph9.fcg -a isp=storedprocedures.dat s=Bipartiteviacycles -g o=bip9.fcg overwrite passed
