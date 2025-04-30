#
PTH='../cmake-build-debug'

$PTH/flagcalc -r 150 75 1 -a s="THREADED PARTITION (u,v IN V, connvc(u,v)) == PARTITION (u,v IN V, connvc(u,v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 120 70 1 -a s="st(THREADED PARTITION (u,v IN V, connvc(u,v))) == connm" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 120 70 1 -a s="st(PARTITION (u,v IN V, connvc(u,v))) == connm" all -v set allsets i=minimal3.cfg

