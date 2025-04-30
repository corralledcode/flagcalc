#
PTH='../cmake-build-debug'

$PTH/flagcalc -r 150 75 1 -a s="THREADED PARTITION (u,v IN V, connvc(u,v)) == PARTITION (u,v IN V, connvc(u,v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 120 70 1 -a s="st(THREADED PARTITION (u,v IN V, connvc(u,v))) == connm" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -r 120 70 1 -a s="st(PARTITION (u,v IN V, connvc(u,v))) == connm" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="ab cde fghi jklmn" -a e="THREADED PARTITION (u,v IN V, w AS u+10, w > 10, connvc(u,v))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="ab cde fghi jklmn" -a e="THREADED PARTITION (u,v IN V, w AS u+10, connvc(u,v))" all -v set allsets i=minimal3.cfg

$PTH/flagcalc -d f="a bc" -a e="THREADED PARTITION (a,b IN Ps(V), st(a) > 0, st(a) == st(b))" all -v set allsets i=minimal3.cfg
$PTH/flagcalc -d f="ab cde" -a s="THREADED PARTITION (a,b IN Ps(V), st(a) > 0, st(a) == st(b) == PARTITION (a,b IN Ps(V), st(a) > 0, st(a) == st(b)))" all -v set allsets i=minimal3.cfg

