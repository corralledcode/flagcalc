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

