PTH=${PTH:-'../bin'}

# Request 10,000 random graphs on six vertices, sort by fingerprint, and save to file one representative per iso class
$PTH/flagcalc -r 6 p=0.375 10000 -f all -g o=out6vertices.dat sorted overwrite -v fp Fp fpnone rt

# Read the data back in from the file above, and check that there are zero pairwise isomorphic
$PTH/flagcalc -d out6vertices.dat -f all -C all -v cmp rt

# See "-C" at work: check each pair for isomorphism (contrast with -R sample randomly-chosen pairs)
$PTH/flagcalc -r 8 p=0.5 300 -f all -C all -v cmp rt fp Fp fpnone

# new feature: see fingerprinting only those that pass the criteria "conn1c" (aka "connected graph")
$PTH/flagcalc -r 5 p=0.3 30000 -a s1="conn1c" all -f passed -v cmp rt fp Fp fpnone i=minimal3.cfg

# new feature: see pairwise check "C" for isomorphism only on those that pass the criteria "conn1c"
$PTH/flagcalc -r 8 p=0.5 30 -a s1="conn1c" all -C passed -v cmp rt fp Fp fpnone i=minimal3.cfg

# not a new feature, just not used much elsewhere: "sorted" to check conn1c once per fingerprint-equiv-class
$PTH/flagcalc -r 5 p=0.5 3000 -f all -a s1="conn1c" sorted -v cmp rt fp Fp fpnone i=minimal3.cfg

# some fun with Kuratowski's theorem on graph planarity
$PTH/flagcalc -r 7 p=0.75 3000 -f all -a s1="NOT (hasminorc(\"abcde\") OR hasminorc(\"abc=def\"))" sorted \
s1="NOT (hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\"))" sorted -v cmp rt fp Fp fpnone i=minimal3.cfg

# more elementary checks
$PTH/flagcalc -r 8 p=0.5 30000 -f all -a s1="embedsc(\"abc\") IFF NOT cr1" sorted -v cmp rt fp Fp fpnone i=minimal3.cfg

# more elementary checks
$PTH/flagcalc -r 8 p=0.5 30000 -f all -a s1="embedsc(\"abc\") IFF NOT cr1" all -a s1="embedsc(\"abc\") IFF NOT cr1" sorted -v cmp rt fp Fp fpnone i=minimal3.cfg

