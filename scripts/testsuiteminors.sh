PTH=${PTH:-'../bin'}

# Should be true
$PTH/flagcalc -d f="-abcdefgha -cijkf -aid" -a s="hastopologicalminorc(\"-abcda -bd ac\")" all 

# Should be true
$PTH/flagcalc -d f="-abcdefgha -cijkf" -a s="NOT hastopologicalminorc(\"-abcda -bd ac\")" all

# intractable
# $PTH/flagcalc -d f="-abcdefgha -cijkf" -a isp="../testgraph/storedprocedures.dat" s="HasHminor(\"-abcda -bd ac\")" all

# Should be true (copied from Diestel p. 19)
$PTH/flagcalc -d f="-abcdefgha -cijkf" -a s="hastopologicalminorc(\"-abcda -bd\")" all 

# intractable
# $PTH/flagcalc -d f="-abcdefgha -cijkf" -a isp="../testgraph/storedprocedures.dat" s="HasHminor(\"-abcda -bd\")" all

# Should be true
$PTH/flagcalc -d f="abcd -cefghijd" -a s="hastopologicalminorc(\"-abcdea f\")" all

# Should be true
$PTH/flagcalc -d f="-abcdef -bg -chi -djkl" -a s="NOT hastopologicalminorc(\"-abcdef -bghi -cjk -dl\")" all

# Should be true (testing if vertices are properly reversed during the search)
$PTH/flagcalc -d f="-abcdef -bg -chi -djkl -am" -a s="hastopologicalminorc(\"-abcdef -bghi -cjk -dl\")" all

# Should be about half in the first query and all TRUE in the second
$PTH/flagcalc -r 10 p=0.177 2000 -a s="forestc" s="hastopologicalminorc(\"abc\") IFF NOT forestc" all -v i=minimal3.cfg

# Same as above, more vertices
$PTH/flagcalc -r 15 p=0.11 500 -a s="forestc" s="hastopologicalminorc(\"abc\") IFF NOT forestc" all -v i=minimal3.cfg

# Should be true
$PTH/flagcalc -d f="-abcdefgha -cijkf -aid" -a s="hastopologicalminorc4(\"-abcda -bd ac\")" all

# Should be true
$PTH/flagcalc -d f="-abcdefgha -cijkf" -a s="NOT hastopologicalminorc4(\"-abcda -bd ac\")" all

# intractable
# $PTH/flagcalc -d f="-abcdefgha -cijkf" -a isp="../testgraph/storedprocedures.dat" s="HasHminor(\"-abcda -bd ac\")" all

# Should be true (copied from Diestel p. 19)
$PTH/flagcalc -d f="-abcdefgha -cijkf" -a s="hastopologicalminorc4(\"-abcda -bd\")" all

# intractable
# $PTH/flagcalc -d f="-abcdefgha -cijkf" -a isp="../testgraph/storedprocedures.dat" s="HasHminor(\"-abcda -bd\")" all

# Should be true
$PTH/flagcalc -d f="abcd -cefghijd" -a s="hastopologicalminorc4(\"-abcdea f\")" all

# Should be true
$PTH/flagcalc -d f="-abcdef -bg -chi -djkl" -a s="NOT hastopologicalminorc4(\"-abcdef -bghi -cjk -dl\")" all

# Should be true (testing if vertices are properly reversed during the search)
$PTH/flagcalc -d f="-abcdef -bg -chi -djkl -am" -a s="hastopologicalminorc4(\"-abcdef -bghi -cjk -dl\")" all

# Should be about half in the first query and all TRUE in the second
$PTH/flagcalc -r 10 p=0.177 2000 -a s="forestc" s="hastopologicalminorc4(\"abc\") IFF NOT forestc" all -v i=minimal3.cfg

# Same as above, more vertices
#$PTH/flagcalc -r 15 p=0.11 500 -a s="forestc" s="hastopologicalminorc4(\"abc\") IFF NOT forestc" all -v i=minimal3.cfg

# Should be true
$PTH/flagcalc -d f="-abcda ag bh ci dj" -a s="NOT hastopologicalminorc4(\"-abcdea\")" all -v crit allcrit set i=minimal3.cfg rt

$PTH/flagcalc -d f="-abcdef -ghijc" -a s="NOT hastopologicalminorc4(\"abc\")" all -v crit allcrit set i=minimal3.cfg rt

# should be true
$PTH/flagcalc -r 8 p=0.25 100 -a s="hastopologicalminorc4(\"-abcda\") IFF hastopologicalminorc(\"-abcda\")" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 7 p=0.5 10 -a s="hastopologicalminorc4(\"ab=cde\") IFF hastopologicalminorc(\"ab=cde\")" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 8 p=0.25 25 -a s="hastopologicalminorc4(\"abc=def\") IFF hastopologicalminorc(\"abc=def\")" all -v i=minimal3.cfg

# should be 18 out of 20 (note three-fold faster with this order of the two OR'ed statements)
$PTH/flagcalc -d testplanarshort.dat testplanarsmall.dat -a s="hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\")" all -v i=minimal3.cfg

# should be true (i.e. 16-cell graph is non-planar)

$PTH/flagcalc -d f="a+bcefgh  b+defgh c+defgh d+efgh -efghe" -a s="hastopologicalminorc4(\"abc=def\") OR hastopologicalminorc4(\"abcde\")" all -v i=minimal3.cfg

# should be true (a relic from a bug fix)
$PTH/flagcalc -d f="ab bc ce ef fg ga ae bf bg cf eg" -a s="NOT embedsgenerousc(\"abc=def\")" -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 8 p=0.25 25 -a s="embedsgenerousc(\"-abcda\") IFF (embedsc(\"-abcda\") \
OR embedsc(\"-abcda ac\") OR embedsc(\"abcd\"))" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 10 p=0.5 500 -a s1="embedsgenerousc(\"abc=def\")" s2="embedsc(\"abc=def\") \
OR embedsc(\"abc=def ab\") OR embedsc(\"abc=def ab bc\") OR embedsc(\"abc=def abc\") \
OR embedsc(\"abc=def abc de\") OR embedsc(\"abc=def abc de ef\") OR embedsc(\"abc=def abc def\") \
OR embedsc(\"abc=def ab de\") OR embedsc(\"abc=def ab bc de\") OR embedsc(\"abc=def ab bc de ef\")" all -v i=minimal3.cfg

# clearly TRUE
$PTH/flagcalc -r 10 p=0.5 500 -a s1="embedsc(\"abc=def\") \
OR embedsc(\"abc=def ab\") OR embedsc(\"abc=def ab bc\") OR embedsc(\"abc=def abc\") \
OR embedsc(\"abc=def abc de\") OR embedsc(\"abc=def abc de ef\") OR embedsc(\"abc=def abc def\") \
OR embedsc(\"abc=def ab de\") OR embedsc(\"abc=def ab bc de\") OR embedsc(\"abc=def ab bc de ef\")" \
s2="embedsgenerousc(\"abc=def\")" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 12 p=0.5 500 -a s="embedsgenerousc(\"abc=def\") IFF (embedsc(\"abc=def\") \
OR embedsc(\"abc=def ab\") OR embedsc(\"abc=def ab bc\") OR embedsc(\"abc=def abc\") \
OR embedsc(\"abc=def abc de\") OR embedsc(\"abc=def abc de ef\") OR embedsc(\"abc=def abc def\") \
OR embedsc(\"abc=def ab de\") OR embedsc(\"abc=def ab bc de\") OR embedsc(\"abc=def ab bc de ef\"))" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 16 p=0.125 1500 -a s="embedsgenerousc(\"-abcda\") IFF (embedsc(\"abcd\") \
OR embedsc(\"-abcda ac\") OR embedsc(\"-abcda\"))" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 12 p=0.125 500 -a s="hastopologicalminorc4(\"-abcda\") IF (cr1 AND NOT forestc)" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 12 p=0.125 500 -a s1="NOT forestc" s2="cr1" s3="hastopologicalminorc4(\"-abcda\")" all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 12 p=0.125 250 -a s1="(NOT forestc) AND NOT hastopologicalminorc4(\"-abcda\")" s2="NOT cr1" all -v i=minimal3.cfg

# should be true (Diestel example graph p. 20)
$PTH/flagcalc -d f="ab ag ah bc bh cd ch de ef eh fg" -a s="hasminorc(\"-abcda bd\")" all -v i=minimal3.cfg

# should be true (implied by Diestel Prop 1.7.3)
$PTH/flagcalc -r 10 p=0.175 50 -a s="hasminorc(\"-abcda bd\") IFF hastopologicalminorc4(\"-abcda bd\")" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 p=0.215 50 -a s="hasminorc(\"-abcda ae be ce\") IFF hastopologicalminorc4(\"-abcda ae be ce\")" all -v i=minimal3.cfg

# should be true (i.e. the Petersen graph has a K_5 minor but not a K_5 topological minor)
$PTH/flagcalc -d f="-abcdea -fhjgif af bg ch di ej" -a s="hasminorc(\"abcde\")" s="NOT hastopologicalminorc4(\"abcde\")" all -v i=minimal3.cfg

# should both be true
$PTH/flagcalc -d f="-abcdefa -ad ae bf ce df" -a s="hasminorc(\"abcde\")" s="NOT hastopologicalminorc4(\"abcde\")" all -v i=minimal3.cfg

# should be 18 out of 20 (note three-fold faster with this order of the two OR'ed statements)
$PTH/flagcalc -d testplanarshort.dat testplanarsmall.dat -a s="hasminorc(\"abcde\") OR hasminorc(\"abc=def\")" all -v i=minimal3.cfg

# should be true (Diestel Lemma 4.4.2) (note in the following, runtimes vary widely, hence doing only 15)
$PTH/flagcalc -r 8 p=0.5 50 -a s="(hasminorc(\"abcde\") OR hasminorc(\"abc=def\")) IFF (hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\"))" all -v i=minimal3.cfg
$PTH/flagcalc -r 9 p=0.4 5 -a s="(hasminorc(\"abcde\") OR hasminorc(\"abc=def\")) IFF (hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\"))" all -v i=minimal3.cfg

# should match numbers, testing relative runtimes
$PTH/flagcalc -r 10 p=0.3 5 -a s="hasminorc(\"abcde\") OR hasminorc(\"abc=def\")" all -a s="hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\")" all -v i=minimal3.cfg

# should be partly true partly false (testing adjacent queries for run times)
$PTH/flagcalc -r 8 p=0.5 150 -a s="hasminorc(\"abcde\") OR hasminorc(\"abc=def\")" all -g o=out.dat overwrite all -v i=minimal3.cfg
$PTH/flagcalc -d out.dat -a s="hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\")" all -v i=minimal3.cfg

# -d out.dat -a s1="hasminorc(\"abcde\") OR hasminorc(\"abc=def\")" s="NOT (hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\"))" s="NOT (hastopologicalminorc(\"abcde\") OR hastopologicalminorc(\"abc=def\"))" all -v i=minimal3.cfg
# -d f="ab ac ad ae af be bg cf cg df dg ef fg" -a s1="hasminorc(\"abcde\") OR hasminorc(\"abc=def\")" s="NOT (hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\"))" s="NOT (hastopologicalminorc(\"abcde\") OR hastopologicalminorc(\"abc=def\"))" all -v i=minimal3.cfg
# -d f="ah ac ad ae af he hg cf cg df dg ef fg" -a s1="hasminorc(\"abcde\") OR hasminorc(\"abc=def\")" s="NOT (hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\"))" s="NOT (hastopologicalminorc(\"abcde\") OR hastopologicalminorc(\"abc=def\"))" all -v i=minimal3.cfg
# ab ac ad ae af bc bd be cf df ef

# -d f="ab ac ad ae af bc bd be cf df ef" -a s="embedsgenerousc(\"abc=def\")"  all -v i=minimal3.cfg

# should be true
$PTH/flagcalc -r 12 p=0.2 50 -a s="st(cyclest) > 0 IFF hasminorc(\"abc\")" all -v i=minimal3.cfg

# Diestel Prop 7.2.1
$PTH/flagcalc -r 10 20 250 -a s1="dm >= 4" s2="hasminorc(\"abcd\")" all -v i=minimal3.cfg
$PTH/flagcalc -r 10 40 250 -a s1="dm >= 8" s2="hasminorc(\"abcde\")" all -v i=minimal3.cfg
$PTH/flagcalc -r 12 48 250 -a s1="dm >= 8" s2="hasminorc(\"abcde\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 16 64 15 -a s1="dm >= 8" s2="hasminorc(\"abcde\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 24 96 500 -a s1="dm >= 8" s2="hasminorc(\"abcde\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 32 128 500 -a s1="dm >= 8" s2="hasminorc(\"abcde\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 32 256 500 -a s1="dm >= 16" s2="hasminorc(\"abcdef\")" all -v i=minimal3.cfg


# Diestel Prop 7.2.2
$PTH/flagcalc -r 10 40 25 -a s1="dm >= 8" s2="hastopologicalminorc4(\"abc\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 100 400 250 -a s1="dm >= 8" s2="hastopologicalminorc4(\"abc\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 100 3200 25 -a s1="dm >= 64" s2="hastopologicalminorc4(\"abcd\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 125 4000 250 -a s1="dm >= 64" s2="hastopologicalminorc4(\"abcd\")" all -v i=minimal3.cfg
#$PTH/flagcalc -r 150 4800 2500 -a s1="dm >= 64" s2="hastopologicalminorc4(\"abcd\")" all -v i=minimal3.cfg

# -d out2.dat -a s="hasminorc(\"-abcda bd\")" s="hastopologicalminorc4(\"-abcda bd\")" s="hastopologicalminorc(\"-abcda bd\")" all -v i=minimal3.cfg
# -d out2.dat -a s="hasminorc(\"abcde\") OR hasminorc(\"abc=def\")" all -a s="hastopologicalminorc4(\"abcde\") OR hastopologicalminorc4(\"abc=def\")" all -a s="hastopologicalminorc(\"abcde\") OR hastopologicalminorc(\"abc=def\")" all -v i=minimal3.cfg
# -d out.dat -a s="hasminorc(\"-abcda bd\")" all -a s="hastopologicalminorc4(\"-abcda bd\")" all -a s="hastopologicalminorc(\"-abcda bd\")" all -v i=minimal3.cfg

# Diestel Prop 1.7.3
$PTH/flagcalc -r 9 p=0.4 1000 -a s1="hastopologicalminorc4(\"abcde\")" s2="hasminorc(\"abcde\")" all -v i=minimal3.cfg

$PTH/flagcalc -r 8 p=0.4 100 -a s1="hasminorc(\"abcde\")" s2="NOT hastopologicalminorc4(\"abcde\")" all -v i=minimal3.cfg
# -d f="a b c dg eg fg -ghijklmnop oq" -a s="hasminorc(\"-abc -bcdef eh ei\")" all -v i=minimal3.cfg
# -d f="-abc -ade c+fgh e+ijk" -a s="hasminorc(\"a+bcd e+fgh ae\")" all -v i=minimal3.cfg