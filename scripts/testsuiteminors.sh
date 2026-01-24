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
$PTH/flagcalc -r 15 p=0.11 500 -a s="forestc" s="hastopologicalminorc4(\"abc\") IFF NOT forestc" all -v i=minimal3.cfg

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


