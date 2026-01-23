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