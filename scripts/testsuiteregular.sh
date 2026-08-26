PTH=${PTH:-'../bin'}

$PTH/flagcalc -d testregular.dat -a isp="regular.dat" s="stronglyregular" s="regular" all -v i=minimal3.cfg crit allcrit

$PTH/flagcalc -d testregular.dat -a isp="regular.dat" s="stronglyregularofparams(16,5,0,2)" all -v i=minimal3.cfg

