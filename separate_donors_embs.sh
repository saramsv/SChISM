names="UT100 UT102 UT106 UT108 UT109 UT110 UT111"
for name in $names
do
    #grep  $name $1 > 'ut1all_2011/'$name'_incept'
    grep  $name $1 > 'ut1all_2011/'$name'_gt'
done
#run:  bash separate_donors_embs.sh ut1all_2011/2011_ut1_all_correct_classes
