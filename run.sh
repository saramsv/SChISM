NAMES="UT100 UT102 UT106 UT108 UT109 UT110 UT111"
for name in $NAMES
do
    f=final_clusters_ut1all_2011/$name
    GT_FILE=GTs/$name'_gt'
    echo $f
    #echo $GT_FILE
    #`python3 merge_gt_cluster_labels.py $GT_FILE $f > final_clusters_ut1all_2011/$name'merged'`
    echo `python3 eval_.py final_clusters_ut1all_2011/$name'merged'`
    echo `cat final_clusters_ut1all_2011/$name'merged' | cut -d ":" -f 2| sort -u| wc -l`
done
