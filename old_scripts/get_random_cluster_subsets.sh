#!/bin/bash
#bash get_random_cluster_subsets.sh clusters_file new_clusters_file how_many num_clusters
#bash get_random_cluster_subsets.sh 50000PCAed64_15ClusAll new 1000 15
i=0
cat $1 | grep ": $i" | shuf -n $3 > $2
for i in $(seq 1 1 $4) #start step end
do
    cat $1 | grep ": $i" | shuf -n $3 >> $2
done
