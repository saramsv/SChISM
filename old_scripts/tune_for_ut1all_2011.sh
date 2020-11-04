FILES=("UT100") # "UT102" "UT106" "UT108" "UT109" "UT110" "UT111")
for f in ${FILES[*]}
do
    echo "data/ut1all_2011_usedForECCV20/PCAed/"$f"_incept_PCAed"
    `python3 decom_unsupervised_clustering.py --embeding_file "data/ut1all_2011_usedForECCV20/PCAed/"$f"_incept_PCAed" --cluster_number 9`
done

###NOTE: alphas, betas and windows should be the same values in the decom_sequence.py
## this is assuming that the decom_unsupervised_clustering.py has been run for these valuse and therefore the
## coresponding files exist

#alphas=(0.99) #(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
#betas=(0.7) #(0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
#windows=(4)
#for alpha in ${alphas[*]}
#do
#    for beta in ${betas[*]}
#    do
#        for w in ${windows[*]} 
#        do
#            for f in ${FILES[*]}
#            do
#                name=$f'-11Dclusters_'$alpha'_'$beta'_'$w
#                `python3 merge_gt_cluster_labels.py "data/ut1all_2011_usedForECCV20/GTs/"$f"_gt" $name > $name"Cl_Gt_merged"`
#                `python3 eval_.py $name"Cl_Gt_merged" > "evaled_"$name"Cl_Gt_merged"`
#            done
#        done
#    done
#done
#
