FILES=/data/sara/ImageSimilarityMultiMethods/3donors/ut45_16_57/*PCAed
for f in $FILES
do
    echo $f
    `python3 decom_unsupervised_clustering.py --embeding_file $f --cluster_number 9`
done
