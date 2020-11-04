FILES=ut1all_2011/*incept
for f in $FILES
do
    echo $f
    echo $f'_PCAed'
    `python3 resnet2pca.py --embeding_file $f > $f'_PCAed'`
done
