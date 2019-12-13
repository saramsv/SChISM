num_cluster=7

resnet_file=UT57-14D_res
pca_file=pcaUT29-15

#python3 path2embeding.py --img_path UT57-14D_paths --weight_type pt  > $resnet_file

#sed -i 's/,//g;  s/\[//g; s/\]//g' $resnet_file
#sed -i "s/'//g" $resnet_file

#python3 resnet2pca.py --embeding_file $resnet_file > $pca_file


#sed -i 's/,//g;  s/\[//g; s/\]//g' $pca_file
#sed -i "s/'//g" $pca_file

python3 unsupervised_clustering.py --embeding_file $pca_file --modify false --daily false  --cluster_number $num_cluster --method merge --merge_type single --ADD_file imgname2weather > $oca_file"_seq"

sed -i 's/\s*_\s*/_/g' "simple_"$num_cluster"Cl_kmens"
