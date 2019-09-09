num_cluster=10

#python3 path2embeding.py --img_path data/UT42-12D_paths --weight_type pt  > img2resnetEmbs

#sed -i 's/,//g;  s/\[//g; s/\]//g' img2resnetEmbs
#sed -i "s/'//g" img2resnetEmbs

#python3 resnet2pca.py --embeding_file img2resnetEmbs > img2PCA256Embs


#sed -i 's/,//g;  s/\[//g; s/\]//g' img2PCA256Embs
#sed -i "s/'//g" img2PCA256Embs

python3 embeding_clustering_full_kmeans.py --embeding_file img2PCA256Embs --modify false --daily false  --cluster_number $num_cluster --method notmerge --merge_type single > "simple_"$num_cluster"Cl"

sed -i 's/_ /_/g; s/ _ /_/g' "simple_"$num_cluster"Cl"

if [ -d "simple_"$num_cluster"Clusters" ] 
then
    echo "Directory "simple_"$num_cluster"Clusters" exist." 
else
    echo "Error: Directory does not exist."
    mkdir "simple_"$num_cluster"Clusters"
fi
bash make_html.sh "simple_"$num_cluster"Cl"
mv *html "simple_"$num_cluster"Clusters"
mv "simple_"$num_cluster"Cl" "simple_"$num_cluster"Clusters"


python3 embeding_clustering_full_kmeans.py --embeding_file img2PCA256Embs --modify true --daily false  --cluster_number $num_cluster --method notmerge --merge_type single > "modified_"$num_cluster"Cl"
sed -i 's/_ /_/g; s/ _ /_/g' "modified_"$num_cluster"Cl"

if [ -d "modified_"$num_cluster"Clusters" ] 
then
    echo "Directory "modified_"$num_cluster"Clusters" exist." 
else
    echo "Error: Directory does not exist."
    mkdir "modified_"$num_cluster"Clusters"
fi
bash make_html.sh "modified_"$num_cluster"Cl"
mv *html "modified_"$num_cluster"Clusters"
mv "modified_"$num_cluster"Cl" "modified_"$num_cluster"Clusters"

python3 embeding_clustering_full_kmeans.py --embeding_file img2PCA256Embs --modify false --daily false  --cluster_number $num_cluster --method merge --merge_type single > "dailyMergeSinle_"$num_cluster"Cl"

sed -i 's/ _ /_/g; s/_ /_/g ; s/ _/_/g' "dailyMergeSinle_"$num_cluster"Cl"

if [ -d "dailyMergeSinle_"$num_cluster"Clusters" ] 
then
    echo "Directory "dailyMergeSinle_"$num_cluster"Clusters" exist." 
else
    echo "Error: Directory does not exist."
    mkdir "dailyMergeSinle_"$num_cluster"Clusters"
fi

bash make_html.sh "dailyMergeSinle_"$num_cluster"Cl"
mv *html "dailyMergeSinle_"$num_cluster"Clusters"
mv "dailyMergeSinle_"$num_cluster"Cl" "dailyMergeSinle_"$num_cluster"Clusters"

python3 embeding_clustering_full_kmeans.py --embeding_file img2PCA256Embs --modify false --daily false  --cluster_number $num_cluster --method merge --merge_type single > "dailyMergemulti_"$num_cluster"Cl"
sed -i 's/ _ /_/g; s/ _/_/g; s/_ /_/g' "dailyMergemulti_"$num_cluster"Cl"


if [ -d "dailyMergemulti_"$num_cluster"Clusters" ] 
then
    echo "Directory "dailyMergemulti_"$num_cluster"Clusters" exist." 
else
    echo "Error: Directory does not exist."
    mkdir "dailyMergemulti_"$num_cluster"Clusters"
fi
bash make_html.sh "dailyMergemulti_"$num_cluster"Cl"
mv *html "dailyMergemulti_"$num_cluster"Clusters"
mv "dailyMergemulti_"$num_cluster"Cl" "dailyMergemulti_"$num_cluster"Clusters"
