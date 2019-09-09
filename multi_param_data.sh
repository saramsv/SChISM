#embedding file is PCAed
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/newPCA/50000PCAed64modified_Daily7Clus
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/newPCA/50000PCAed64_Daily7Clus
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/newPCA/50000PCAed64modified_7ClusAll
python3 d_embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify false --daily false --method agglomerative --cluster_number 7 > d/PCAed64_Clusters/PCAed64_7C_agglomverative


#embedding file is resnet
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnetModified_DailyClus 
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnet_DailyClus 
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnetModified_Clus
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify false --daily false --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnet_Clus


#embedding file is PCAed fine tuned
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_modified_Daily7ClusAll  
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_Daily7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_modified_7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify false --daily false --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_7ClusAll 


#embedding file is resnet fine tuned
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000resnet_FT --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/50000resnet_FT_modified_Daily7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000resnet_FT --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/50000resnet_FT_Daily7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000resnet_FT --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/50000resnet_FT_modified_7ClusAll 
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_FT_emb/NoStakesPlasticPCAed64_FT_emb --modify false --daily false --cluster _number 7 --cluster_type agglomerative > d/PCAed64_Clusters/PCAed64_FT_7C

#embedding file is PCAed
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/newPCA/50000PCAed64modified_Daily7Clus
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/newPCA/50000PCAed64_Daily7Clus
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/newPCA/50000PCAed64modified_7ClusAll
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_emb/noStakesPlasticPCAed64 --modify false --daily false --cluster _number 7 --cluster_type agglomerative > d/PCAed64_Clusters/PCAed64_7C


#embedding file is resnet
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnetModified_DailyClus 
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnet_DailyClus 
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnetModified_Clus
#python3 embeding_clustering_full.py --embeding_file d/resnet_emb/noStakesPlasticResnet --modify false --daily false --cluster _number 7 --cluster_type agglomerative > d/resnet_Clusters/resnet_Clus


#embedding file is PCAed fine tuned
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_modified_Daily7ClusAll  
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_Daily7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_modified_7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000PCAed_FT --modify false --daily false --cluster _number 7 --cluster_type agglomerative > d/50000PCAed64_FT_7ClusAll 


#embedding file is resnet fine tuned
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000resnet_FT --modify true --daily true --cluster _number 7 --cluster_type agglomerative > d/50000resnet_FT_modified_Daily7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000resnet_FT --modify false --daily true --cluster _number 7 --cluster_type agglomerative > d/50000resnet_FT_Daily7ClusAll 
#python3 embeding_clustering_full.py --embeding_file data/50000imgs_clustering_data/50000resnet_FT --modify true --daily false --cluster _number 7 --cluster_type agglomerative > d/50000resnet_FT_modified_7ClusAll 
#python3 embeding_clustering_full.py --embeding_file d/PCAed64_FT_emb/NoStakesPlasticPCAed64_FT_emb --modify false --daily false --cluster _number 7 --cluster_type agglomerative > d/PCAed64_Clusters/PCAed64_FT_7C

#TODO Do the same for the fine_tuned ones (Done)
#TODO make sure about the nameing (Done)

## if you want to see sub samples of the cluster use my get_random.... script
