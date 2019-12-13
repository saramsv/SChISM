#python3 embeding_clustering_full_kmeans.py --embeding_file data/pcaUT29-15 --modify false/true --daily false/true  --cluster_number 7 --method merge/not_merge --merge_type single/multi --ADD_file imgname2ADD > daily_merge_7ClusAll
# the file for this script should be image_name va1l val2.... valn.
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import ast
import pickle
import argparse
from operator import itemgetter 
from functools import reduce
from scipy.spatial import distance
import math
import sequence
csv.field_size_limit(sys.maxsize)


def cluster(donor2img2embeding, donor2day2img):
    for donor in donor2img2embeding:
        img_names = []
        vectors = []
        for img in donor2img2embeding[donor]:
            img_names.append(img)
            vectors.append(donor2img2embeding[donor][img])
        vectors = np.array(vectors)
        vectors = vectors / vectors.max(axis=0)
        ## kmeans:
        kmeans = KMeans(n_clusters = num_clusters)
        kmeans.fit(vectors)
        labels = kmeans.predict(vectors)
        for index, label in enumerate(labels):
            print(img_names[index] , ":" , donor, "_" , label)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeding_file', type = str)
    parser.add_argument('--cluster_number') # A number

    args = parser.parse_args()

    embedings_file = args.embeding_file #sys.argv[1] # This should be a pca version the embedings
    num_clusters = int(args.cluster_number)

    donors2imgs = {}
    donors2img2embed = {}
    donor2day2imgs = {}
    imgname2add = {}

    with open(embedings_file, 'r') as csv_file:
        data = csv.reader(csv_file,delimiter = '\n')
        vectors = []
        for row in data:
            
            row = row[0]
            row= row.split(',')
            picture_num = int(row[0])
            img_name = row[1]
            embeding = row[2].strip()
            embeding = embeding.replace(' ', ',')
            embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
            if "criminal" not in donors2img2embed:
                donors2img2embed["criminal"] = {} 
            donors2img2embed["criminal"][img_name] = embeding 
            if "criminal" not in donor2day2imgs:
                donor2day2imgs["criminal"] = {}
            if picture_num not in donor2day2imgs["criminal"]:
                donor2day2imgs["criminal"][picture_num] = []
            donor2day2imgs["criminal"][picture_num].append(img_name)

        #day2clus2emb = sequence.sequence_finder(donors2img2embed, donor2day2imgs) 
        cluster(donors2img2embed, donor2day2imgs)
