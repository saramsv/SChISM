from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import ast
import datetime

embedings_file = sys.argv[1]
cluster_on = sys.argv[2]

with open(embedings_file, 'r') as csv_file:
    data = csv.reader(csv_file,delimiter = '\n')
    img_names = []
    vectors = []
    for row in data:
        row = row[0]
        row= row.split('JPG')
        img_name = row[0] + 'JPG'
        embeding = row[1]
        embeding = embeding.replace(' ', ',')
        embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
        vectors.append(embeding)
        img_names.append(img_name)

    vectors = np.array(vectors)
    ## kmean cluster 
    num_clusters = 9
    ## Reduce dimention to visualize
    #model = TSNE(n_components=2, perplexity=40)
    if cluster_on == 'pca'
        model = PCA(n_components=2)
        results = model.fit_transform(vectors)
        kmeans = KMeans(n_clusters = num_clusters)
        kmeans.fit(results)
        labels = kmeans.predict(results)

    elif cluster_on == 'features':
        kmeans = KMeans(n_clusters = num_clusters)
        kmeans.fit(vectors)
        labels = kmeans.predict(vectors)

    for index, label in enumerate(labels):
        print(img_names[index] , ":",label)
