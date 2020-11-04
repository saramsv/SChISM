import pickle
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sys
import numpy as np


embedings_file = sys.argv[1]
img_names_file = sys.argv[2]


def read_pkl_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

#read in the files in dictionary format
embedings = read_pkl_file(embedings_file)
img_names = read_pkl_file(img_names_file)

for key in embedings:
    vectors = np.array(embedings[key])
    model = TSNE(n_components=2, perplexity=50)
    results = model.fit_transform(vectors)

    plt.figure(figsize=(16,10))
    plt.scatter(results[:,0], results[:,1])
    plt.show()

    '''
    kmeans = KMeans(n_clusters = 6)
    kmeans.fit(vectors)
    labels = kmeans.predict(vectors)

    for index, label in enumerate(labels):
        print(img_names[index] , ":", label)
    '''
