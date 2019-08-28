from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import ast
import datetime
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--embeding_files', type = str)

args = parser.parse_args()

embedings_file = args.embeding_files #sys.argv[1] # This should be a pca version the embedings

donors2imgs = {}
donors2img2embed = {}

def sort_dates(donors2imgs): #sorts the dates by getting a list of img_names for each donor and sorting that
    for key in donors2imgs:
        dates = []
        for img in donors2imgs[key]:
            dates.append(img)
        dates.sort()
        donors2imgs[key] = dates
    return donors2imgs

def convert_to_time(img_name):
    date = img_name.split('/')[-1].split("D_")[1].split('@')[0]
    # this is the format and we only need the date part UT06-12D_07_26_12 (21).JPG
    date = date.replace('_', '') # to remove the '_'
    return datetime.datetime.strptime(date, '%m%d%y') #formated as date

def cal_day_from_deth(donors2imgs_sorted):
    for key in donors2imgs_sorted:
        day2imgs = {} 
        first_img = True
        for img in donors2imgs_sorted[key]:
            if first_img == True:
                start_time = convert_to_time(img)
                first_img = False
            img_time = convert_to_time(img)
            time_from_start = (img_time - start_time).days
            if time_from_start not in day2imgs:
                day2imgs[time_from_start] = []
            day2imgs[time_from_start].append(img)
        donors2imgs_sorted[key] = day2imgs 
    return donors2imgs_sorted 
    # this a dictionary with each donor_id as keys and values are another 
    #dictionary with keys being xth days since day one and the values are a 
    #list of images that belong to day xth for that donor.
def save_to_pickle(dict_, file_name):
    f = open(file_name + '.pkl', 'wb')
    pickle.dump(dict_, f)
    #f.colse()

def cal_cluster2img(labels, names):
        cluster2img = {}
        for index, label in enumerate(labels):
            if label not in cluster2img:
                cluster2img[label] = []
            cluster2img[label].append(names[index])
        return cluster2img

def find_cluster_num(vectors):
    potential_cluster_num = np.arange(2, 5, 1) # arange(start, stop, step)
    silhouette_scores = []
    for c in potential_cluster_num:
        ## kmean cluster 
        kmeans = KMeans(n_clusters = c)
        kmeans.fit(vectors)
        labels = kmeans.predict(vectors)
        score = silhouette_score(vectors, labels)
        silhouette_scores.append(score)
        print("[INFO] number of clusters is {} and the silhouette score is {}".format(c, score))
    num_clusters = silhouette_scores.index(max(silhouette_scores)) + 2 #because I started with 2 for potential_cluster_num
    import bpython
    bpython.embed(locals())
    print("best cluster number is ", num_clusters)
    return num_clusters

def cluster(donor2img2embeding, donor2day2img):
    img_names = []
    vectors = []
    for donor in donor2img2embeding:
        for img in donor2img2embeding[donor]:
            img_names.append(img.replace('JPG','icon.JPG').replace('@',' '))
            vectors.append(donor2img2embeding[donor][img])
    vectors = np.array(vectors)
    num_clusters = find_cluster_num(vectors)
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(vectors)
    labels = kmeans.predict(vectors)

    #for index, label in enumerate(labels):
    #    print(img_names[index] , ":" ,  label)


def cal_feature_extention(img, donors_id, day_number, donor):
    extention = np.zeros(len(donors_id))
    index = donors_id.index(donor)
    extention[index] = 1
    extention = list(extention)
    extention.append(day_number)
    img_num = img.split('(')[1].split(')')[0]
    extention.append(int(img_num))
    return extention
     
    
def modify_features(donors2img2embed, donor2day2imgs): 
    #This function is going to append a one hot encoding, 
    #nth picture of the day, 
    #nth day of de come to the surrent feature vector for each image
    donors_id = [key for key in donor2day2imgs] 
    for donor in donor2day2imgs:
        daily_imgs = donor2day2imgs[donor] # this is a dict with this format: {day1: [images], day2: [images]...}
        for day in daily_imgs: #iterate for all of the days
            for img in daily_imgs[day]: #iterate through images for each day
                extention = cal_feature_extention(img, donors_id, day, donor) 
                donors2img2embed[donor][img].extend(extention)
    return donors2img2embed
    

with open(embedings_file, 'r') as csv_file:
    data = csv.reader(csv_file,delimiter = '\n')
    vectors = []
    for row in data:
        row = row[0]
        row= row.split('JPG')
        img_name = row[0] + 'JPG'
        embeding = row[1]
        embeding = embeding.replace(' ', ',')
        embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
        donor_id = img_name.split('/Daily')[0].split('/')[-1]
        if donor_id not in donors2img2embed and donor_id not in donors2imgs:
            donors2img2embed[donor_id] = {} 
            donors2imgs[donor_id] = []
        donors2img2embed[donor_id][img_name] = embeding 
        # this a dictionary with each donor_id as keys and values are another dictionary
        # with keys being an image and the values being the feature vector for that imag

        donors2imgs[donor_id].append(img_name)

    donors2imgs_sorted = sort_dates(donors2imgs)
    donor2day2imgs = cal_day_from_deth(donors2imgs_sorted)
    donors2img2embed = modify_features(donors2img2embed, donor2day2imgs)
    #save_to_pickle(donors2img2embed, 'donors2img2embedPCAed.pkl')
    #save_to_pickle(donor2day2imgs, 'donor2day2imgs.pkl')

    cluster(donors2img2embed, donor2day2imgs)

    ''' 
    for index, label in enumerate(labels):
         print(donors2imgs[index] , ":", label)
    '''
