#python3 embeding_clustering_full.py --embeding_file data/pcaUT29-15 --modify false/true --daily false/true  --cluster_number 7 --method merge/not_merge > daily_merge_7ClusAll
# the file for this script should be image_name va1l val2.... valn.
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
from operator import itemgetter 
from functools import reduce
from scipy.spatial import distance
#from kneed import KneeLocator
## to use itemgetter(the indices seperated by ,)(the list name)


parser = argparse.ArgumentParser()
parser.add_argument('--embeding_file', type = str)
parser.add_argument('--modify', type = str) #True or False
parser.add_argument('--daily', type = str) #True or False
parser.add_argument('--cluster_number') # A number
parser.add_argument('--method') # A number

args = parser.parse_args()

embedings_file = args.embeding_file #sys.argv[1] # This should be a pca version the embedings
modify = args.modify 
Daily = args.daily 
num_clusters = int(args.cluster_number)
method = args.method

donors2imgs = {}
donors2img2embed = {}

def key_func(x):
    # For some yesr line 2011 the year is 2 digits so the date format should ne %m%d%y but for others like 2015 it should be %m%d%Y
    if '(' in x:     
        return datetime.datetime.strptime(x.split('D_')[-1].split('(')[0].strip().replace('_',''), '%m%d%y')
    else: 
        return datetime.datetime.strptime(x.split('D_')[-1].split('.JPG')[0].strip().replace('_',''), '%m%d%y')

def sort_dates(donors2imgs): #sorts the dates by getting a list of img_names for each donor and sorting that
    for key in donors2imgs:
        donors2imgs[key] = sorted(donors2imgs[key], key=key_func)
    return donors2imgs

def convert_to_time(img_name):
    if '(' not in img_name:
        date = img_name.split('/')[-1].split("D_")[1].split('.JPG')[0]
    else:
        date = img_name.split('/')[-1].split("D_")[1].split(' ')[0]
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
    potential_cluster_num = np.arange(2, 10, 1) # arange(start, stop, step)
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
    print("best cluster number is ", num_clusters)
    return num_clusters

def cluster(donor2img2embeding, donor2day2img):
    img_names = []
    vectors = []
    for donor in donor2img2embeding:
        for img in donor2img2embeding[donor]:
            img_names.append(img.replace('JPG','icon.JPG').replace(' ',' '))
            vectors.append(donor2img2embeding[donor][img])
    vectors = np.array(vectors)
    #num_clusters = 15 #find_cluster_num(vectors)
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(vectors)
    labels = kmeans.predict(vectors)
    score = silhouette_score(vectors, labels)
    print(score)

    for index, label in enumerate(labels):
        print(img_names[index] , ":" ,  label)

def daily_clustering(donor2img2embeding, donor2day2img):
    img_names = []
    vectors = []
    days_data = {}
    for donor in donor2day2img:
        for day in donor2day2img[donor]:
            if day not in days_data:
                days_data[day] = [0, 0]
                days_data[day][0] = [] # this would contain the image_names for this day
                days_data[day][1] = [] # this would contain the embedings for this day
            days_data[day][0].extend(donor2day2img[donor][day]) # used extend beacuse donor2day2img[donor][day] is a list
            for img in donor2day2img[donor][day]:
                days_data[day][1].append(donor2img2embeding[donor][img])

    for day in days_data:
        vectors = days_data[day][1]
        vectors = np.array(vectors)
        #num_clusters = 15 #find_cluster_num(vectors)

        if len(days_data[day][0]) < num_clusters:
            continue

        kmeans = KMeans(n_clusters = num_clusters)
        kmeans.fit(vectors)
        labels = kmeans.predict(vectors)
        score = silhouette_score(vectors, labels)
        print(score)

        for index, label in enumerate(labels):
            print(days_data[day][0][index] , ":" , day, "_", label)

# This function will put all images for each day across donor and then cluster per day and then merge each day to the next
def daily_clustering_per_multidonor(donor2img2embeding, donor2day2img):

    day2clus2emb = {}#each cluster will have one vector which is the center/average of the ones belonging to it
    day2clus2imgs = {} # each claster (key) would have a value of the lsit of image names belonging to it

    img_names = []
    vectors = []
    days_data = {}
    for donor in donor2day2img:
        for day in donor2day2img[donor]:
            if day not in days_data:
                days_data[day] = [0, 0]
                days_data[day][0] = [] # this would contain the image_names for this day
                days_data[day][1] = [] # this would contain the embedings for this day
            days_data[day][0].extend(donor2day2img[donor][day]) # used extend beacuse donor2day2img[donor][day] is a list
            for img in donor2day2img[donor][day]:
                days_data[day][1].append(donor2img2embeding[donor][img])

    days = []
    for day in days_data:
        vectors = days_data[day][1]
        img_names = days_data[day][0]
        vectors = np.array(vectors)

        if len(vectors) < num_clusters :
            continue
        else:
            days.append(day)
            if day not in day2clus2emb:
                day2clus2emb[day] = {}
                day2clus2imgs[day] = {}

            kmeans = KMeans(n_clusters = num_clusters)
            kmeans.fit(vectors)
            labels = kmeans.predict(vectors)
            
            #score = silhouette_score(vectors, labels)
            #print(score)
            cluster_ids = np.array(labels)
            clus2embs = {}
            clus2imgs = {}
            for clus in range(num_clusters):
                clus_index = np.where(cluster_ids == clus)[0] #get the indices for the cluster id = clus

                clus2embs[clus] = kmeans.cluster_centers_[clus]
                day2clus2emb[day] = clus2embs
                temp = itemgetter(*clus_index)(img_names)
                clus2imgs[clus] = list(temp) if type(temp) is tuple else [temp]
                day2clus2imgs[day] = clus2imgs

            for index, label in enumerate(labels):
                print(img_names[index] , ":" ,donor, "_", day, "_", label)
    find_associations(days, day2clus2imgs, day2clus2emb)
    return day2clus2emb

############################################################## 
# This clusters images of one day per donor and then merges the
##############################################################

def daily_clustering_per_donor(donor2img2embeding, donor2day2img):
    day2clus2emb = {}#each cluster will have one vector which is the center/average of the ones belonging to it
    day2clus2imgs = {} # each claster (key) would have a value of the lsit of image names belonging to it
    days = []
    for donor in donor2day2img:
        for day in donor2day2img[donor]:
            vectors = []
            img_names = []
            imgs = donor2day2img[donor][day]
            for img in imgs:
                vectors.append(donor2img2embeding[donor][img]) 
                img_names.append(img.replace('JPG','icon.JPG'))

            vectors = np.array(vectors)

            if len(vectors) < num_clusters :
                continue
            else:
                days.append(day)
                if day not in day2clus2emb:
                    day2clus2emb[day] = {}
                    day2clus2imgs[day] = {}
                ''' 
                sum_of_squared_distances = []
                K = range(2,15)
                for k in K:
                    km = KMeans(n_clusters=k)
                    km = km.fit(vectors)
                    sum_of_squared_distances.append(km.inertia_)
                print(sum_of_squared_distances)
                x = range(1, len(sum_of_squared_distances)+1)
                kn = KneeLocator(x, sum_of_squared_distances, curve='convex', direction='decreasing')
                print(kn.knee)
                '''

                kmeans = KMeans(n_clusters = num_clusters)
                kmeans.fit(vectors)
                labels = kmeans.predict(vectors)
                
                #score = silhouette_score(vectors, labels)
                #print(score)
                cluster_ids = np.array(labels)
                clus2embs = {}
                clus2imgs = {}
                for clus in range(num_clusters):
                    clus_index = np.where(cluster_ids == clus)[0] #get the indices for the cluster id = clus
                    '''
                    clus_embs = itemgetter(*clus_index)(list(vectors)) #find all embedings for the cluster. it's tuple
                    clus_embs_sum = reduce(lambda a, b: a+b, clus_embs) #get the average for the embedings for cluster = clus
                    clus_embs_ave = clus_embs_sum / len(clus_embs)
                    clus2embs[clus] = clus_embs_ave
                    '''
                    clus2embs[clus] = kmeans.cluster_centers_[clus]
                    day2clus2emb[day] = clus2embs
                    temp = itemgetter(*clus_index)(imgs)
                    clus2imgs[clus] = list(temp) if type(temp) is tuple else [temp]
                    day2clus2imgs[day] = clus2imgs

                for index, label in enumerate(labels):
                    print(img_names[index] , ":" ,donor, "_", day, "_", label)
        find_associations(days, day2clus2imgs, day2clus2emb)
    return day2clus2emb

def find_associations(days, day2clus2imgs, day2clus2emb):
    days.sort()
    num_days = len(days)
    merges = []
    day = days[0]


    for index in range(num_days - 1):
        day = days[index]
        next_day = days[index + 1]
        #if they don't have image for the next 40 days it must be only bones
        if next_day - day > 40:
            break
        all_dists = []
        for clus1 in range(num_clusters):
            for clus2 in range(num_clusters):
                row = [] # row = clus1, clus2, distance
                try:
                    dist = distance.euclidean(day2clus2emb[day][clus1], day2clus2emb[next_day][clus2])
                except KeyError: 
                    print('KeyError')
                    import bpython
                    bpython.embed(locals())
                    exit()
                row.append(clus1)
                row.append(clus2)
                row.append(dist) 
                all_dists.append(row)
            #print("dists: ", dists) 
            #similar_clus = dists.index(min(dists)) 
            #print("similar: ", similar_clus)
            #clusters[clus1].extend(day2clus2imgs[next_day][similar_clus])
        '''
        for key in clusters:
            for img in clusters[key]:
                print(img, ": merged_", key)
        '''
        seen1 = set()
        seen2 = set()
        merged = []
        print(sorted(all_dists, key=lambda x: x[2], reverse=False))
        for i, l in enumerate(sorted(all_dists, key=lambda x: x[2], reverse=False)):
            if l[0] not in seen1 and l[1] not in seen2: 

                merged.append([l[0], l[1]])
                seen1.add(l[0])
                seen2.add(l[1])
        merges.append(merged)

    def findNext(pair, day_index):
        if day_index >= len(merges):
            return []
        else: 
            for group in merges[day_index]:
                if group[0] == pair[1]: 
                    # found the associated pair
                    return [group[1]] + findNext(group, day_index + 1)

    groups  = []
    for i, pair in enumerate(merges[0]):
        day = 0
        temp = pair + findNext(pair, 1)
        groups.append(temp)

    clusters = []
    print(groups)
    for group in groups:
        images = []
        for day, cluster_index in enumerate(group): 
            day = days[day]
            images += day2clus2imgs[day][cluster_index]
        clusters.append(sorted(images, key = key_func))
    for i, imgs in enumerate(clusters):
        for img in imgs:
            print(img.replace('.JPG', '.icon.JPG'), ": merged_", i) 
    '''
    for merging_clusters in merged: #merging clusters are akways 2. The id of the cluster is always determined by the first day
        if merging_clusters[0] not in clusters:
            clusters[merging_clusters[0]] = day2clus2imgs[day][merging_clusters[0]] 
    '''      

def cal_feature_extention(img, donors_id, day_number, donor, max_day):
    extention = []
    #commented out the followings to get rid of the info about which donor = one hot encoding
    #extention = np.zeros(len(donors_id))
    #index = donors_id.index(donor)
    #extention[index] = 1
    #extention = list(extention)
    extention.append(day_number / max_day)
    if '(' not in img:
        img_num = 0
    else:
        img_num = img.split('(')[1].split(')')[0]
    extention.append(int(img_num))
    return extention
     
    
def modify_features(donors2img2embed, donor2day2imgs): 
    #This function is going to append a one hot encoding, 
    #nth picture of the day, 
    #nth day of de come to the surrent feature vector for each image
    for donor in donor2day2imgs:
        days = [key for key in donor2day2imgs[donor]]
    days.sort()
    max_day = days[-1]
    donors_id = [key for key in donor2day2imgs] 
    for donor in donor2day2imgs:
        daily_imgs = donor2day2imgs[donor] # this is a dict with this format: {day1: [images], day2: [images]...}
        for day in daily_imgs: #iterate for all of the days
            for img in daily_imgs[day]: #iterate through images for each day
                extention = cal_feature_extention(img, donors_id, day, donor, max_day) 
                donors2img2embed[donor][img].extend(extention)
    return donors2img2embed
    

with open(embedings_file, 'r') as csv_file:
    data = csv.reader(csv_file,delimiter = '\n')
    vectors = []
    for row in data:
        row = row[0]
        row= row.split('JPG')
        img_name = row[0] + 'JPG'
        embeding = row[1].strip()
        embeding = embeding.replace(' ', ',')
        embeding = ast.literal_eval("[" + embeding[1:-1] + "]") # this embeding is a list now
        donor_id = img_name.split('/Daily')[0].split('/')[-1]
        if donor_id not in donors2img2embed and donor_id not in donors2imgs:
            donors2img2embed[donor_id] = {} 
            donors2imgs[donor_id] = [] # a list for all of the images belonging to the same donor
        donors2img2embed[donor_id][img_name] = embeding 
        # this a dictionary with each donor_id as keys and values are another dictionary
        # with keys being an image and the values being the feature vector for that imag

        donors2imgs[donor_id].append(img_name)

    donors2imgs_sorted = sort_dates(donors2imgs) # this sorts the images for a donor based on their dates
    donor2day2imgs = cal_day_from_deth(donors2imgs_sorted)

    if method == 'merge':
        if modify == 'true':
            donors2img2embed = modify_features(donors2img2embed, donor2day2imgs)
        #day2clus2emb = daily_clustering_per_multidonor(donors2img2embed, donor2day2imgs)
        day2clus2emb = daily_clustering_per_donor(donors2img2embed, donor2day2imgs)


       
    else:
        if modify == 'true':
            donors2img2embed = modify_features(donors2img2embed, donor2day2imgs)
        
        if Daily == 'true':
            daily_clustering(donors2img2embed, donor2day2imgs)
        elif Daily == 'false':
            cluster(donors2img2embed, donor2day2imgs)
