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
import pickle


embedings_file = sys.argv[1] # This should be a pca version the embedings

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

def cluster(donor2img2embeding, donor2day2img):
    img_names = []
    donor2day2cluster2img = {}
    for donor in donor2img2embeding:
        all_days = []
        for day in donor2day2img[donor]:
            all_days.append(day)
        all_days.sort() # this is a sorted list of day_from_frist_day
        vectors = []
        for day in all_days:
            #day_vector = []
            for img in donor2day2img[donor][day]:
                img_names.append(img.replace('JPG','icon.JPG').replace('@',' '))
                #day_vector.append(donor2img2embeding[donor][img])
                vectors.append(donor2img2embeding[donor][img])

            #vectors = np.array(day_vector)
        vectors = np.array(vectors)

        '''
        ## DBscan clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5).fit(vectors)
        labels = dbscan.labels_

        ## Agglomerative clustering
        clustering = AgglomerativeClustering(n_clusters = 7).fit(vectors)           
        labels = clustering.labels_
        '''
        ## kmean cluster 
        num_clusters = 9
        ## Reduce dimention to visualize
        #model = TSNE(n_components=2, perplexity=40)
        #model = PCA(n_components=2)
        #results = model.fit_transform(vectors)
        kmeans = KMeans(n_clusters = num_clusters)
        kmeans.fit(vectors)
        labels = kmeans.predict(vectors)


        #day_cluster = {}
        #cluster2img = cal_cluster2img(labels, img_names)
        #day_cluster[day] = cluster2img 
        #donor2day2cluster2img[donor] = day_cluster
        for index, label in enumerate(labels):
            print(img_names[index] , " : " , donor, '_', day, '_' , label)
        '''
        num_clusters = len(np.unique(labels))
        colors = [np.random.rand(3,) for i in range(num_clusters)]

        ## Reduce dimention to visualize
        model = TSNE(n_components=2, perplexity=40)
        results = model.fit_transform(vectors)

        plt.figure(figsize=(8,5))
        for row_number in range(0, results.shape[0]):
            plt.scatter(results[row_number,0]*100, results[row_number,1]*100, c = colors[labels[row_number]])
        plt.show()
        '''

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
