from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import ast
import datetime


embedings_file = sys.argv[1]

num_clusters = 3
colors = [np.random.rand(3,) for i in range(num_clusters)]
donors2imgs = {}
donors2embed = {}

def sort_dates(donors2imgs):
    print(donors2imgs)
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

def cal_day_from_deth(donors2imgs):
    for key in donors2imgs:
        img2day = {}
        first_img = True
        for img in donors2imgs[key]:
            if first_img == True:
                start_time = convert_to_time(img)
                first_img = False
            img_time = convert_to_time(img)
            time_from_start = (img_time - start_time).days
            img2day[img] = time_from_start
        donors2imgs[key] = img2day 
    print(donors2imgs)
    return donors2imgs

                

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
        if donor_id not in donors2embed and donor_id not in donors2imgs:
            print(donor_id)
            donors2embed[donor_id] = {} 
            donors2imgs[donor_id] = []
        donors2embed[donor_id][img_name] = embeding
        donors2imgs[donor_id].append(img_name)

        #vectors.append(embeding)
    #vectors = np.array(vectors)
    donors2imgs2 = sort_dates(donors2imgs)
    dates = cal_day_from_deth(donors2imgs2)
    import bpython
    bpython.embed(locals())
    exit()


    ## cluster 
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans.fit(vectors)
    labels = kmeans.predict(vectors)
    
    ## Reduce dimention to visualize
    model = TSNE(n_components=2, perplexity=40)
    results = model.fit_transform(vectors)
    
    plt.figure(figsize=(8,5))
    for row_number in range(0, results.shape[0]):
        plt.scatter(results[row_number,0]*100, results[row_number,1]*100, c = colors[labels[row_number]])
    plt.show()

    
    for index, label in enumerate(labels):
        print(donors2imgs[index] , ":", label)
