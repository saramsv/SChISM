from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import sys
import numpy as np
import csv
import ast
import datetime


embedings_file = sys.argv[1]

num_clusters = 2
colors = [np.random.rand(3,) for i in range(num_clusters)]
donors2imgs = {}
donors2img2embed = {}

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

def cluster(donor2img2embeding, donor2day2img):
    for donor in donor2img2embeding:
        embedings_per_day_per_donor = [i for i in range(len(donor2day2img[donor]))] #how many days per donor
        for day in donor2day2img[donor]: #iterate through each day per donor
            embedings_per_day_per_donor[day] = donor2day2img[donor][day] # only to sort days
        for day in embedings_per_day_per_donor:#each day would include a list of images
            day_vector = []
            for img in day:
                day_vector.append(donor2img2embeding[donor][img])

            ## cluster 
            vectors = np.array(day_vector)
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
    cluster(donors2img2embed, donor2day2imgs)

    ''' 
    for index, label in enumerate(labels):
         print(donors2imgs[index] , ":", label)
    '''
