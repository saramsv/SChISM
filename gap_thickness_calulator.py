import pandas as pd
import sys
import csv
import datetime
import numpy as np
#import seaborn as sns
#import matplotlib.pylab as plt


def key_func(x):
    # For some year like 2011 the year is 2 digits so the date format
    #should ne %m%d%y but for others like 2015 it should be %m%d%Y
    try:
        #date = ""
        if '(' in x:     
            date_ = x.split('D_')[-1].split('(')[0].strip()
        else:
            date_ = x.split('D_')[-1].split('.')[0].strip()
        mdy = date_.split('_')
        m = mdy[0]
        d = mdy[1]
        y = mdy[2]
        if len(m) == 1:
            m = '0' + m
        if len(d) == 1:
            d = '0' + d
        date_  = m + d + y
        if len(date_) == 6: #the format that has 2 digits for year
            return datetime.datetime.strptime(date_, '%m%d%y')
        else:
            return datetime.datetime.strptime(date_, '%m%d%Y')
    except:
        print(x)
        import bpython
        bpython.embed(locals())
        exit()

def convert_to_time(img_name):
    if '(' in img_name:     
        date_ = img_name.split('D_')[-1].split('(')[0].strip()
    else:
        date_ = img_name.split('D_')[-1].split('.')[0].strip()
    mdy = date_.split('_')
    m = mdy[0]
    d = mdy[1]
    y = mdy[2]
    if len(m) == 1:
        m = '0' + m
    if len(d) == 1:
        d = '0' + d
    date_  = m + d + y
    if len(date_) == 6: #the format that has 2 digits for year
        return datetime.datetime.strptime(date_, '%m%d%y')
    else:
        return datetime.datetime.strptime(date_, '%m%d%Y')

metric_final = 0
names = ['UT100', 'UT102', 'UT106', 'UT108', 'UT109', 'UT110', 'UT111']

for name in names:
    donor_metric = 0
    cluster_filename = "ut1all_2011/final_clusters_ut1all_2011/" + name
    #cluster_filename = "ut1all_2011/basic_clustering/" + name
    #emb_filename = "ut1all_2011/PCAed/" + name + "_incept_PCAed"

    cluster_df = pd.read_csv(cluster_filename, sep=":", names=['path', 'label'])
    #emb_df = pd.read_csv(emb_filename, sep="G", names=['path', 'emb'])

    labels = cluster_df.label.unique()
    all_imgs = []

    clusters_imgnames = {}
    clusters_dates = {}
    days = {}
    clusters_embs = {}

    for l in labels:
        rows = cluster_df.loc[cluster_df['label'] == l]
        imgs = []
        for index, row in rows.iterrows():
            path = row['path']
            imgs.append(path)
            all_imgs.append(path)
            #line = emb_df.loc[emb_df['path'] + 'G' == row['path'].replace('.icon','')]
            if l not in clusters_dates:
                clusters_dates[l] = []
            clusters_dates[l].append(convert_to_time(path))


        #imgs = sorted(imgs, key = key_func)
        clusters_imgnames[l] = imgs

    all_imgs = sorted(all_imgs, key = key_func)
    for index, img in enumerate(all_imgs):
        date = convert_to_time(img)
        if date not in days:
            day_id = len(days)
            days[date] = day_id

    matrix = np.zeros((len(clusters_dates), len(days))) 
    keys = list(clusters_dates.keys())
    cluster_starts = []
    for index, cluster_key in enumerate(keys):
        for index2, date in enumerate(clusters_dates[cluster_key]):
            matrix[index][days[date]] += 1
            print(clusters_imgnames[cluster_key][index2]+ ":" + cluster_key + ":" + str(days[date]))

        ## gap is the number of days with no image in the cluster
        gap = len(matrix[index]) - np.count_nonzero(matrix[index])
        ## thickness is the number of images for each day in each cluster
        thickness = sum(matrix[index])//np.count_nonzero(matrix[index]) + 1
        #print("cluster {} has gap of {} and thikness of {}".format(cluster_key, gap, thickness))
        #print("{}, {}".format( gap, thickness))

    ### sort the matrix
    for row in matrix:
       cluster_starts.append(np.where(row != 0)[0][0]) 

    sorted_index = np.argsort(np.array(cluster_starts))
    matrix_sorted = matrix[sorted_index, :]
    #np.savetxt('seq.txt', matrix_sorted, delimiter=',')
    matrix_sorted = matrix_sorted/18.0 #np.amax(matrix_sorted)
    '''
    ax = sns.heatmap(matrix_sorted, vmin = 0.0, vmax = 1.0)
    ax.set_title(name + "seq")
    plt.xlabel("Day")
    plt.ylabel("Cluster number")
    #plt.show()   
    plt.savefig(name + "seq")
    plt.clf()
    '''
    metric = 0
    for l in clusters_dates:
        metric += len(set(clusters_dates[l]))
    donor_metric = metric /len(clusters_dates)
    metric_final += donor_metric
    break


#print(donor_metric/7) 









