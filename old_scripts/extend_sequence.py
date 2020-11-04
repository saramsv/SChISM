import pandas as pd
import sys
import csv
import datetime
import numpy as np
import ast
import math

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

def get_embedding(row):
    row= row.split('JPG')
    img_name = row[0] + 'JPG'
    embedding = row[1].strip()
    embedding = embedding.replace(' ', ',')
    embedding = ast.literal_eval("[" + embedding[1:-1] + "]")
    return embedding

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

def print_(clusters_imgnames, name):
    label = 0
    with open("extended_seq_th25" + name, 'a') as cluster_file:
        for key in clusters_imgnames:
            label = label + 1
            for img in clusters_imgnames[key]:
                row = img + ":" + str(label).zfill(3) + '\n'
                cluster_file.write(row)

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def neighbore_cluster_merge(clusters_imgnames, clusters_dates, days, clusters_embs, matrix):
    no_more_merge = False
    keys = list(clusters_dates.keys())
    while no_more_merge == False :
        no_more_merge = True
        seen = []

        keys = list(clusters_dates.keys())
        print(len(keys))

        cluster_ave = {}
        for key in keys:
            vectors = 0
            for emb in clusters_embs[key]:
                vectors += np.asarray(emb)
            cluster_ave[key] = list(vectors/len(clusters_embs[key])) 
            # the average of the embeddings in the cluster

        #merge with the goal of filling out the empty indexes of the seq
        for index, cluster_key in enumerate(keys):
            empty = np.where(matrix[index] == 0)
            similarities = []
            for index2, cluster_key2 in enumerate(keys):
                if index2 > index :
                    full = np.where(matrix[index2] != 0)
                    flag = 0
                    if (any(x in list(full[0]) for x in list(empty[0]))):
                    #if len(np.intersect1d(full, empty)) > 1:
                        flag = 1
                    if index not in seen and index2 not in seen and flag == 1 and len(empty) > 0:
                        sim = cosine_similarity(cluster_ave[cluster_key], 
                                cluster_ave[cluster_key2])
                        similarities.append([cluster_key2, index2, sim])

            if len(similarities) > 0 and similarities[0][2] > 0.25:
                similarities = sorted(similarities, key=lambda x: x[2], reverse=True) 
                no_more_merge = False
                key2 = similarities[0][0] # the key with highest similarity
                ind = similarities[0][1] # the key with highest similarity
                seen.append(index)
                seen.append(ind)
                clusters_imgnames[cluster_key].extend(clusters_imgnames[key2])
                del clusters_imgnames[key2]
                clusters_dates[cluster_key].extend(clusters_dates[key2])
                del clusters_dates[key2]
                clusters_embs[cluster_key].extend(clusters_embs[key2])
                del clusters_embs[key2]
                cluster_ave[cluster_key] = list((np.asarray(cluster_ave[cluster_key]) +
                                        np.asarray(cluster_ave[key2]))/2) # the new ave
                del cluster_ave[key2]
                matrix[index, list(full[0])] = 1

    return clusters_imgnames

####################################################################################################
names = ['UT57', 'UT45', 'UT16']# ,'UT100', 'UT102', 'UT106', 'UT108', 'UT109', 'UT110', 'UT111']

for name in names:
    #cluster_filename = "ut1all_2011/PCAedClusters/" + name + "/" + name + "-11Dclusters_0.7"  #"ut1all_2011/final_clusters_ut1all_2011/" + name
    cluster_filename =  name + "-13Dclusters_0.75"  #"ut1all_2011/final_clusters_ut1all_2011/" + name
    print(cluster_filename)
    #cluster_filename = "ut1all_2011/basic_clustering/" + name
    emb_filename = "/data/sara/ImageSimilarityMultiMethods/3donors/ut45_16_57/" + name + "_incep_PCAed"

    cluster_df = pd.read_csv(cluster_filename, sep=":", names=['path', 'label'])
    emb_df = pd.read_csv(emb_filename, sep="G", names=['path', 'emb'])

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
            path = row['path'].strip()
            imgs.append(path)
            all_imgs.append(path)
            line = emb_df.loc[emb_df['path'] + 'G' == path.replace('.icon','')]
            try:
                line = line.path.values[0]+ 'G' + line.emb.values[0]
            except:
                import bpython
                bpython.embed(locals())
                exit()
            emb = get_embedding(line)
            if l not in clusters_dates:
                clusters_dates[l] = []
            clusters_dates[l].append(convert_to_time(path))
            if l not in clusters_embs:
                clusters_embs[l] = []
            clusters_embs[l].append(emb)

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

    clusters_imgnames = neighbore_cluster_merge(clusters_imgnames, clusters_dates, days, clusters_embs, matrix)
    print_(clusters_imgnames, name)































