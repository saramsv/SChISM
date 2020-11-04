import pandas as pd
import sys
import csv
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt


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

def count_essential_sequences(matrix, decending_cluster_size_index):
    matrix = matrix[decending_cluster_size_index, :]
    for index, row in enumerate(matrix):
        for index2, val in enumerate(row):
            if matrix[index][index2] > 0: #if val > 0:
                for next_rows in range(index+1, matrix.shape[0]):
                    matrix[next_rows][index2] = 0

    essential_seq_num = 0
    for row in matrix:
        if len(np.where(row != 0)[0]) > 5:
            essential_seq_num += 1
    
    print(essential_seq_num)

def plot_(matrix, f, name):
    ax = sns.heatmap(matrix_sorted)#, vmin = 0.0, vmax = 1.0)
    ax.set_title(f.split('/')[-1] + name)
    plt.xlabel("Day")
    plt.ylabel("Cluster number")
    plt.savefig(f.split('/')[-1] + name)
    plt.clf()

def get_seq_stat(row, days, name, f, cluster_key): #calculate the stats on gap lengs and seq length
    '''
    ## gap is the number of days with no image in the cluster
    gap = len(matrix[index]) - np.count_nonzero(matrix[index])
    ## thickness is the number of images for each day in each cluster
    thickness = sum(matrix[index])//np.count_nonzero(matrix[index]) + 1
    '''

    gap_len_list = []
    seq_len_list = []
    gap = 0
    seq = 0
    if len(np.where(row != 0)[0]) > 5:
        for index, val in enumerate(row):
            if val == 0:
                gap +=1 
                if seq != 0:
                    seq_len_list.append(seq)
                    seq = 0
            else:
                seq +=1
                if gap != 0:
                    gap_len_list.append(gap)
                    gap = 0
        if gap > 0:
            gap_len_list.append(gap)
        if seq > 0:
            seq_len_list.append(seq)
        if len(seq_len_list) > 0:
            longest_seq = max(seq_len_list)
        else:
            longest_seq = 0
        if len(gap_len_list) > 0:
            longest_gap = max(gap_len_list)
        else:
            longest_gap = 0
        gaps_more_than = len(np.where(np.array(gap_len_list) > 5)[0])
        gap_ratio = len(np.where(row ==0)[0])/days
        print(name + f,',',longest_seq,',',longest_gap,',', gaps_more_than,',' ,gap_ratio)
            



metric_final = 0
names = ['data_for_metric_calculation/basicResnet/corrected_merged_basicResnet','data_for_metric_calculation/mymethod/corrected_merge_large']
print('name,longest_seq,longest_gap,gaps_more_than,gap_ratio')
for name in names:
    files = ['UT100', 'UT102', 'UT106', 'UT108', 'UT109', 'UT110', 'UT111']
    for f in files:
        donor_metric = 0
        #cluster_filename = "ut1all_2011/final_clusters_ut1all_2011/" + name
        #cluster_filename = "ut1all_2011/basic_clustering/" + name
        cluster_filename =  name + f #"ut1all_2011/GTs/" + name + '_gt'

        cluster_df = pd.read_csv(cluster_filename, sep=":", names=['path', 'label'])

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


            imgs = sorted(imgs, key = key_func)
            clusters_imgnames[l] = imgs

        all_imgs = sorted(all_imgs, key = key_func)
        for index, img in enumerate(all_imgs):
            date = convert_to_time(img)
            if date not in days:
                day_id = len(days)
                days[date] = day_id

        matrix = np.zeros((len(clusters_dates), len(days))) 
        keys = list(clusters_dates.keys())
        for index, cluster_key in enumerate(keys):
            for index2, date in enumerate(clusters_dates[cluster_key]):
                matrix[index][days[date]] += 1
                #print(clusters_imgnames[cluster_key][index2]+ ":" + cluster_key + ":" + str(days[date]).zfill(3))

            get_seq_stat(matrix[index], len(days), name, f, cluster_key)
        
        '''
        ### sort the matrix
        cluster_starts = []
        cluster_counts = []
        for row in matrix:
           cluster_starts.append(np.where(row != 0)[0][0]) 
           cluster_counts.append(len(np.where(row != 0)[0]))

        sorted_index = np.argsort(np.array(cluster_starts))
        cluster_counts = np.array(cluster_counts)
        decending_cluster_size_index =cluster_counts.argsort()[::-1][:len(cluster_counts)]
        
        matrix_sorted = matrix[sorted_index, :]
        count_essential_sequences(matrix_sorted, decending_cluster_size_index)
        #plot_(matrix_sorted, f, name)
        #np.savetxt('test.txt', matrix_sorted, delimiter=',')
        #matrix_sorted = matrix_sorted/18.0 #np.amax(matrix_sorted)
        #metric = 0
        #for l in clusters_dates:
        #    metric += len(set(clusters_dates[l]))
        #donor_metric = metric /len(clusters_dates)
        #metric_final += donor_metric
        '''
#print(donor_metric/7) 

        








