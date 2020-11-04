import sys
import numpy as np
import csv
import ast
import datetime
import math

alphas = [0.99] #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #[0.99] #[ 0.8, 0.85, 0.9, 0.95]
betas = [0.7]#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
window_sizes = [4] #[2, 3, 4, 5, 6, 7, 8, 9, 10]
threshold = [0.75]#, 0.8, 0.85, 0.9, 0.95]

def key_func(x):
    # For some year like 2011 the year is 2 digits so the date format should ne %m%d%y but for others like 2015 it should be %m%d%Y
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

def cosine_similarity(v1,v2):
    "compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)

def overlap_merge(all_sims):
    no_more_merge = False
    while no_more_merge == False:
        merged_dict = {}
        seen = []
        all_sims_keys = list(all_sims.keys())
        no_more_merge = True
        for key1 in all_sims_keys:
            if key1 not in seen:
                if key1 not in merged_dict :
                    merged_dict[key1] = list(set(all_sims[key1]))#to remove the duplicates
                for key2 in all_sims_keys:
                    if key1 != key2:
                        intersect = len(set(all_sims[key1]).intersection(set(all_sims[key2])))
                        if intersect != 0:
                            no_more_merge = False
                            merged_dict[key1].extend(list(set(all_sims[key2])))
                            merged_dict[key1] = sorted(merged_dict[key1], key = key_func)
                            seen.append(key2)
        all_sims = merged_dict
    return all_sims 
#########################################################################
def find_tail_head(all_sims, key1, key2):
    list1 = sorted(all_sims[key1], key = key_func)
    list2 = sorted(all_sims[key2], key = key_func)
    sequence1 = []
    sequence2 = []
    if len(list1) > 0 and len(list2) > 0:
        if convert_to_time(list1[0]) <  convert_to_time(list2[0]) and \
            convert_to_time(list1[-1]) >  convert_to_time(list2[0]):

            sequence1 = list1
            sequence2 = list2        
        else:
            sequence1 = list2
            sequence2 = list1        

        head = tail = []
        
        sequence1_times = [x.split("os/")[1].split()[0] for x in sequence1]
        sequence2_times = [x.split("os/")[1].split()[0] for x in sequence2]
        time_overlap = list(set(sequence1_times).intersection(set(sequence1_times)))

        tail = [x for x in sequence1 if x.split("os/")[1].split()[0] in time_overlap]
        head = [x for x in sequence2 if x.split("os/")[1].split()[0] in time_overlap]

        sequence1 = tail
        sequence2 = head

        tail_size = min(len(sequence1), len(sequence2))

        if tail_size == 1:
            tail = [sequence1[-1]]
            head = [sequence2[0]]
        else:
            tail = sequence1[-tail_size:]
            head = sequence2[:tail_size]
    return head, tail, tail_size 
    

def neighbore_cluster_merge(cluster_ave, all_sims, threshold):
    no_more_merge = False
    while no_more_merge == False:
        no_more_merge = True
        keys = list(cluster_ave.keys())
        ordered_keys = sorted(cluster_ave, key = key_func)
        seen = []

        for index, key in enumerate(ordered_keys):
            limit = len(ordered_keys)
            if index not in seen:
                cluster_size = len(all_sims[key])
                if cluster_size < 4:
                    #compare to few before and after
                    #pick the closest
                    similarities = []
                    count = 1
                    while index >= 1 and count < 5:
                        window_index = index - count
                        if window_index not in seen and window_index >= 0:
                            key2 = ordered_keys[window_index]
                            sim = cosine_similarity(cluster_ave[key], cluster_ave[key2])
                            similarities.append([key2, window_index, sim])
                        count += 1

                    while index < len(ordered_keys)  and count  < 10:
                        window_index = index + count
                        if window_index not in seen and window_index < len(ordered_keys):
                            key2 = ordered_keys[window_index]
                            sim = cosine_similarity(cluster_ave[key], cluster_ave[key2])
                            similarities.append([key2, window_index, sim])
                        count += 1
                    if len(similarities) > 0:
                        similarities = sorted(similarities, key=lambda x: x[2], reverse=True) 
                        #print("in: ", key)
                        #print("oon: ", key2)
                        if similarities[0][2] > threshold - 0.1:
                            no_more_merge = False
                            key2 = similarities[0][0] # the key with highest similarity
                            seen.append(index)
                            seen.append(similarities[0][1]) #the index
                            all_sims[key].extend(list(set(all_sims[key2])))
                            cluster_ave[key] = list((np.asarray(cluster_ave[key]) +
                                                    np.asarray(cluster_ave[key2]))/2) # the new ave

    return all_sims

##########################################################################
#########################################################################
def cluster_similarity(all_sims, donor2img2embeding, donor2day2img, donor, alpha, beta, w):
    #to reduce the thresehold for the second round of merging
    keys = list(all_sims.keys())
    cluster_ave = {}
    for key in keys:
        vectors = 0
        imgs = all_sims[key]
        num_imgs = len(imgs)
        for img in imgs:
            vectors += np.asarray(donor2img2embeding[donor][img])
        cluster_ave[key] = list(vectors/num_imgs) # the average of the embeddings in the cluster

    no_more_merge = False
    while no_more_merge == False:
        no_more_merge = True
        keys = list(cluster_ave.keys())
        num_keys = len(keys)
        all_dists = []
        for index1 in range(num_keys):
            for index2 in range(index1 + 1, num_keys):
                emb1 = cluster_ave[keys[index1]]
                emb2 = cluster_ave[keys[index2]]
                simi = cosine_similarity(emb1, emb2)
                row = []
                row.append(keys[index1])
                row.append(keys[index2])
                row.append(simi)
                all_dists.append(row)

        seen = set()
        for i, l in enumerate(sorted(all_dists, key=lambda x: x[2], reverse=True)): #descending
            if l[2] < beta:
                break
            if l[0] not in seen and l[1] not in seen and l[2] > (beta):
                #l[0] is key1, l[1] is key2, l[2] is the similarity
                no_more_merge = False
                all_sims[l[0]].extend(list(set(all_sims[l[1]])))
                del all_sims[l[1]]
                cluster_ave[l[0]] = list((np.asarray(cluster_ave[l[0]]) +
                                        np.asarray(cluster_ave[l[1]]))/2) # the new ave
                del cluster_ave[l[1]]
                seen.add(l[0])
                seen.add(l[1])
    all_sims = neighbore_cluster_merge(cluster_ave, all_sims, beta)
    print_(all_sims, donor, alpha, beta, w)

##########################################################################
def similarity_merge(all_sims, donor2img2embeding, donor2day2img, donor,alpha,beta, w):
    no_more_merge = False
    while no_more_merge == False:
        #merged_dict = {}
        seen = []
        all_sims_keys = list(all_sims.keys())
        no_more_merge = True

        for key1 in all_sims_keys:
            all_sims_keys = list(all_sims.keys())
            if key1 in seen:
                continue

            one2nsimi = []
            for key2 in all_sims_keys:
                if all_sims_keys.index(key2) <= all_sims_keys.index(key1):
                    continue
                head, tail, tail_size = find_tail_head(all_sims, key1, key2)
                if tail_size >= 1 :
                    similarity = []
                    for img_index in range(tail_size):
                        emb1 = donor2img2embeding[donor][tail[img_index]]
                        emb2 = donor2img2embeding[donor][head[img_index]]
                        simi = cosine_similarity(emb1, emb2)
                        similarity.append(simi)
                    sub_seq_simi = sum(similarity) / tail_size #average similarity
                    one2nsimi.append([key2,sub_seq_simi])
            if len(one2nsimi) > 0:
                one2nsimi = sorted(one2nsimi, key=lambda x: x[1], reverse=True)
                val = max(one2nsimi[0][1], beta - 0.1)
                if one2nsimi[0][1] >= val:
                    #one2nsimi.append([key2,sub_seq_simi])
                    no_more_merge = False
                    all_sims[key1].extend(list(set(all_sims[one2nsimi[0][0]])))
                    del all_sims[one2nsimi[0][0]]
                    seen.append(one2nsimi[0][0])
    print("number of the clusters after sim:")
    print(len(list(all_sims.keys())))
    cluster_similarity(all_sims, donor2img2embeding, donor2day2img, donor, alpha,beta, w)

####################################################################
def add_to_similarity_dict(all_sims, similarities, key, alpha,beta):
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    max_ = similarities[0][1]
    threshold = max(alpha * max_, beta)
    if key not in all_sims:
        all_sims[key] = [key]
    for ind, pair in enumerate(similarities):
        if pair[1] >= threshold:
            #if key not in all_sims:
            #    all_sims[key] = []
            all_sims[key].append(pair[0])
    return all_sims


##################################################################
def print_(all_sims, donor, alpha, beta, w):
    label = 0
    print(beta)
    with open(str(donor) + 'clusters_'+ str(alpha) +"_"+ str(beta) + "_" + str(w), 'a') as cluster_file:
        for key in all_sims:
            label = label + 1
            for img in all_sims[key]:
                temp = img.replace('JPG', 'icon.JPG: ')
                #print(temp + donor + "_" + str(label))
                row = temp + donor + "_" + str(label).zfill(4) + '\n'
                cluster_file.write(row)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#################################################################
def sequence_finder(donor2img2embeding, donor2day2img):

    for donor in donor2img2embeding:
        print(donor)
        #if donor == 'UT16-13D':
        days = list(donor2day2img[donor].keys())
        days.sort()
        all_embs = donor2img2embeding[donor]
        for alpha in alphas:
            for beta in betas:
                for w in window_sizes:
                    all_sims = {} #key = imgs, value = [[im1, dist],im2, dit[],...]

                    window_size = w
                    compared = []
                    windows = rolling_window(np.array(range(len(days))), window_size)
                    for window in windows:
                        for ind1 in range(len(window)):
                            for ind2 in range(ind1 + 1, len(window)):
                                pair = (window[ind1], window[ind2])
                                if pair not in compared:
                                    compared.append(pair)
                                    day1_ind = pair[0]
                                    day2_ind = pair[1]
                                    day1_imgs = donor2day2img[donor][days[day1_ind]]
                              
                                    for day1_img in day1_imgs:
                                        emb = all_embs[day1_img]
                                        key = day1_img
                                        for seen in all_sims:
                                            for x in all_sims[seen]:
                                                if day1_img ==  x: # if it is one of the matched ones
                                                    key = seen
                                            
                                        day2_imgs = donor2day2img[donor][days[day2_ind]]
                                        similarities = []
                                        for day2_img in day2_imgs:
                                            emb2 = all_embs[day2_img] 
                                            sim = cosine_similarity(emb, emb2)
                                            similarities.append([day2_img, sim])
                                        all_sims = add_to_similarity_dict(all_sims, similarities, key, alpha, beta)
                    all_sims2 = overlap_merge(all_sims)
                    print_(all_sims2, donor, alpha, beta, w)
                    #cluster_similarity(all_sims2, donor2img2embeding, donor2day2img, donor,alpha, beta, w)
                    #similarity_merge(all_sims2, donor2img2embeding, donor2day2img, donor, alpha beta, w)

