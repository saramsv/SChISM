import sys
from scipy import spatial
import numpy as np
import csv
import ast
import datetime
import math


def key_func(x):
    date_ = x.split('/')[-1]
    y = '00'
    if date_[3] == '1':
        y = '12'
    elif date_[3] == '0':
        y = '11'
    m = date_[4:6]
    d = date_[6:8]
    if d == '29' and m == '02':
        d = '28'
    date_ = m + d + y
    return datetime.datetime.strptime(date_, '%m%d%y')
#########################################################
def cosine_similarity(v1,v2):
    return 1 - spatial.distance.cosine(v1, v2)
##########################################################
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
##########################################################################
def add_to_similarity_dict(all_sims, similarities, key, count, mean_sim):#, ratio):
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    max_ = similarities[0][1]
    mean_sim = mean_sim * (count - 1) + max_
    mean_sim = mean_sim / count
    threshold = max(0.99 * max_, mean_sim)
    if key not in all_sims:
        all_sims[key] = [key]
    for ind, pair in enumerate(similarities):
        if pair[1] >= threshold:
            all_sims[key].append(pair[0])
    return all_sims, mean_sim


##################################################################
def print_(all_sims, donor, root_dir):
    label = 0
    not_sequenced = []
    print(len(all_sims))
    with open( root_dir + donor + "_pcaed_sequenced", 'w') as f_seq:
        for key in all_sims:
            if len(all_sims[key]) > 1:
                label = label + 1
                for img in all_sims[key]:
                    temp = img.replace('JPG', 'icon.JPG: ')
                    #print(temp + donor + "_" + str(label))
                    f_seq.write(temp + donor + "_" + str(label) + "\n")
            else:
                not_sequenced.append(all_sims[key])
    with open(root_dir + donor + "_not_sequenced", 'w') as f:
        for image in not_sequenced: 
            f.write(image[0] + "\n")
#################################################################
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
################################################################
def match(day1, day2, all_sims, count, mean_sim, donor2day2img, all_embs, donor):
    day1_imgs = donor2day2img[donor][day1]
    for day1_img in day1_imgs:
        emb = all_embs[day1_img]
        key = day1_img
        for seen in all_sims:
            for x in all_sims[seen]:
                if day1_img ==  x: # if it is one of the matched ones
                    key = seen
            
        day2_imgs = donor2day2img[donor][day2]
        similarities = []
        for day2_img in day2_imgs:
            emb2 = all_embs[day2_img] 

            sim = cosine_similarity(emb, emb2)
            similarities.append([day2_img, sim])
        count += 1
        #print(day1_img)
        all_sims, mean_sim = add_to_similarity_dict(all_sims, similarities, key, count, mean_sim)
    return all_sims, mean_sim, count
#################################################################
def sequence_finder(donor2img2embeding, donor2day2img, root_dir):
    for donor in donor2img2embeding:
        days = list(donor2day2img[donor].keys())
        days.sort()
        all_embs = donor2img2embeding[donor]
        all_sims = {} #key = imgs, value = [[im1, dist],im2, dit[],...]
        window_size = 3
        compared = []
        mean_sim = 0
        count = 0
        windows = rolling_window(np.array(range(len(days))), window_size)
        for window in windows:
            for ind1 in range(len(window)):
                for ind2 in range(ind1 + 1, len(window)):
                    pair = (window[ind1], window[ind2])
                    if pair not in compared:
                        compared.append(pair)
                        day1_ind = pair[0]
                        day2_ind = pair[1]
                        day1 = days[day1_ind]
                        day2 = days[day2_ind]
                        #import bpython
                        #bpython.embed(locals())
                        all_sims, mean_sim, count = match(day1, day2, all_sims, count, mean_sim, donor2day2img, all_embs, donor)
                        #print(all_sims)
                        #_ = input()

        all_sims, mean_sim, count = match(day2, day1, all_sims, count, 
                mean_sim, donor2day2img, all_embs, donor)
        all_sims = overlap_merge(all_sims)
        print_(all_sims, donor, root_dir)
        #cluster_similarity(all_sims2, donor2img2embeding, donor2day2img, donor,alpha, beta, w)
        #similarity_merge(all_sims2, donor2img2embeding, donor2day2img, donor, alpha beta, w)

