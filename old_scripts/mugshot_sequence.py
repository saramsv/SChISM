import sys
import numpy as np
import csv
import ast
import datetime
import math



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
                            seen.append(key2)
        all_sims = merged_dict
    return all_sims 
#########################################################################
def find_tail_head(all_sims, key1, key2):
    list1 = all_sims[key1]
    list2 = all_sims[key2]
    sequence1 = []
    sequence2 = []
    if len(list1) > 0 and len(list2) > 0:
        sequence1 = list1
        sequence2 = list2        

        head = tail = []

        tail_size = min(len(sequence1), len(sequence2))

        if tail_size == 1:
            tail = [sequence1[-1]]
            head = [sequence2[0]]
        else:
            tail = sequence1[-tail_size:]
            head = sequence2[:tail_size]
    return head, tail, tail_size 
    
##########################################################################
def similarity_merge(all_sims, donor2img2embeding, donor2day2img, donor):
    no_more_merge = False
    while no_more_merge == False:
        merged_dict = {}
        seen = []
        all_sims_keys = list(all_sims.keys())
        no_more_merge = True

        for key1 in all_sims_keys:
            if key1 in seen:
                continue

            if key1 not in merged_dict :
                # to remove the duplicates
                merged_dict[key1] = list(set(all_sims[key1]))

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
                        #print(tail[img_index], head[img_index], simi)
                        similarity.append(simi)
                    sub_seq_simi = sum(similarity) / tail_size
                    one2nsimi.append([key2,sub_seq_simi])
            if len(one2nsimi) > 0:
                one2nsimi = sorted(one2nsimi, key=lambda x: x[1], reverse=True)
                val = max(one2nsimi[0][1], 0.57)
                if one2nsimi[0][1] >= val:
                    #one2nsimi.append([key2,sub_seq_simi])
                    no_more_merge = False
                    merged_dict[key1].extend(list(set(all_sims[one2nsimi[0][0]])))
                    seen.append(one2nsimi[0][0])
        all_sims = merged_dict
    print_(merged_dict, donor)

####################################################################
def add_to_similarity_dict(all_sims, similarities, key):#, ratio):
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    max_ = similarities[0][1]
    threshold = max(0.99 * max_, 0.80)
    if key not in all_sims:
        all_sims[key] = [key]
    for ind, pair in enumerate(similarities):
        if pair[1] >= threshold:
            #if key not in all_sims:
            #    all_sims[key] = [key]
            all_sims[key].append(pair[0])
    return all_sims


##################################################################
def print_(all_sims, donor):
    label = 0
    for key in all_sims:
        label = label + 1
        for img in all_sims[key]:
            temp = img.replace('MORPH/', '')
            print(temp + ": " + donor + "_" + str(label))

################################################################
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#################################################################
def sequence_finder(donor2img2embeding, donor2day2img):

    for donor in donor2img2embeding:
        days = list(donor2day2img[donor].keys())
        days.sort()
        all_embs = donor2img2embeding[donor]
        all_sims = {} #key = imgs, value = [[im1, dist],im2, dit[],...]

        window_size = 6
        compared = []
        windows = rolling_window(np.array(range(len(days))), window_size)
        #print(windows)
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
                                #print(day1_img, day2_img, sim)
                                similarities.append([day2_img, sim])
                            all_sims = add_to_similarity_dict(all_sims, similarities, key)
        #print_(all_sims, donor)
        all_sims = overlap_merge(all_sims)
        #print_(all_sims, donor)

        similarity_merge(all_sims, donor2img2embeding, donor2day2img, donor)

