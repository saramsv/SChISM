import sys
import numpy as np
import csv
import ast
import datetime
import math

def key_func(x):
    # For some year like 2011 the year is 2 digits so the date format should be %m%d%y but for others like 2015 it should be %m%d%Y
    try:
        if '(' in x:     
            return datetime.datetime.strptime(x.split('D_')[-1].split('(')[0].strip().replace('_',''), '%m%d%y')
        else: 
            return datetime.datetime.strptime(x.split('D_')[-1].split('.')[0].strip().replace('_',''), '%m%d%y')
    except:
        print(x)
        import bpython
        bpython.embed(locals())
        exit()

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
        
        sequence1_times = [x.split("os//")[1].split()[0] for x in sequence1]
        sequence2_times = [x.split("os//")[1].split()[0] for x in sequence2]
        time_overlap = list(set(sequence1_times).intersection(set(sequence1_times)))

        tail = [x for x in sequence1 if x.split("os//")[1].split()[0] in time_overlap]
        head = [x for x in sequence2 if x.split("os//")[1].split()[0] in time_overlap]

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
                        similarity.append(simi)
                    sub_seq_simi = sum(similarity) / tail_size
                    one2nsimi.append([key2,sub_seq_simi])
            if len(one2nsimi) > 0:
                one2nsimi = sorted(one2nsimi, key=lambda x: x[1], reverse=True)
                val = max(one2nsimi[0][1], 0.5)
                if one2nsimi[0][1] >= val:
                    #one2nsimi.append([key2,sub_seq_simi])
                    no_more_merge = False
                    merged_dict[key1].extend(list(set(all_sims[one2nsimi[0][0]])))
                    merged_dict[key1] = sorted(merged_dict[key1], key = key_func)
                    seen.append(one2nsimi[0][0])
        all_sims = merged_dict
    print_(merged_dict, donor)

####################################################################
def add_to_similarity_dict(all_sims, similarities, key):#, ratio):
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    max_ = similarities[0][1]
    threshold = max(0.99 * max_, 0.75)
    #if key not in all_sims:
    #    all_sims[key] = [key]
    for ind, pair in enumerate(similarities):
        if pair[1] >= threshold:
            if key not in all_sims:
                all_sims[key] = []
            all_sims[key].append(pair[0])
    return all_sims


##################################################################
def print_(all_sims, donor):
    label = 0
    for key in all_sims:
        label = label + 1
        for img in all_sims[key]:
            temp = img.replace('JPG', 'icon.JPG: ')
            print(temp + donor + "_" + str(label))

#################################################################
def sequence_finder(donor2img2embeding, donor2day2img):

    for donor in donor2img2embeding:
        #if donor == 'UT16-13D':
        days = list(donor2day2img[donor].keys())
        days.sort()
        all_embs = donor2img2embeding[donor]
        all_sims = {} #key = imgs, value = [[im1, dist],im2, dit[],...]

        window = 5

        for i in range(window, len(days) - 30 - window):
            imgs = donor2day2img[donor][days[i]] # images for day = days[i]

            for img in imgs:
                emb = all_embs[img]
                key = img
                for seen in all_sims:
                    for x in all_sims[seen]:
                        if img ==  x: # if it is one of the matched ones
                            key = seen
                    
                '''
                day_current_imgs = donor2day2img[donor][days[i]]
                similarities = []
                for day_current_img in day_current_imgs:
                    emb2 = all_embs[day_current_img] 
                    sim = cosine_similarity(emb, emb2)
                    similarities.append([day_current_img, sim])
                all_sims = add_to_similarity_dict(all_sims, similarities, key)
                '''

                for w in range(1, window + 1):
                    day_before_imgs = donor2day2img[donor][days[i-w]]
                    day_after_imgs = donor2day2img[donor][days[i+w]]
                    similarities = []
                    for day_before_img in day_before_imgs:
                        emb2 = all_embs[day_before_img] 
                        sim = cosine_similarity(emb, emb2)
                        similarities.append([day_before_img, sim])
                    all_sims = add_to_similarity_dict(all_sims, similarities, key)

                    similarities = []
                    for day_after_img in day_after_imgs:
                        emb2 = all_embs[day_after_img] 
                        sim = cosine_similarity(emb, emb2)
                        similarities.append([day_after_img, sim])
                    all_sims = add_to_similarity_dict(all_sims, similarities, key) 

        #print_(all_sims, donor)
        all_sims = overlap_merge(all_sims)
        print_(all_sims, donor)

        #similarity_merge(all_sims, donor2img2embeding, donor2day2img, donor)

