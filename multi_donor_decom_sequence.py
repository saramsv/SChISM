import sys
import numpy as np
import csv
import ast
import datetime
import math

#threshold = [0.7, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92, 0.95, 0.97]
threshold = [0.85]#, 0.75, 0.8, 0.85, 0.9, 0.95]

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

def overlap_merge(all_sims, all_embs):
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
                    if key1 != key2 and key2 not in seee:
                        intersect = len(set(all_sims[key1]).intersection(set(all_sims[key2])))
                        if intersect != 0:
                            no_more_merge = False
                            '''
                            merged_dict[key1].extend(list(set(all_sims[key2])))
                            all_embs[key1].extend(list(set(all_embs[key2])))
                            '''
                            try:
                                merged_dict[key1].extend(all_sims[key2])
                                all_embs[key1].extend(all_embs[key2])
                            except:
                                import bpython
                                bpython.embed(locals())
                                exit()

                            del all_embs[key2]
                            merged_dict[key1] = sorted(merged_dict[key1], key = key_func)
                            seen.append(key2)
                            seen.append(key1)
        all_sims = merged_dict
    return all_sims, all_embs
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
                    sub_seq_simi = sum(similarity) / tail_size #average similarity
                    one2nsimi.append([key2,sub_seq_simi])
            if len(one2nsimi) > 0:
                one2nsimi = sorted(one2nsimi, key=lambda x: x[1], reverse=True)
                val = max(one2nsimi[0][1], 0.8)
                if one2nsimi[0][1] >= val:
                    #one2nsimi.append([key2,sub_seq_simi])
                    no_more_merge = False
                    merged_dict[key1].extend(list(set(all_sims[one2nsimi[0][0]])))
                    merged_dict[key1] = sorted(merged_dict[key1], key = key_func)
                    seen.append(one2nsimi[0][0])
        all_sims = merged_dict
    print_(merged_dict, donor)

#########################################################################
def cluster_similarity(all_sims, all_embs, threshold):
    print(threshold)
    keys = list(all_sims.keys())
    cluster_ave = {}
    for key in keys:
        vectors = 0
        num = len(all_embs[key])
        for emb in all_embs[key]:
            vectors += np.asarray(emb)
        cluster_ave[key] = list(vectors/num) # the average of the embeddings in the cluster

    no_more_merge = False
    threshold = threshold - 0.1
    while no_more_merge == False:
        print("another_loop")
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
            if l[2] < threshold:
                break
            if l[0] not in seen and l[1] not in seen and l[2] > threshold:
                #l[0] is key1, l[1] is key2, l[2] is the similarity
                no_more_merge = False
                all_sims[l[0]].extend(list(set(all_sims[l[1]])))
                del all_sims[l[1]]
                cluster_ave[l[0]] = list((np.asarray(cluster_ave[l[0]]) +
                                        np.asarray(cluster_ave[l[1]]))/2) # the new ave
                del cluster_ave[l[1]]
                seen.add(l[0])
                seen.add(l[1])
    print_(all_sims, threshold)


##########################################################################
def similarity_merge2(all_sims, donor2img2embeding, donor2day2img, donor, threshold):
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
                val = max(one2nsimi[0][1], threshold - 0.1)
                if one2nsimi[0][1] >= val:
                    #one2nsimi.append([key2,sub_seq_simi])
                    no_more_merge = False
                    all_sims[key1].extend(list(set(all_sims[one2nsimi[0][0]])))
                    del all_sims[one2nsimi[0][0]]
                    seen.append(one2nsimi[0][0])
    print("number of the clusters after sim:")
    print(len(list(all_sims.keys())))
    cluster_similarity(all_sims, donor2img2embeding, donor2day2img, donor, threshold)

####################################################################
def add_to_similarity_dict(all_sims, similarities, key, t, all_embs, emb1, embs):
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    embs = sorted(embs, key=lambda x: x[1], reverse=True)
    max_ = similarities[0][1]
    threshold = max(0.99 * max_, t)
    if key not in all_sims:
        all_sims[key] = [key]
        all_embs[key] = [emb1]
    for ind, pair in enumerate(similarities):
        if pair[1] >= threshold:
            #if key not in all_sims:
            #    all_sims[key] = []
            all_sims[key].append(pair[0])
            all_embs[key].append(embs[ind][0])
    print(len(all_sims))
    print(len(all_embs))
    return all_sims, all_embs


##################################################################
def print_(all_sims, threshold):
    label = 0
    with open('clusters_multidonor', 'a') as cluster_file:
        for key in all_sims:
            label = label + 1
            for img in all_sims[key]:
                temp = img.replace('JPG', 'icon.JPG: ')
                row = temp + str(label) + '\n'
                cluster_file.write(row)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

#################################################################
def sequence_finder(donor2img2embeding, donor2day2img, daily_data):
    #daily_data[day][0] is the image names for day
    #daily_data[day][1] is the image embeddings for day
    days = list(daily_data.keys())
    days.sort()
    all_sims = {} #key = imgs, value = [[im1, dist],im2, dit[],...]
    all_embs = {}
    print(days)
    for t in threshold:
        window_size = 4
        compared = []
        windows = rolling_window(np.array(range(len(days))), window_size)
        for window in windows:
            for ind1_ in window:
                for ind2_ in window:
                    ind1 = days[ind1_]
                    ind2 = days[ind2_]
                    if ind1 == ind2:
                        continue
                    else:
                        pair = (ind1, ind2)
                        pair2 = (ind2, ind1)
                        if pair not in compared and pair2 not in compared:
                            compared.append(pair)
                            compared.append(pair2)
                            day1_ind = pair[0]
                            day2_ind = pair[1]
                            day1_imgs = daily_data[day1_ind][0]
                            day1_embs = daily_data[day1_ind][1]
                            day2_imgs = daily_data[day2_ind][0]
                            day2_embs = daily_data[day2_ind][1]
                      
                            for index, day1_img in enumerate(day1_imgs):
                                emb = day1_embs[index]
                                key = day1_img
                                for seen in all_sims:
                                    for x in all_sims[seen]:
                                        if day1_img ==  x: # if it is one of the matched ones
                                            key = seen
                                    
                                similarities = []
                                embs = []
                                for index2, day2_img in enumerate(day2_imgs):
                                    emb2 = day2_embs[index2]
                                    sim = cosine_similarity(emb, emb2)
                                    similarities.append([day2_img, sim])
                                    embs.append([emb2, sim])
                                all_sims, all_embs = add_to_similarity_dict(all_sims, similarities, key, t, all_embs, emb,embs)

        all_sims2, all_embs2 = overlap_merge(all_sims, all_embs)

        cluster_similarity(all_sims2, all_embs2, t)
 
