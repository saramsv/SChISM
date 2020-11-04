#This method clusters images of 3 neighboring days  and then moves the window one forward and repeat... the clusters with the most image overlaps are merged.
import numpy as np
from sklearn.cluster import SpectralClustering

def streaming_cluster(donor2img2embeding, donor2day2img):
    n_clusters = 7

    # For each donor
    for donor in donor2day2img:
        days = list(donor2day2img[donor].keys())
        days.sort()

        # Window information
        start = 0
        window_size = 3
        end = start + window_size

        # Store all window clusters
        window_clusters_imgs = []
        
        while end < len(days):
            img_names = []
            embeddings = []
            # For every day in our window
            for day_index in range(start, end): 
                day = days[day_index]
                # Grab all the features in this window for this donor
                for img in donor2day2img[donor][day]:
                    img_names.append(img)
                    embeddings.append(donor2img2embeding[donor][img])

            embeddings = np.array(embeddings)
            embeddings = embeddings / embeddings.max(axis = 0)
            kmeans = SpectralClustering(n_clusters = n_clusters, affinity='nearest_neighbors', assign_labels='kmeans')
            labels = kmeans.fit_predict(embeddings)
            #labels = kmeans.predict(embeddings)

            window_cluster = {}
            for i, label in enumerate(labels):
                if label in window_cluster: 
                    window_cluster[label].append(img_names[i])
                else: 
                    window_cluster[label] = [img_names[i]]

            window_clusters_imgs.append(window_cluster)

            start += 1
            end = start + window_size

        # Merge the clusters
        i = 0
        merges = []
        while i < len(window_clusters_imgs) - 1:
            merged = []
            first_clusters = window_clusters_imgs[i] # For the ith window
            second_clusters = window_clusters_imgs[i + 1] # For the (i + 1)th window
            for c1_index, c1_imgs in first_clusters.items(): 
                max_overlap = 0
                max_overlap_index = -1
                for c2_index, c2_imgs in second_clusters.items():
                    overlap = len(set(c1_imgs).intersection(set(c2_imgs)))
                    if overlap > max_overlap: 
                        max_overlap = overlap
                        max_overlap_index = c2_index
            

                merged.append([c1_index , max_overlap_index])
            merges.append(merged)
            i += 1
            
            
        def findNext(pair, bin_index):
            if bin_index >= len(merges):
                return []
            else: 
                for group in merges[bin_index]:
                    if group[0] == pair[1]: 
                        # found the associated pair
                        return [group[1]] + findNext(group, bin_index + 1)
        groups  = []

        for i, pair in enumerate(merges[0]):
            temp = pair + findNext(pair, 1)
            # temp is a list [a, b, c, ...z] where its index is the bin_index and its values are cluster numbers

            group = []#{'cluster': [], 'day': 0}
            # group is a temporary super-cluster that is used to break the chains 

            for window_index, cluster in enumerate(temp):
                # skip last one since it'll be out of bounds
                group.append(cluster)
                if window_index == len(temp) - 1: 
                    groups.append(group)

        clusters = []
        for group in groups:
            images = []
            for window_index, cluster_index in enumerate(group): 
                images += window_clusters_imgs[window_index][cluster_index]
            clusters.append(sorted(images, key = key_func))
        for i, imgs in enumerate(clusters):
            for img in imgs:
                print(img, ":",donor,"_", i) 
