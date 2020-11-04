import numpy as np
import sys
import csv
from operator import itemgetter

if(len(sys.argv) < 2):
    print("Usage: python3 eval.py labels_file > outputfile")
    sys.exit()

def get_correct_line(cluster_labels, all_lines, labels):
    all_correct_lines = []
    cluster_names = list(cluster_labels.keys())
    for cluster_name in cluster_names:
        dominant_l = cluster_labels[cluster_name]
        dominant_l_indices = [i for i in range(len(labels[cluster_name])) if labels[cluster_name][i] == dominant_l]
        lines = itemgetter(*dominant_l_indices)(all_lines[cluster_name])
        for line in lines:
            all_correct_lines.append(line)

    return all_correct_lines
        
        
# get image name and label from file
f = open(sys.argv[1], "r")
lines = f.readlines()


labels = {}
all_lines = {}

for line in lines:
    line = line.replace('\n', '')
    gt_label = line.split('_')[0]
    label = line.split(':')[-1]
    #label = label.replace('\n', '')
    label = label.strip()
    if label in labels:
        labels[label].append(gt_label)
        all_lines[label].append(line)
    else:
        labels[label] = [gt_label]
        all_lines[label] = [line]


acc = []
bodypart_accs = {}
cluster_labels = {}
#labels are the predicted labels
num_clusters = 0
for label in sorted(labels.keys()):
    max_count = 0
    num_clusters += 1
    dominant_l  = ""
    #print("Cluster #", label)
    #print("Total Images: ", len(labels[label]))
    seen = []
    for gt_l in labels[label]:
        if gt_l not in seen:
            seen.append(gt_l)
            #print("\t ID #", l, end=" ")
            #print("--", labels[label].count(l))
            if(labels[label].count(gt_l) >= max_count):
                dominant_l = gt_l
                max_count = labels[label].count(gt_l)
            if dominant_l not in bodypart_accs:
                bodypart_accs[dominant_l] = []
    #print("Best ID: ", maxID, "--", max_)
    ac = max_count/len(labels[label])*100
    bodypart_accs[dominant_l].append(ac)
    #print("Accuracy ", ac)
    acc.append(ac)
    ## get the corresponding line to the correctly clustered images
    cluster_labels[label] = dominant_l

result = get_correct_line(cluster_labels, all_lines, labels)
'''
with open(sys.argv[1]+"corrected", 'w') as f:
    writer = csv.writer(f, delimiter = "\n")
    writer.writerow(result)
'''

acc = np.array(acc)
print("Average accuracy: ", np.average(acc))
for key in bodypart_accs.keys():
    bodypart_accs[key] = np.average(np.array(bodypart_accs[key]))
print(bodypart_accs)
print("number of clusters: {}".format(num_clusters))







