import pandas as pd
import csv
import sys

if len(sys.argv) < 3:
    print("Usage: python3 merge_gt_cluster_labels.py gt_file cluster_file > results")
    sys.exit()

gt_file = sys.argv[1]
cluster_file = sys.argv[2]

gt_df = pd.read_csv(gt_file, sep =":",  names=['path', 'gt_label'])
cluster_df = pd.read_csv(cluster_file, sep=':', names= ['path', 'cluster_label'])

merged = pd.DataFrame()


for index, row in cluster_df.iterrows():
    path = row['path'].strip()
    cluster_label = row['cluster_label'].strip()
    try:
        gt = gt_df.loc[gt_df['path'] == path]
        gt_label = gt['gt_label'].values[0].strip()
        print(gt_label + "_"+ path + ":" + cluster_label)
    except:
        pass

