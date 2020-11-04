#this script save the large claster in a file and the small ones along their embeddings in another
#run: python3 find_large_clusters.py dest_file1 dest_file2 img_num
import pandas as pd
import csv
import sys


if len(sys.argv) < 4:
    print("Usage: python3 find_large_clusters.py src_file large_clusters_file small_clusters_file embedding_file img_num")
    sys.exit()

src_file = sys.argv[1] # read from
large_clusters_file = sys.argv[2] # where to save
small_clusters_file = sys.argv[3] # where to save
embedding_file = sys.argv[4] # the file that containes the embeddings
img_num = int(sys.argv[5]) # the number of images in the cluster

df = pd.read_csv(src_file, sep=':', names=['path', 'label'])
emb_df = pd.read_csv(embedding_file, sep='G', names=['path', 'emb'])

large_clusters = pd.DataFrame()
small_clusters = pd.DataFrame()


labels = df.label.unique()
for l in labels:
    rows = df.loc[df['label'] == l]
    if rows.shape[0] > img_num:
        large_clusters = large_clusters.append(rows, ignore_index = True)
    else:
        for index, row in rows.iterrows():
            line = emb_df.loc[emb_df['path'] + 'G' == row['path'].replace('.icon','')]
            small_clusters = small_clusters.append(line, ignore_index = True)
large_clusters.to_csv(large_clusters_file, index = None, sep= ':', header = False)
small_clusters.to_csv(small_clusters_file, index = None, sep= ':', header = False)
