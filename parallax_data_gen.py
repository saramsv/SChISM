import numpy as np
import pickle
import sys
import csv


embedings_file = sys.argv[1]
img_names_file = sys.argv[2]


def read_pkl_file(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    return data

#read in the files in dictionary format
embedings = read_pkl_file(embedings_file)
img_names = read_pkl_file(img_names_file)


def make_csv_rows(img_names, embedings):
    rows = []
    for key in embedings: #assumming the embeding and img_names are created correctly and the both share the same keys
        for img_number in range(len(img_names[key])):
            row = []
            row.append(img_names[key][img_number])
            vectors = np.array(embedings[key][img_number])
            row.extend(vectors)
            rows.append(row)
    return rows


data = make_csv_rows(img_names, embedings)
with open('one_donor_embeding_parallax.csv', 'w') as f:
    writer = csv.writer(f, lineterminator = '\n')
    writer.writerows(data)
    
    
        
