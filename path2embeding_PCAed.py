#python3 all_img2embeding.py data/flat_image_list_sorted.txt  > all_embedings.csv
#then clean the [] and , from the all_embedings.csv
import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import cv2
import glob
import sys
import datetime
import pickle
import os
import csv

imgs_path = sys.argv[1]
#embeding_file = sys.argv[2]

img_size = 224
resnet_weigth_path = 'data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

clustering_model = Sequential()
clustering_model.add(ResNet50(include_top = False, pooling='ave', weights = resnet_weigth_path))
clustering_model.layers[0].trainable = False
clustering_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

line_number = 0
missed_imgs = []
rows = []
    
with open(imgs_path) as csv_file:
    paths = csv.reader (csv_file, delimiter='\n')
    embedings = []
    img_names = []
    for path in paths:
        line_number += 1
        row = []
        correct_path = path[0]
        correct_path.replace(' ', '\ ')
        correct_path.replace('(', '\(')
        correct_path.replace(')', '\)')
        try:
            img_object = cv2.imread(correct_path)
            img_object = cv2.resize(img_object, (img_size, img_size))
            img_object = np.array(img_object, dtype = np.float64)
            img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

            resnet_feature = clustering_model.predict(img_object)
            resnet_feature = np.array(resnet_feature)
            img_names.append(correct_path)
            row.extend(list(resnet_feature.flatten()))
            #print(row)
            rows.append(row)
        except: 
            missed_imgs.append(path)
        #embedings.append(row)
vectors = np.array(rows)
model = PCA(n_components = 32)
results = model.fit_transform(vectors)
for index, img in enumerate(img_names):
    print(img, ",", list(results[index,:]))

print(missed_imgs)
'''
with open(embeding_file+ '.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, lineterminator = '\n')
    writer.writerows(embedings)
'''
