#pythn3 img2embeding.py paths_file# csv_dest_file
import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
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
    
with open(imgs_path) as csv_file:
    paths = csv.reader (csv_file, delimiter='\n')
    embedings = []

    for path in paths:
        all_imgs_in_path = os.listdir(os.path.join(path[0].split('Daily')[0], 'Daily Photos'))
        for img in all_imgs_in_path :
            row = []
            if path[0].split('Photos//')[1] in img and 'icon' not in img and 'html' not in img:#the name part which does not include the (..)
                correct_path = path[0].replace(path[0].split('Photos//')[1], img)
                correct_path.replace(' ', '\ ')
                correct_path.replace('(', '\(')
                correct_path.replace(')', '\)')
                img_object = cv2.imread(correct_path)
                img_object = cv2.resize(img_object, (img_size, img_size))
                img_object = np.array(img_object, dtype = np.float64)
                img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

                resnet_feature = clustering_model.predict(img_object)
                resnet_feature = np.array(resnet_feature)
                row.append(correct_path)
                row.append(list(resnet_feature.flatten()))
                print(row)
                #embedings.append(row)
'''
with open(embeding_file+ '.csv', 'w') as csv_file:
    writer = csv.writer(csv_file, lineterminator = '\n')
    writer.writerows(embedings)
'''
