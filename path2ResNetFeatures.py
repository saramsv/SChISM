# only use it to generate the resnet features
#how to run: python3 path2ResNetFeatures.py --img_path some_imgs  > resnet_feautres_filename
import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
import numpy as np
from keras.models import Model
import cv2
import sys
import csv
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type = str)

args = parser.parse_args()

imgs_path = args.img_path

img_size = 224
resnet_weigth_path = 'data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

clustering_model = Sequential()

clustering_model.add(ResNet50(include_top = False, pooling='ave', weights = resnet_weigth_path))
clustering_model.layers[0].trainable = False


clustering_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

missed_imgs = []
    
for path in glob.glob(imgs_path + '/*.jpg'):
    correct_path = path
    row = []
    try:
        img_object = cv2.imread(correct_path)
        img_object = cv2.resize(img_object, (img_size, img_size))
        img_object = np.array(img_object, dtype = np.float64)
        img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

        resnet_feature = clustering_model.predict(img_object)
        resnet_feature = np.array(resnet_feature)
        import bpython
        bpython.embed(locals())
        row.append(correct_path)
        row.extend(list(resnet_feature.flatten()))
        
        print(row)
    except: 
        missed_imgs.append(path)
