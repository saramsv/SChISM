#python3 clustering_deeplearningtl.py data/UT06-12D # > resnet_clusters 
#cat resnet_clusters | sort --field-separator=":" --key=2 > resnet_clusters
import keras
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import glob
import cv2
import sys
import datetime
import pickle


imgs_path = sys.argv[1]

img_size = 224
resnet_weigth_path = 'data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

clustering_model = Sequential()
clustering_model.add(ResNet50(include_top = False, pooling='ave', weights = resnet_weigth_path))
clustering_model.layers[0].trainable = False
clustering_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

img_names = {}
embedings = {}

def convert_to_time(img_name):
    date = img_name.split("D_")[1].split(' ')[0]  
    # this is the format and we only need the date part UT06-12D_07_26_12 (21).JPG
    date = date.replace('_', '') # to remove the '_'
    return datetime.datetime.strptime(date, '%m%d%y') #formated as date

def save_to_pickle(dict_, file_name):
    f = open(file_name + '.pkl', 'wb')
    pickle.dump(dict_, f)
    #f.colse()
    
def extract_vector(path):
    #resnet_feature_list = []
    first_img = True
    for img in sorted(glob.glob(path + "*.JPG")):

        if first_img:   
            start_time = convert_to_time(img)
            first_img = False

        img_time = convert_to_time(img)
        time_from_start = (img_time - start_time).days

        if time_from_start not in embedings and time_from_start not in img_names:
            img_names[time_from_start] = []
            embedings[time_from_start] = []

        img_names[time_from_start].append(img)
        img_object = cv2.imread(img)
        img_object = cv2.resize(img_object, (img_size, img_size))
        img_object = np.array(img_object, dtype = np.float64)
        img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

        resnet_feature = clustering_model.predict(img_object)
        resnet_feature = np.array(resnet_feature)
        embedings[time_from_start].append(list(resnet_feature.flatten()))
        #resnet_feature_list.append(resnet_feature.flatten())
        print(embedings)
        save_to_pickle(embedings, 'embedings')
        save_to_pickle(img_names, 'names')

    return embedings, img_names #np.array(resnet_feature_list)

embedings, names  = extract_vector(imgs_path)

