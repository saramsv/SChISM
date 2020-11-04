#python3 path2embeding.py --img_path data/some_paths --weight_type pt --feature_type PCA --save_to resnet_features_filename > pca_feautres_filename
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
from keras.models import Model
from multiprocessing import Pool
import cv2
import sys
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type = str)
parser.add_argument('--weight_type', type = str)
parser.add_argument('--feature_type', type = str)
parser.add_argument('--save_to', type = str)

args = parser.parse_args()

imgs_path = args.img_path
weight_type = args.weight_type
feature_type = args.feature_type
embeding_file = args.save_to


img_size = 224
resnet_weigth_path = 'data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
fine_tuned_resnet_weight_path = 'ResNet50/logs/ft-41-0.87.hdf5'

clustering_model = Sequential()

if weight_type == 'pt': # this is for pre_trained
    clustering_model.add(ResNet50(include_top = False, pooling='ave', weights = resnet_weigth_path))
    clustering_model.layers[0].trainable = False
elif weight_type == 'ft':
    num_classes = 9
    base_model = ResNet50
    base_model = base_model(weights = 'imagenet', include_top = False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation= 'relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    clustering_model = Model(inputs = base_model.input, outputs = predictions)
    clustering_model.load_weights(fine_tuned_resnet_weight_path)

    clustering_model.layers.pop()
    clustering_model.layers.pop()
    clustering_model.outputs = [clustering_model.layers[-1].output]
    clustering_model.layers[-1].outbound_nodes = []

    clustering_model.layers[0].trainable = False

clustering_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

rows = []
all_paths = []

with open(imgs_path) as csv_file:
    paths = csv.reader (csv_file, delimiter=',') 
    ## all of the paths in the given file should be separated by , 
    ##so that the endresult for paths after the for loop would be
    ## a list with each elemant being a path to an image
    for path in paths:
        all_paths = path
        print("[INFO] number of read paths are {}".format(len(all_paths)))

def get_img(path):
    correct_path = path#[0]
    correct_path.replace(' ', '\ ')
    correct_path.replace('(', '\(')
    correct_path.replace(')', '\)')
    print(correct_path)
    img_object = cv2.imread(correct_path)
    img_object = cv2.resize(img_object, (img_size, img_size))
    img_object = np.array(img_object, dtype = np.float64)
    img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))

    resnet_feature = clustering_model.predict(img_object)
    resnet_feature = np.array(resnet_feature)
    return resnet_feature

def cal_fearue(path):

        resnet_feature = get_img(path)
        if feature_type == 'PCA':
            img_names.append(correct_path)
            return list(resnet_feature.flatten())
        elif feature_type == 'resnet':
            row = []
            row.append(correct_path)
            row.extend(list(resnet_feature.flatten()))
            return row
p = Pool(10)
rows = p.map(cal_fearue, all_paths)
print("[INFO] Finised Generating the Features")

'''
if feature_type == 'PCA':
    vectors = np.array(rows)
    model = PCA(n_components = 5)
    results = model.fit_transform(vectors)
    for index, img in enumerate(img_names):
        print(img, ",", list(results[index,:]))

if feature_type == 'resnet':
    with open(embeding_file+ '.csv', 'w') as csv_file:
        writer = csv.writer(csv_file, lineterminator = '\n')
        writer.writerows(rows)
'''
