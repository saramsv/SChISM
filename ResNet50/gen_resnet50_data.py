import csv
import os
import cv2
import sys
import numpy as np

def gen_data():
    img_paths = "labeled_body_parts_clean.txt" #sys.argv[1]
    img_size = 224

    X_train = []
    Y_train = []
    missed_imgs = []

    with open(img_paths) as csv_file:
        paths = csv.reader(csv_file, delimiter = '\n')
        
        for path in paths:
            path = path[0] #becase path is a lsit and I need the content
            path = path.split(':')
            correct_path = path[0]
            tag = path[1]
            correct_path.replace(' ', '\ ')
            correct_path.replace('(', '\(')
            correct_path.replace(')', '\)')
            try:
                if os.path.exists(correct_path) == False:
                    correct_path = correct_path.replace("Photo/", "Photos/")
                img_object = cv2.imread(correct_path)
                img_object = cv2.resize(img_object, (img_size, img_size))
                img_object = np.array(img_object, dtype = np.float64) 
                #img_object = preprocess_input(np.expand_dims(img_object.copy(), axis = 0))
                #print(img_object.shape)
                X_train.append(img_object)
                Y_train.append(tag)

            except: 
                missed_imgs.append(path)
    return X_train, Y_train
