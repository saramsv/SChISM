import csv
import os
import cv2
import sys

img_paths = sys.argv[1]
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
            X_train.append(img_object)
            Y_train.append(tag)

        except: 
            missed_imgs.append(path)

return X_train, Y_train
