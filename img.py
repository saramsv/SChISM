import os
import sys
import glob
import cv2
from skimage import measure
import numpy as np
import matplotlib.pyplot as plt

rootdir = sys.argv[1]
#rootdir = sys.argv[1]

for subdir, dirs, files in os.walk(rootdir):
    images = []
    base = os.path.join(subdir , files[0])
    base_img = cv2.imread(base)
    height, width = base_img.shape[:2]
    base_img_gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
    print(base)
    
    for dir_ in dirs:
        max_sim_value = 0
        img_with_max_sim = base
        most_similar_img = base_img
        for img in glob.glob(subdir + "/" +  dir_ + "/" + "*.JPG"):
            test_img = cv2.imread(img)
            test_img_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            test_img_gray = cv2.resize(test_img_gray, (width, height), 0, 0, interpolation = cv2.INTER_NEAREST)

            sim_value = measure.compare_ssim(base_img_gray, test_img_gray)
            if(sim_value > max_sim_value):
                max_sim_value = sim_value
                img_with_max_sim = img
                most_similar_img = test_img
        
        images.append(most_similar_img)
        print(img_with_max_sim, ": "  , max_sim_value)

    #printing the result images
    w = 20
    h = 20
    fig = plt.figure(figsize=(20, 20))
    columns = 1
    rows = len(images)
    for i in range(1, columns*rows +1):
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
    break
