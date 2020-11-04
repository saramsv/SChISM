from skimage import measure
#from skimage.measure import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import sys

src = sys.argv[1]


def compare_images(imageA, imageB, title):
    # compute the mean squared error and structural similarity
    # index for the images
    s = ssim(imageA, imageB)

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))

    # show first image
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(imageA, cmap = plt.cm.gray)
    plt.axis("off")

    # show the second image
    ax = fig.add_subplot(1, 2, 2)
    plt.imshow(imageB, cmap = plt.cm.gray)
    plt.axis("off")

    # show the images
    plt.show()

images = []
for img in glob.glob(src + "*.JPG"):
   img_object = cv2.imread(img)
   gray = cv2.cvtColor(img_object, cv2.COLOR_BGR2GRAY)
   images.append((img, gray))
   # initialize the figure

print("printing result:")
original_image = 'data/extra/UT26-16D_05_06_2016 (1).JPG'
for (name, gray_img) in images:
   s = measure.compare_ssim(images[3][1], gray_img)
   #s = ssim(images[0][1], gray_img)
   print(images[3][0], name, s)
