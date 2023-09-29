import os
import cv2
import numpy as np

from matplotlib import pyplot as plt

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
import skimage

def normalized_ridge_img(img, black_ridges=False):
    ridge_img = skimage.filters.sato(img, sigmas=range(1, 5), black_ridges=black_ridges)
    ridge_img[ridge_img < 0 ] = 0.
    ridge_norm = cv2.normalize(ridge_img, None, 0, 255, cv2.NORM_MINMAX)
    return ridge_norm

def sobelx_ridge_img(img, black_ridges=False):
    ridge_img = normalized_ridge_img(img, black_ridges=black_ridges)
    sobelx = cv2.Sobel(src=ridge_img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)
    sobelx[sobelx < 0 ] = 0.
    sobelx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    return sobelx

def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            #blur = cv2.medianBlur(img, 7)
            
            ridge_norm = normalized_ridge_img(img, black_ridges=False)
            #ridge_norm = normalized_ridge_img(img, black_ridges=True)
            sobelx = sobelx_ridge_img(img, black_ridges=False)

            # Otsu Threshold
            th, sobelx_th = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            #show_img(sobelx_th, "OTSU")
            #show_img((sobelx > 70).astype("uint8") * 255, "SOBEL")
            #import ipdb;ipdb.set_trace()
            #save_img(ridge_black_img, results_folder + "ridge_black/" + fname)
            save_img(ridge_norm, results_folder + "ridge_norm/" + fname)
            save_img(sobelx, results_folder + "ridge_sobel/" + fname)
            save_img(sobelx_th, results_folder + "ridge_sobel_th/" + fname)
            print("Processed ", fname, " (%s)" % th) 


            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
