import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
import skimage


def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            #img = cv2.equalizeHist(raw_img)
            img_color = load_color_img(data_folder + fname)
            raw_blur = cv2.medianBlur(raw_img, 7)
            blur = cv2.medianBlur(img, 7)
            #norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

             
            # SOBEL FILTER
            #
            sobelxy = cv2.Sobel(blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
            sobelx = cv2.Sobel(blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # X Sobel Edge Detection
            save_img(sobelxy, results_folder + "sobelxy/" + fname)
            save_img(sobelx, results_folder + "sobelx/" + fname)
            
            # CANNY DETECTOR
            #
            edges = cv2.Canny(blur, 50, 255)
            save_img(edges, results_folder + "canny/" + fname)

            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
