import os
import cv2
import numpy as np

from rxhands.auxiliary import *
import skimage

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def preprocess_image(raw_gray_img):
    # Equalize histogram
    eq_img = cv2.equalizeHist(raw_gray_img)
    normalized_eq_img = eq_img.astype(np.float32) / 255.
    # Calculate Gamma correction
    # Gamma = log(mean) / log(0.5) (intensity interval 0 to 1)
    gamma = np.log(np.mean(normalized_eq_img)) / np.log(0.5)
    adjusted_img = adjust_gamma(eq_img, .25)
    return adjusted_img

def main(data_folder="./data/", results_folder="./results/"):
    #kernel = np.ones((5,5), np.uint8)
    #eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            img = load_gray_img(data_folder + fname)
            preprocessed_img = preprocess_image(img)

            save_img(preprocessed_img, results_folder + "pre/" + fname)
            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
