import os
import cv2
import numpy as np

import skimage

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
from rxhands.segmentation import four_region_segmentation
from rxhands.superpixels import skeletonize, slic_superpixels


def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            try:
                one_component_img = four_region_segmentation(img)
            except Exception as e:
                print("Couldn't find hand: %s" % fname, " ", e)
                continue
            #
            # SKELETONIZE
            #
            skel_img = skeletonize(one_component_img)
            save_img(one_component_img - skel_img, results_folder + "skel/" + fname)
            m_slic, m_slic_boundaries = slic_superpixels(img, one_component_img) 
            save_img(skimage.img_as_ubyte(m_slic_boundaries), results_folder + "mslic/" + fname)
            
            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
