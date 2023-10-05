import os
import cv2
import numpy as np

import skimage

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
from rxhands.geometry import *
from rxhands.segmentation import four_region_segmentation


def slic_superpixels(img, sp_mask=None, n_segments=100, compactness=.1):
    #
    # SLIC SUPERPIXELS
    #
    
    m_slic = skimage.segmentation.slic(img, n_segments=n_segments, compactness=compactness, mask=sp_mask, start_label=1, channel_axis=None)
    m_slic_boundaries = skimage.segmentation.mark_boundaries(img, m_slic)
    return m_slic, m_slic_boundaries

def watershed_superpixels(img, sp_mask=None, n_segments=250, compactness=.001):
    gradient = skimage.filters.sobel(img)
    ridge_img = skimage.filters.sato(img, sigmas=range(1, 5), black_ridges=True)
    watershed_segments = skimage.segmentation.watershed(ridge_img, markers=n_segments, mask=sp_mask, compactness=compactness)
    watershed_img_boundaries = skimage.segmentation.mark_boundaries(img, watershed_segments)

    return watershed_segments, watershed_img_boundaries

def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            print("\nFile: %s" % fname)
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            try:
                one_component_img = four_region_segmentation(img)
            except Exception as e:
                print("Couldn't find hand: %s" % fname, " ", e)
                continue
            print("\tCalculated one-component img")

            m_slic, m_slic_boundaries = watershed_superpixels(img, one_component_img, 200, .001)
            save_img(skimage.img_as_ubyte(m_slic_boundaries), results_folder + "mslic/" + fname)
            #import ipdb; ipdb.set_trace()

        #break

if __name__ == "__main__" :
    main()

