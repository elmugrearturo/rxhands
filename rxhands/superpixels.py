import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
import skimage

def skeletonize(one_component_img):
    #
    # SKELETONIZE IN LOCAL OTSU ONE COMPONENT
    # 
    skel_img = skimage.morphology.skeletonize(one_component_img == 255)

    skel_img = skel_img.astype("uint8")*127
    #save_img(one_component_local_otsu - skel_img, results_folder + "skel/" + fname)

    return skel_img


def slic_superpixels(img, sp_mask=None, n_segments=100, compactness=.1):
    #
    # SLIC SUPERPIXELS
    #
    
    m_slic = skimage.segmentation.slic(img, n_segments=n_segments, compactness=compactness, mask=sp_mask, start_label=1, channel_axis=None)
    m_slic_boundaries = skimage.segmentation.mark_boundaries(img, m_slic)
    return m_slic, m_slic_boundaries
