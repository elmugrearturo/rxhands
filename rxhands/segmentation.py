import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
import skimage


def to_one_component(bin_img):
    eight_neighbors = np.ones((3, 3), np.uint8)
    # Calculate connected components of binary image
    # and keep only one
    ret, marked_img = cv2.connectedComponents(bin_img)
    center_y = int(marked_img.shape[0]/2)
    center_x = int(marked_img.shape[1]/2)
    # Guess the component in the middle
    selected_component = marked_img[center_y, center_x]

    if selected_component == 0:
        # Guess a little lower
        center_y += int(marked_img.shape[0]/4)
        selected_component = marked_img[center_y, center_x]
         
        if selected_component == 0:
            raise Exception("Nothing found")
    
    # Remove every other component, fill holes and save
    one_component = (marked_img == selected_component).astype("uint8") * 255
    one_component = cv2.morphologyEx(one_component, cv2.MORPH_DILATE, eight_neighbors, iterations=6)
    one_component = skimage.morphology.remove_small_holes(one_component == 255, 500)
    one_component = one_component.astype("uint8") * 255
    return one_component


def four_region_segmentation(img):
    eight_neighbors = np.ones((3, 3), np.uint8)
    
    blur = cv2.medianBlur(img, 7)
    
    #
    # 4-REGION OTSU THRESHOLD
    #
    patches = divide_image(blur, 4)
    thresholds = []
    for patch in patches:
        otsu_thr, otsu_img = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresholds.append(otsu_thr)
    min_threshold = min(thresholds)
    _, patch_threshold_img = cv2.threshold(blur, min_threshold, 255, cv2.THRESH_BINARY)
    dilate_img = cv2.morphologyEx(patch_threshold_img, cv2.MORPH_DILATE, eight_neighbors, iterations=3)
    erode_img = cv2.morphologyEx(dilate_img, cv2.MORPH_ERODE, eight_neighbors, iterations=3)
    
    # Clone result
    erode_img_original = erode_img.copy()
    
    # Remove 1/8 of the image
    eighth_division_h = int(erode_img.shape[0]/8)
    erode_img[erode_img.shape[0] - eighth_division_h:erode_img.shape[0], :] = 0
    
    # Remove small objects
    no_small_obj = skimage.morphology.remove_small_objects(erode_img==255, 32)
    no_small_holes = skimage.morphology.remove_small_holes(no_small_obj, 500)
    clean_thr_img = no_small_holes.astype("uint8") * 255

    one_component_local_otsu = to_one_component(clean_thr_img)
    
    return one_component_local_otsu


def global_otsu_segmentation(img):
    eight_neighbors = np.ones((3, 3), np.uint8)
    
    blur = cv2.medianBlur(img, 7)
     
    #
    # OTSU THRESHOLD
    #
    otsu_thr, otsu_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    dilate_img = cv2.morphologyEx(otsu_img, cv2.MORPH_DILATE, eight_neighbors, iterations=3)
    erode_img = cv2.morphologyEx(dilate_img, cv2.MORPH_ERODE, eight_neighbors, iterations=3)
    
    # Clone result
    erode_img_original = erode_img.copy()
    
    # Remove 1/8 of the image
    eighth_division_h = int(erode_img.shape[0]/8)
    erode_img[erode_img.shape[0] - eighth_division_h:erode_img.shape[0], :] = 0
    
    # Remove small objects
    no_small_obj = skimage.morphology.remove_small_objects(erode_img==255, 32)
    no_small_holes = skimage.morphology.remove_small_holes(no_small_obj, 500)
    clean_thr_img = no_small_holes.astype("uint8") * 255

    one_component_global_otsu = to_one_component(clean_thr_img)
    
    return one_component_global_otsu


def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            try:
                one_component_img = four_region_segmentation(img)
                save_img(four_region_segmentation(img), results_folder + "patch_thresh/" + fname)
                save_img(global_otsu_segmentation(img), results_folder + "otsu/" + fname)
            except Exception as e:
                print("Couldn't find hand: %s" % fname, " ", e)
                continue
            
            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
