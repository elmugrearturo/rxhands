import os
import cv2
import numpy as np
import pickle

import skimage

from rxhands.auxiliary import *
from rxhands.hand_model import Hand
from rxhands.preprocessing import preprocess_image
#from rxhands.ridge import normalized_ridge_img, sobelx_ridge_img
from rxhands.segmentation import four_region_segmentation
from rxhands.superpixels import skeletonize, slic_superpixels
from rxhands.svc_classification import create_classifier

def main(data_folder="./data/", results_folder="./results/", binary_folder="./bin/"):
    
    try:
        with open(binary_folder + "clf.bin", "rb") as fp:
            clf = pickle.load(fp)
        with open(binary_folder + "selected_img_names.bin", "rb") as fp:
            selected_img_names = pickle.load(fp)
    except:
        clf, selected_img_names = create_classifier(train_with_all=True)
        with open(binary_folder + "clf.bin", "wb") as fp:
            pickle.dump(clf, fp)
        with open(binary_folder + "selected_img_names.bin", "wb") as fp:
            pickle.dump(selected_img_names, fp)

    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            #img_patches = img_to_patches(img, (51, 51))
            #for i in range(len(img_patches)):
            #    save_img(img_patches[i], results_folder + f"patches/{i}_" + fname)
            #import ipdb;ipdb.set_trace()
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
            #ridge_img = sobelx_ridge_img(img)
            try:
                one_component_img = four_region_segmentation(img)
            except Exception as e:
                print("Couldn't find hand: %s" % fname, " ", e)
                continue
            print(f"Image {fname}")


            #
            # FIT PARTIAL HAND MODEL
            #
            #
            # Fingers are found from the partially segmented
            # image and the morphological skeleton
            print(f"\tFitting partial hand model...")
            partial_hand_model = Hand(raw_img, img, one_component_img)
            
            ## 
            ## CLASSIFY
            ##
            partial_hand_model.classify_internal_finger_points(clf)
            
            ## 
            ## DISPLAY
            ##
            
            print(f"\t\tMarking regions in raw_img...")
            marked_img = partial_hand_model.paint_to_img(raw_img)
            
            for finger in partial_hand_model.fingers:
                clusters = finger.approximate_prediction_points("kmeans")
                for label in clusters:
                    marked_img = cv2.circle(marked_img, clusters[label][::-1], 2, (0, 255, 0), -1)

            save_img(marked_img, results_folder + "hand_model/" + fname)
            
        #break

if __name__ == "__main__" :
    main()
