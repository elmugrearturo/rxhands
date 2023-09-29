import os
import cv2
import numpy as np
import pickle

import skimage

from rxhands.auxiliary import *
from rxhands.geometry import *
from rxhands.preprocessing import preprocess_image
from rxhands.ridge import normalized_ridge_img, sobelx_ridge_img
from rxhands.segmentation import four_region_segmentation
from rxhands.superpixels import skeletonize, slic_superpixels
from rxhands.classification import create_classifier, extract_haar_features


def main(data_folder="./data/", results_folder="./results/", binary_folder="./bin/"):
    
    try:
        with open(binary_folder + "clf.bin", "rb") as fp:
            clf = pickle.load(fp)
        with open(binary_folder + "feature_type_sel.bin", "rb") as fp:
            feature_type_sel = pickle.load(fp)
        with open(binary_folder + "feature_coord_sel.bin", "rb") as fp:
            feature_coord_sel = pickle.load(fp)
        with open(binary_folder + "selected_img_names.bin", "rb") as fp:
            selected_img_names = pickle.load(fp)
    except:
        clf, feature_type_sel, feature_coord_sel, selected_img_names = \
                create_classifier()
        with open(binary_folder + "clf.bin", "wb") as fp:
            pickle.dump(clf, fp)
        with open(binary_folder + "feature_type_sel.bin", "wb") as fp:
            pickle.dump(feature_type_sel, fp)
        with open(binary_folder + "feature_coord_sel.bin", "wb") as fp:
            pickle.dump(feature_coord_sel, fp)
        with open(binary_folder + "selected_img_names.bin", "wb") as fp:
            pickle.dump(selected_img_names, fp)

    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname in selected_img_names:
            continue
        elif fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            sobel_img = sobelx_ridge_img(img)
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
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
            strong_skel = (skel_img != 0).astype("uint8") * 255
            save_img(cv2.add(raw_img, strong_skel), results_folder + "raw_skel/" + fname)
            
            # 
            # FIND SKELETON POINTS IN RIDGE IMAGE
            #
            skel_positions = find_positions(skel_img)
            skel_patches = patches_from_positions(sobel_img, 
                                                  skel_positions,
                                                  (25, 25),
                                                  0)

            # CLASSIFY
            X = np.array([extract_haar_features(patch, feature_type_sel, feature_coord_sel) for patch in skel_patches])
            y_pred = clf.predict(X)

            for i in range(len(skel_positions)):
                if y_pred[i] == 1:
                    color_img = cv2.circle(color_img, skel_positions[i][::-1], 2, (0, 0, 255), -1)
            #        show_img(skel_patches[i], "Skel patch")
            #show_img(color_img, "Candidates")
            save_img(color_img, results_folder + "classifier/" + fname)

            #color_img = cv2.circle(color_img, point[::-1], 2, (0, 0, 255), -1)
            #m_slic, m_slic_boundaries = slic_superpixels(img, one_component_img) 
            #save_img(skimage.img_as_ubyte(m_slic_boundaries), results_folder + "mslic/" + fname)
            
            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
