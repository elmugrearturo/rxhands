import os
import cv2
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt

import skimage

from rxhands.auxiliary import *
from rxhands.cut_img import crop_image
from rxhands.neural_model import load_multi_mask_model
from rxhands.dataset import show_multi_heatmap
from rxhands.hand_model import Hand
from rxhands.preprocessing import preprocess_image
from rxhands.segmentation import four_region_segmentation
from rxhands.superpixels import skeletonize, slic_superpixels
from rxhands.svc_classification import create_classifier

binary_folder = os.path.join(os.path.dirname(__file__), "bin/")
home_folder = os.path.expanduser("~")
weights_path = os.path.join(home_folder, "rxhands_models", "model-heatmap-rxhands-raw.h5")

# Lookup table for neural algorithm
lookup_table = { 0 : 3,
                 1 : 2,
                 2 : 1,
                 3 : 8,
                 4 : 7,
                 5 : 6,
                 6 : 5,
                 7 : 13,
                 8 : 12,
                 9 : 11,
                 10 : 10,
                 11 : 18,
                 12 : 17,
                 13 : 16,
                 14 : 15,
                 15 : 23,
                 16 : 22,
                 17 : 21,
                 18 : 20
                }

def symbolic(data_folder, results_folder):
    
    try:
        with open(os.path.join(binary_folder, "clf.bin"), "rb") as fp:
            clf = pickle.load(fp)
        with open(os.path.join(binary_folder, "selected_img_names.bin"), "rb") as fp:
            selected_img_names = pickle.load(fp)
    except:
        clf, selected_img_names = create_classifier(train_with_all=True)
        with open(os.path.join(binary_folder, "clf.bin"), "wb") as fp:
            pickle.dump(clf, fp)
        with open(os.path.join(binary_folder, "selected_img_names.bin"), "wb") as fp:
            pickle.dump(selected_img_names, fp)

    try:
        # Create folder structure
        os.makedirs(os.path.join(results_folder, "hand_model/"))
    except:
        pass

    try:
        os.makedirs(os.path.join(results_folder, "poi/"))
    except:
        pass
    
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    dataset_dict = {}
    for fname in os.listdir(data_folder) :
        if fname.endswith(".jpg") or fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(os.path.join(data_folder, fname))
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
            partial_hand_model.find_points_in_finger(clf, "kmeans")
            
            ##
            ## SAVE POINTS
            ##
            
            hand_dict = partial_hand_model.to_dictionary()
            dataset_dict[fname] = hand_dict
            
            ## 
            ## DISPLAY
            ##
            
            print(f"\t\tMarking regions in raw_img...")
            marked_img = partial_hand_model.paint_to_img(raw_img)
            marked_points = partial_hand_model.paint_poi_to_img(raw_img)
            
            save_img(marked_img, os.path.join(results_folder, "hand_model/", fname))
            save_img(marked_points, os.path.join(results_folder, "poi/",  fname))
            
        #break
    df = pd.DataFrame(dataset_dict).T
    df.to_csv(os.path.join(results_folder, "datapoints.csv"), sep=",", index=True)

def neural(data_folder,
           results_folder,
           crop_img=False,
           preprocess_img=False): 
    try:
        os.makedirs(results_folder)
    except:
        pass

    # Hard coded shape and model name
    shape=(256, 256, 1)

    model = load_multi_mask_model(weights_path, shape, 19)

    point_dataset = {}
    img_no = 0
    for fname in os.listdir(data_folder):
        
        # Initialize record
        img_no += 1
        point_dataset[img_no] = {}
        point_dataset[img_no]["image_id"] = fname.split(".")[0]

        # Get image
        img = load_gray_img(os.path.join(data_folder,
                                         fname))
        if crop_img:
            img = crop_image(img)

        if preprocess_img:
            img = preprocess_image(img)
        
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        height, width = img.shape

        # Calculate ratio
        y_ratio = shape[0]/height
        x_ratio = shape[0]/width

        # Resize
        img = cv2.resize(img, shape[:2][::-1])
        #show_img(img)

        # Normalize
        img = img.astype("float32")
        img /= 255

        # Predict
        predicted_masks = model.predict(np.array([img]))[0]
        #plt.figure(figsize=(15, 15))
        for i in range(19):
            #plt.subplot(4, 5, i+1)
            mask_i = predicted_masks[:,:,i]
            #plt.imshow(mask_i)
            y, x = np.unravel_index(np.argmax(mask_i, axis=None), mask_i.shape)
            
            # Project point location
            y /= y_ratio
            x /= x_ratio
            
            y = int(y)
            x = int(x)
            
            # Put in output image
            color_img = cv2.circle(color_img, (x,y), 4, (0, 0 ,255), -1)
            color_img = cv2.putText(color_img, 
                                    str(lookup_table[i]),
                                    (x, y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    .5,
                                    (0, 255, 0),
                                    1)
            # Save to dict
            point_dataset[img_no][str(i)+"_X"] = x
            point_dataset[img_no][str(i)+"_Y"] = height - y
        
        #plt.show()
        #show_img(color_img, "color")
        #import ipdb;ipdb.set_trace()
        save_img(color_img, os.path.join(results_folder, f"{fname}"))
    
    df = pd.DataFrame(point_dataset).T
    df.to_csv(os.path.join(results_folder, 
                           "point_positions.csv"),
              sep=",")
