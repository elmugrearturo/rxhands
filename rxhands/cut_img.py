import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
from rxhands.segmentation import four_region_segmentation

def crop_image(raw_img):
    img = preprocess_image(raw_img)
    try:
        one_component_img = four_region_segmentation(img)
    except Exception as e:
        print("Couldn't find hand: %s" % fname, " ", e)
        return raw_img

    new_raw_img = raw_img * (one_component_img // 255)
    _, _, bboxes, _ = cv2.connectedComponentsWithStats(one_component_img, 8)
    # Cut using the bounding boxes
    # First bounding box (0) is the entire image
    # (1) is the cc of interest
    x, y, b_width, b_height, _ = bboxes[1]
    cropped_img = new_raw_img[y:y+b_height, x:x+b_width]
    return cropped_img

def main(data_folder="./data/", results_folder="./results/", binary_folder="./bin/", dataset="intra_inter"):
    
    data_folder = os.path.join(data_folder, dataset)
    results_folder = os.path.join(results_folder, dataset)
    
    try:
        # Create folder structure
        os.makedirs(os.path.join(results_folder, "cut_dataset/"))
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

            raw_img *= (one_component_img // 255)
            _, _, bboxes, _ = cv2.connectedComponentsWithStats(one_component_img, 8)
            # Cut using the bounding boxes
            # First bounding box (0) is the entire image
            # (1) is the cc of interest
            x, y, b_width, b_height, _ = bboxes[1]
            cut_img = raw_img[y:y+b_height, x:x+b_width]
            save_img(cut_img, os.path.join(results_folder, "cut_dataset/", f"{fname.split('.')[0]}.png"))


if __name__ == "__main__" :
    main(dataset="mex_left")
