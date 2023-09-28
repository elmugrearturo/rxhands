import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image

import skimage

# Points of interest
poi = [1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22]


def load_data_points(file_path):
    img_point_dict = {}
    with open(file_path, "r") as fp:
        point_count = 0
        point_list = []
        for line in fp.readlines():
            line = line.replace("\n", "")
            if "=" in line:
                if "LM" in line:
                    _, current_count = line.split("=")
                    point_count = int(current_count)
                elif "IMAGE" in line:
                    _, name = line.split("=")
                    assert len(point_list) == point_count
                    img_point_dict[name] = point_list
                    point_count = 0
                    point_list = []
            else:
                x, y = line.split(" ")
                point_list.append((int(float(x)), int(float(y))))
    return img_point_dict


def extract_haar_features(img, feature_type, feature_coord=None):
    integral_img = skimage.transform.integral_image(img)
    features = skimage.feature.haar_like_feature(integral_img, 0, 0, 
                                                 integral_img.shape[0],
                                                 integral_img.shape[1],
                                                 feature_type=feature_type,
                                                 feature_coord=feature_coord)
    return features


def correct_point(original_img, point):
    # Points in the TPS files are in 
    # (X,inv(Y)) format
    # Set them in (i, j)
    j, i = point
    height, width = original_img.shape
    i = height-i
    point = (i, j)
    return point

def build_training_set(data_folder="./data/", no_images=20):
    img_point_dict = load_data_points(data_folder + "points/T2.TPS")
    selected = []
    dataset = []
    y = []
    current_imgs = 0
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            for i, point in enumerate(img_point_dict[fname]): 
                # Correct point
                point = correct_point(img, point)
                try:
                    patch_i = get_patch(img, point, 25)
                except:
                    import ipdb;ipdb.set_trace()
                selected.append(fname)
                dataset.append(patch_i)
                if i in poi:
                    y.append(1)
                else:
                    y.append(0)
            current_imgs += 1
        if current_imgs == no_images:
            break
    return np.array(dataset), np.array(y), selected

def main(data_folder="./data/", results_folder="./results/"):
    # Load dataset
    dataset, y, selected = build_training_set()
    import ipdb;ipdb.set_trace()
    # Read previous points
    img_point_dict = load_data_points(data_folder + "points/T2.TPS")
    for fname in os.listdir(data_folder):
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 

            points = img_point_dict[fname]
            for i, point in enumerate(points):
                # Correct Y (source labeling was shady af, fucking useless ppl)
                point = correct_point(img, point)

                color_img = cv2.circle(color_img, point[::-1], 2, (0, 0, 255), -1)
                color_img = cv2.putText(color_img, str(i), point[::-1], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                if i in poi:
                    patch_i = get_patch(img, point, 25)
                    show_img(patch_i, "Parche %d" % i)

            show_img(color_img, "Puntos")
            import ipdb;ipdb.set_trace()
        break


if __name__ == "__main__" :
    main()
