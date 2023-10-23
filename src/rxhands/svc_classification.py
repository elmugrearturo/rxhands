import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
from rxhands.ridge import normalized_ridge_img, sobelx_ridge_img

import skimage

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report 

# Points of interest
poi = [1, 2, 5, 6, 7, 10, 11, 12, 15, 16, 17, 20, 21, 22]


def load_data_points(file_path):
    # Load data points from a TPS file
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


def correct_point(original_img, point):
    # Points in the TPS files are in 
    # (X,inv(Y)) format
    # Set them in (i, j)
    j, i = point
    height, width = original_img.shape
    i = height-i
    point = (i, j)
    return point


def build_training_set(img_folder, point_file_path, no_images, patch_size):
    img_point_dict = load_data_points(point_file_path)
    print("Loaded manual tags from: " + point_file_path)
    selected = []
    dataset = []
    y = []
    current_imgs = 0
    for fname in os.listdir(img_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            print("\nSelecting: ", fname)
            raw_img = load_gray_img(img_folder + fname)
            print("Preprocessing: ", fname)
            img = preprocess_image(raw_img)
            img = sobelx_ridge_img(img)
            print("Adding points: ")
            for i, point in enumerate(img_point_dict[fname]): 
                # Correct point
                point = correct_point(img, point)
                patch_i = cut_patch(img, point, patch_size, 0)
                assert patch_i.shape == patch_size
                selected.append(fname)
                dataset.append(patch_i)
                if i in poi:
                    print(f"\t{i}: 1")
                    y.append(1)
                else:
                    print(f"\t{i}: 0")
                    y.append(0)
                #show_img(patch_i, "Parche %d" % i)
            current_imgs += 1
        if current_imgs == no_images:
            break
    return np.array(dataset), np.array(y), selected


def create_classifier(img_folder="./data/", point_file_path="./data/points/T2.TPS", no_images=20, patch_size=(51, 51), train_with_all=False):
    # Load dataset
    dataset, y, selected = build_training_set(img_folder,
                                              point_file_path,
                                              no_images,
                                              patch_size)

    print("\nFlattening dataset...")
    X = np.array([img.flatten() for img in dataset])
    
    if not train_with_all:
        # Use 50% to train
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5,
                                                            random_state=0,
                                                            stratify=y)
    else:
        X_train = X
        y_train = y

    clf = SVC()
    clf.fit(X_train, y_train)

    if not train_with_all:
        y_pred = clf.predict(X_test)

        print(
                f"Classification report for classifier {clf}:\n"
                f"{classification_report(y_test, y_pred)}\n"
                )

    return clf, selected

def main(data_folder="./data/", results_folder="./results/", binary_folder="./bin/"):
    #try:
    #    with open(binary_folder + "clf.bin", "rb") as fp:
    #        clf = pickle.load(fp)
    #    with open(binary_folder + "feature_type_sel.bin", "rb") as fp:
    #        feature_type_sel = pickle.load(fp)
    #    with open(binary_folder + "feature_coord_sel.bin", "rb") as fp:
    #        feature_coord_sel = pickle.load(fp)
    #    with open(binary_folder + "selected_img_names.bin", "rb") as fp:
    #        selected_img_names = pickle.load(fp)
    #except:
    #    clf, feature_type_sel, feature_coord_sel, selected_img_names = \
    #            create_classifier()
    #    with open(binary_folder + "clf.bin", "wb") as fp:
    #        pickle.dump(clf, fp)
    #    with open(binary_folder + "feature_type_sel.bin", "wb") as fp:
    #        pickle.dump(feature_type_sel, fp)
    #    with open(binary_folder + "feature_coord_sel.bin", "wb") as fp:
    #        pickle.dump(feature_coord_sel, fp)
    #    with open(binary_folder + "selected_img_names.bin", "wb") as fp:
    #        pickle.dump(selected_img_names, fp)
    clf, selected = create_classifier()    
    import ipdb;ipdb.set_trace()
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            pass

    kernel = np.ones((5,5), np.uint8)

if __name__ == "__main__":
    main()
