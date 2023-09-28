import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image

import skimage

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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


def extract_haar_features(img, feature_type=None, feature_coord=None):
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

    # Calculate Haar-like features to build X
    X = np.array([extract_haar_features(img) for img in dataset])

    # Use 50% to train
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5,
                                                        random_state=0,
                                                        stratify=y)
    # Extract all possible features
    feature_coord, feature_type = \
        skimage.feature.haar_like_feature_coord(width=dataset.shape[2], 
                                                height=dataset.shape[1])

    # Classifier
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                 max_features=100, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    auc_full_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    # Sort features in order of importance
    idx_sorted = np.argsort(clf.feature_importances_)[::-1]

    print(f'Full for an AUC of {auc_full_features:.2f}. ')
    
    # Extract only the most important features (80% cumulative value)
    cdf_feature_importances = np.cumsum(clf.feature_importances_[idx_sorted])
    cdf_feature_importances /= cdf_feature_importances[-1]  # divide by max value
    sig_feature_count = np.count_nonzero(cdf_feature_importances < 0.8)
    sig_feature_percent = round(sig_feature_count /
                                len(cdf_feature_importances) * 100, 1)
    print(f'{sig_feature_count} features, or {sig_feature_percent}%, '
           f'account for 80% of branch points in the random forest.')

    # Select the determined number of most informative features
    feature_coord_sel = feature_coord[idx_sorted[:sig_feature_count]]
    feature_type_sel = feature_type[idx_sorted[:sig_feature_count]]

    # Retrain only with selected features
    # Recalculate Haar-like features to build X
    X = np.array([extract_haar_features(img, feature_type_sel, feature_coord_sel) for img in dataset])

    # Use 50% to train
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.5,
                                                        random_state=0,
                                                        stratify=y)
    # New classifier
    clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                                 max_features=100, n_jobs=-1, random_state=0)
    clf.fit(X_train, y_train)
    auc_subs_features = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    print(f'Restricted for an AUC of {auc_full_features:.2f}. ')
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
