import gzip

import os
import shutil

import numpy as np
import pandas as pd

import random

import cv2
import skimage

import tensorflow as tf

from rxhands.auxiliary import show_img, save_img, load_gray_img
from rxhands.preprocessing import preprocess_image

import math

poi = [i for i in range(18, 37)]
   
def normalized_ridge_img(img, black_ridges=False):
    ridge_img = skimage.filters.sato(img, sigmas=range(1, 5), black_ridges=black_ridges)
    ridge_img[ridge_img < 0 ] = 0.
    ridge_norm = cv2.normalize(ridge_img, None, 0, 255, cv2.NORM_MINMAX)
    return np.round(ridge_norm).astype("uint8")

def non_normalized_gaussian_heatmap(xL, yL, H, W, sigma=5):

    channel = [math.exp(-((c - xL) ** 2 + (r - yL) ** 2) / (2 * sigma ** 2)) for r in range(H) for c in range(W)]
    channel = np.array(channel, dtype=np.float32)
    channel = np.reshape(channel, newshape=(H, W))

    return channel

def load_numpy_tensor(npy_path):
    with gzip.GzipFile(npy_path, "r") as fp:
        uncompressed_npy = np.load(fp)
    return tf.convert_to_tensor(uncompressed_npy, dtype=tf.float32)

def load_img_tensor(img_path):
    decoded_img = tf.image.decode_image(tf.io.read_file(img_path), dtype=tf.float32)
    return decoded_img

def get_tfdataset(datasets_folder):
    datasets = []
    for ds_type in ["train", "test"]:
        X_images = tf.data.Dataset.list_files(os.path.join(datasets_folder,
                                                           ds_type,
                                                           "predictors",
                                                           "*.png"), shuffle=False)
        X_images = X_images.map(load_img_tensor)
        Y_tensors = tf.data.Dataset.list_files(os.path.join(datasets_folder,
                                                            ds_type,
                                                            "to_predict",
                                                            "*.npy.gz"), shuffle=False)
        Y_tensors = Y_tensors.map(lambda item: tf.numpy_function(load_numpy_tensor, 
                                                                 [item],
                                                                 (tf.float32)))
        ds = tf.data.Dataset.zip((X_images, Y_tensors))
        datasets.append(ds)
    
    return datasets[0], datasets[1]

def show_multi_heatmap(multi_heatmap):
    single_heatmap = multi_heatmap.sum(axis=2)
    result = cv2.normalize(single_heatmap, None, 0, 255, cv2.NORM_MINMAX)
    result = result.astype("uint8")
    show_img(result, "multi-heatmap")

def gaussian_mask_dataset(dhi_files="./dhi/", dataset_path="./training/heatmap/", size=(256, 256), preprocess_img=False, ridge_img=False):
    '''Generates training dataset (heatmap)'''
    
    hand_data_points = pd.read_csv(os.path.join(dhi_files, "all.csv"))
    original_information = {}
    index = 0
    
    selected_files = []
    for root, folders, filenames in os.walk(dhi_files):
        if len(filenames) == 0:
            continue
        for filename in filenames:
            if filename.lower().endswith("jpg") or filename.lower().endswith("jpeg"):
                print(filename)
                image_id = int(filename.split(".")[0])

                # Open image file
                img = load_gray_img(os.path.join(root, filename))
                height, width = img.shape
                
                # Transform to ridge img
                if preprocess_img:
                    img = preprocess_image(img)
                if ridge_img:
                    img = normalized_ridge_img(img)

                # Resize
                img = cv2.resize(img, size[::-1])
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                y_ratio = size[0] / height
                x_ratio = size[1] / width
                # Select row
                try:
                    row = hand_data_points[hand_data_points["image_id"] == image_id]
                    assert len(row) > 0
                except:
                    continue
                heatmaps = []
                
                # Label points
                for i in poi:
                    x_i = int(row[f"X_{i}"].values[0] * x_ratio)
                    y_i = int(row[f"Y_{i}"].values[0] * y_ratio)
                    heatmap_i = non_normalized_gaussian_heatmap(x_i, y_i, size[1], size[0])
                    heatmaps.append(heatmap_i)
                    #norm_heatmap_i = cv2.normalize(heatmap_i, None, 0, 255, cv2.NORM_MINMAX)
                    #save_img(norm_heatmap_i, os.path.join(dataset_path, "heatmaps/", f"{image_id}_{i}.png"))
                    #color_img = cv2.circle(color_img, (x_i, y_i), 5, (0, 0, 255), -1)
                    #color_img = cv2.putText(color_img, f"{i}", (x_i, y_i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                #show_img(color_img, "points")
                stacked_heatmaps = np.dstack(heatmaps)
                
                # Register data for csv
                original_information[index] = {'id' : image_id,
                                               'height' : height,
                                               'width' : width} 
                
                # Register files for partition
                selected_files.append(image_id)

                # Temporary save
                with gzip.GzipFile(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"), "w") as fp:
                    np.save(fp, stacked_heatmaps)
                save_img(img, os.path.join(dataset_path, "all/", f"{image_id}.png"))
                index += 1

    original_information_df = pd.DataFrame(original_information).T
    original_information_df.to_csv(os.path.join(dataset_path, "original_information.csv"), sep=",")

    # Make partition
    no_train_examples = int(len(selected_files)*0.8)

    random.shuffle(selected_files)

    for image_id in selected_files[:no_train_examples]:
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.png"),
                    os.path.join(dataset_path, "train/", "predictors/", f"{image_id}.png"))
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"),
                    os.path.join(dataset_path, "train/" , "to_predict/" , f"{image_id}.npy.gz"))
    
    for image_id in selected_files[no_train_examples:]:
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.png"),
                    os.path.join(dataset_path, "test/", "predictors/", f"{image_id}.png"))
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"),
                    os.path.join(dataset_path, "test/" , "to_predict/" , f"{image_id}.npy.gz"))

def multi_mask_dataset(dhi_files="./dhi/", dataset_path="./training/multi/", size=(256, 256), preprocess_img=False, ridge_img=False):
    '''Generates training dataset (multi bin_mask)'''
    
    hand_data_points = pd.read_csv(os.path.join(dhi_files, "all.csv"))
    original_information = {}
    index = 0
    
    selected_files = []
    for root, folders, filenames in os.walk(dhi_files):
        if len(filenames) == 0:
            continue
        for filename in filenames:
            if filename.lower().endswith("jpg") or filename.lower().endswith("jpeg"):
                print(filename)
                image_id = int(filename.split(".")[0])

                # Open image file
                img = load_gray_img(os.path.join(root, filename))
                height, width = img.shape
                
                if preprocess_img:
                    img = preprocess_image(img)

                # Transform to ridge img
                if ridge_img:
                    img = normalized_ridge_img(img)
                # Resize
                img = cv2.resize(img, size[::-1])
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


                y_ratio = size[0] / height
                x_ratio = size[1] / width
                # Select row
                try:
                    row = hand_data_points[hand_data_points["image_id"] == image_id]
                    assert len(row) > 0
                except:
                    continue
                
                masks = []
                
                # Label points
                for i in poi:
                    x_i = int(row[f"X_{i}"].values[0] * x_ratio)
                    y_i = int(row[f"Y_{i}"].values[0] * y_ratio)
                    mask_i = np.zeros(img.shape)
                    mask_i = cv2.circle(mask_i, (x_i, y_i), 2, (1.), -1)
                    #show_img(mask_i.astype("uint8")*255, "mask")
                    masks.append(mask_i)

                stacked_masks = np.dstack(masks)

                # Register data for csv
                original_information[index] = {'id' : image_id,
                                               'height' : height,
                                               'width' : width} 
                
                # Register files for partition
                selected_files.append(image_id)

                # Temporary save
                with gzip.GzipFile(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"), "w") as fp:
                    np.save(fp, stacked_masks)
                save_img(img, os.path.join(dataset_path, "all/", f"{image_id}.png"))
                index += 1

    original_information_df = pd.DataFrame(original_information).T
    original_information_df.to_csv(os.path.join(dataset_path, "original_information.csv"), sep=",")

    # Make partition
    no_train_examples = int(len(selected_files)*0.8)

    random.shuffle(selected_files)

    for image_id in selected_files[:no_train_examples]:
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.png"),
                    os.path.join(dataset_path, "train/", "predictors/", f"{image_id}.png"))
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"),
                    os.path.join(dataset_path, "train/" , "to_predict/" , f"{image_id}.npy.gz"))
    
    for image_id in selected_files[no_train_examples:]:
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.png"),
                    os.path.join(dataset_path, "test/", "predictors/", f"{image_id}.png"))
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"),
                    os.path.join(dataset_path, "test/" , "to_predict/" , f"{image_id}.npy.gz"))


def single_mask_dataset(dhi_files="./dhi/", dataset_path="./training/single/", size=(256, 256), preprocess_img=False, ridge_img=False):
    '''Generates training dataset (bin_mask)'''
    
    hand_data_points = pd.read_csv(os.path.join(dhi_files, "all.csv"))
    original_information = {}
    index = 0
    
    selected_files = []
    for root, folders, filenames in os.walk(dhi_files):
        if len(filenames) == 0:
            continue
        for filename in filenames:
            if filename.lower().endswith("jpg") or filename.lower().endswith("jpeg"):
                print(filename)
                image_id = int(filename.split(".")[0])

                # Open image file
                img = load_gray_img(os.path.join(root, filename))
                height, width = img.shape
                
                if preprocess_img:
                    img = preprocess_image(img)

                # Transform to ridge img
                if ridge_img:
                    img = normalized_ridge_img(img)
                # Resize
                img = cv2.resize(img, size[::-1])
                color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


                y_ratio = size[0] / height
                x_ratio = size[1] / width
                # Select row
                try:
                    row = hand_data_points[hand_data_points["image_id"] == image_id]
                    assert len(row) > 0
                except:
                    continue
                
                masks = []
                
                # Label points
                for i in poi:
                    x_i = int(row[f"X_{i}"].values[0] * x_ratio)
                    y_i = int(row[f"Y_{i}"].values[0] * y_ratio)
                    mask_i = np.zeros(img.shape)
                    mask_i = cv2.circle(mask_i, (x_i, y_i), 2, (1.), -1)
                    #show_img(mask_i.astype("uint8")*255, "mask")
                    masks.append(mask_i)

                stacked_masks = np.dstack(masks)
                multi_mask = np.sum(stacked_masks, axis=2)
                #show_img(multi_mask.astype("uint8")*255, "mask")

                # Register data for csv
                original_information[index] = {'id' : image_id,
                                               'height' : height,
                                               'width' : width} 
                
                # Register files for partition
                selected_files.append(image_id)

                # Temporary save
                with gzip.GzipFile(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"), "w") as fp:
                    np.save(fp, multi_mask)
                save_img(img, os.path.join(dataset_path, "all/", f"{image_id}.png"))
                index += 1

    original_information_df = pd.DataFrame(original_information).T
    original_information_df.to_csv(os.path.join(dataset_path, "original_information.csv"), sep=",")

    # Make partition
    no_train_examples = int(len(selected_files)*0.8)

    random.shuffle(selected_files)

    for image_id in selected_files[:no_train_examples]:
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.png"),
                    os.path.join(dataset_path, "train/", "predictors/", f"{image_id}.png"))
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"),
                    os.path.join(dataset_path, "train/" , "to_predict/" , f"{image_id}.npy.gz"))
    
    for image_id in selected_files[no_train_examples:]:
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.png"),
                    os.path.join(dataset_path, "test/", "predictors/", f"{image_id}.png"))
        shutil.move(os.path.join(dataset_path, "all/", f"{image_id}.npy.gz"),
                    os.path.join(dataset_path, "test/" , "to_predict/" , f"{image_id}.npy.gz"))


if __name__ == "__main__" :
    #multi_mask_dataset(preprocess_img=True, ridge_img=False)
    #single_mask_dataset(preprocess_img=True, ridge_img=False)
    gaussian_mask_dataset(preprocess_img=True, ridge_img=False)

#def main():
#    heatmap = gaussian(48, 48, 96, 96)
#    heatmap2 = gaussian(48, 300, 800, 480)
#    plt.figure(figsize=(30,30))
#
#    #plt.subplot(1,4,1)
#    plt.imshow(heatmap2)
#    plt.title("Single Heatmap")
#    plt.show()
#    import ipdb;ipdb.set_trace()
