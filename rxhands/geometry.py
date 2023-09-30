import os
import cv2
import numpy as np
import multiprocessing as mp

from collections import OrderedDict

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
from rxhands.segmentation import four_region_segmentation
from rxhands.superpixels import skeletonize

import skimage

def distances_array_to_list(distances):
    # Convert from shared memory obj
    for k in distances.keys():
       arr = distances[k]
       distances[k] = []
       for i in range(9):
           if i == 4:
               continue
           distances[k].append(arr[i])

def distances_stats(distances):
    distances_max = []
    distances_min = [] 
    distances_std = []
    distances_mean = [] 
    distances_median = [] 
    for k in distances.keys():
        dists = distances[k]
        distances_max.append((k, max(dists)))
        distances_min.append((k, min(dists)))
        distances_std.append((k, np.std(dists)))
        distances_mean.append((k, np.mean(dists)))
        distances_median.append((k, np.median(dists)))
    
    distances_max.sort(key=lambda x:x[1], reverse=True)
    distances_min.sort(key=lambda x:x[1], reverse=True)
    distances_std.sort(key=lambda x:x[1], reverse=True)
    distances_mean.sort(key=lambda x:x[1], reverse=True)
    distances_median.sort(key=lambda x:x[1], reverse=True)

    #distances_max = OrderedDict(distances_max)
    #distances_min = OrderedDict(distances_min)
    #distances_std = OrderedDict(distances_std)
    #distances_mean = OrderedDict(distances_mean)
    #distances_median = OrderedDict(distances_median)
    
    return distances_max, distances_min, distances_std, distances_mean, distances_median


def distance_to_border(img, i, j, kernel):
    # position i, j
    # kernel must be 3x3 with
    # only 1 direction marked

    assert kernel.shape[0] == 3 and kernel.shape[1] == 3
    assert np.sum(kernel) == 1
    assert kernel[1, 1] == 0
    
    values = cut_patch(img, i, j, (3, 3))
    masked_values = values * kernel
    total = np.sum(masked_values)
    if total == 0:
        # We are in a border pixel
        return 0
    else:
        # We are not in a border pixel, move
        k_position = np.argmax(kernel)
        # Get next step
        if k_position == 0:
            new_i = i - 1
            new_j = j - 1
        elif k_position == 1:
            new_i = i - 1
            new_j = j
        elif k_position == 2:
            new_i = i - 1
            new_j = j + 1
        elif k_position == 3:
            new_i = i
            new_j = j - 1
        elif k_position == 5:
            new_i = i
            new_j = j + 1
        elif k_position == 6:
            new_i = i + 1
            new_j = j - 1
        elif k_position == 7:
            new_i = i + 1
            new_j = j
        elif k_position == 8:
            new_i = i + 1
            new_j = j + 1
        
        return 1 + distance_to_border(img, new_i, new_j, kernel)


def pixel_to_border_distances(bin_img, position, distances):
    # distances is mp.Array

    for i in range(9):
        if i == 4:
            continue
        kernel = np.zeros(9, dtype=np.uint32)
        kernel[i] = 1
        kernel = kernel.reshape((3, 3))
        dist = distance_to_border(bin_img, position[0], position[1], kernel)
        distances[i] = dist


def skel_to_border_distances(bin_img, skel_positions, parallelization_factor=1):
    assert parallelization_factor >= 1
    distances = {}
    batch = mp.cpu_count()
    all_positions = skel_positions.copy()
    while len(all_positions) > 0 :
        procs = []
        for i in range(batch * parallelization_factor):
            try:
                position = all_positions.pop(0)
            except:
                continue
            distances[position] = mp.Array("i", 9)
            proc = mp.Process(target=pixel_to_border_distances, args=(bin_img, position, distances[position]))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    distances_array_to_list(distances)
    return distances

def recursive_skel_labelling(skel_img, label_img, init_position, label="1"):
    # 0 1 2
    # 3 4 5
    # 6 7 8
    assert isinstance(label, str)
    if skel_img[init_position] == 0:
        return
    
    eight_neighbors = np.ones((3, 3), dtype="uint8")
    eight_neighbors[1, 1] = 0
    
    height, width = skel_img.shape
    current_position = init_position

    while 0 <= current_position[0] < height and 0 <= current_position[1] < width :
        # Assign label
        label_img[current_position] = label
        # Remove from skeleton
        skel_img[current_position] = 0
        # Find the path(s)
        patch = cut_patch(skel_img, current_position, (3, 3), 0)
        conv_patch = (eight_neighbors * patch).flatten()
        no_paths = np.sum(conv_patch)
        if no_paths == 0:
            return
        elif no_paths == 1:
            # Just continue following the path without changing label
            for p in range(9):
                if conv_patch[p] != 0:
                    if p in [0, 1, 2]:
                        next_i = current_position[0] - 1
                    elif p in [3, 5]:
                        next_i = current_position[0]
                    elif p in [6, 7, 8]:
                        next_i = current_position[0] + 1
                    
                    if p in [0, 3, 6]:
                        next_j = current_position[1] - 1
                    elif p in [1, 7]:
                        next_j = current_position[1]
                    elif p in [2, 5, 8]:
                        next_j = current_position[1] + 1
                            
                    current_position = (next_i, next_j)
                    break
        else:
            # Change label for every path
            for p in range(9):
                if conv_patch[p] != 0:
                    if p in [0, 1, 2]:
                        next_i = current_position[0] - 1
                    elif p in [3, 5]:
                        next_i = current_position[0]
                    elif p in [6, 7, 8]:
                        next_i = current_position[0] + 1
                    
                    if p in [0, 3, 6]:
                        next_j = current_position[1] - 1
                    elif p in [1, 7]:
                        next_j = current_position[1]
                    elif p in [2, 5, 8]:
                        next_j = current_position[1] + 1
                            
                    next_position = (next_i, next_j)
                    next_label = label + str(p)
                    recursive_skel_labelling(skel_img, label_img, next_position, next_label)
            return

def label_skel_branches(skel_img, initial_point):
    labeled_skel_img = skel_img.astype("object")
    recursive_skel_labelling(skel_img.copy(), labeled_skel_img, initial_point)
    return labeled_skel_img

def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            print("\nFile: %s" % fname)
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            try:
                one_component_img = four_region_segmentation(img)
            except Exception as e:
                print("Couldn't find hand: %s" % fname, " ", e)
                continue
            print("\tCalculated one-component img")

            #
            # SKELETONIZE
            #
            skel_img = skeletonize(one_component_img, 1)
            print("\tCalculated skeleton")
            #save_img(one_component_img - skel_img, results_folder + "skel/" + fname)
            
            # Find position of skeleton pixels
            skel_positions = find_positions(skel_img)
            print("\tCalculated skeleton pixel positions")

            # Prune skeleton
            skel_positions.sort(reverse=True)
            lowest_skel_positions = [point for point in skel_positions if point[0] == skel_positions[0][0]]
            lowest_skel_positions.sort()
            initial_position = lowest_skel_positions[0]
            labeled_skel_img = label_skel_branches(skel_img, initial_position)
            show_img(labeled_skel_img, "labeled img")
            # Find distance to border of each skeleton pixel
            distances = skel_to_border_distances(one_component_img, skel_positions, 8)
            print("\tCalculated skeleton to border distances")

            d_max, d_min, d_std, d_mean, d_median = distances_stats(distances)
            # Find center of possible palm
            center, distance = d_min[0]
            dic_mean = OrderedDict(d_mean)
            dic_median = OrderedDict(d_median)
            print("\tCenter: ", center)
            print("\tDistance: ", distance)
            print("\tMean radius: ", dic_mean[center])
            print("\tMedian radius: ", dic_median[center])
            no_palm =  cv2.circle(one_component_img, center[::-1], 
                                  int(distance),
                                  #int(min(dic_mean[center], dic_median[center])), 
                                  (0), -1)
            save_img(no_palm - skel_img, results_folder + "no_palm/" + fname)
            #import ipdb; ipdb.set_trace()

        #break

if __name__ == "__main__" :
    main()

