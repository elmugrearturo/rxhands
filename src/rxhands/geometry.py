import os
import cv2
import numpy as np
import multiprocessing as mp

from collections import OrderedDict

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
from rxhands.segmentation import four_region_segmentation

import skimage


def rotate_img(img, angle, rotation_point=None):
    ''' rotation_point (i, j) '''
    height, width = img.shape
    rotation_matrix = get_rotation_matrix(img, angle, rotation_point)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    return rotated_img

def get_rotation_matrix(img, angle, rotation_point=None):
    height, width = img.shape
    if rotation_point != None:
        assert len(rotation_point) == 2
    else:
        rotation_point = (int(height/2), int(width/2)) # (i,j)
    rotation_matrix = cv2.getRotationMatrix2D(rotation_point[::-1], angle, 1.0)
    return rotation_matrix

def transform_positions(positions, rotation_matrix):
    new_positions = []
    for position in positions:
        modif_position = np.ones(3)
        # Invert (i,j) to (x,y,1)
        modif_position[1] = position[0]
        modif_position[0] = position[1]
        new_position = np.matmul(rotation_matrix, modif_position).astype("uint32")
        new_positions.append(new_position[::-1])
    return new_positions

def centroid(points):
    new_i = 0
    new_j = 0
    for k in range(len(points)):
        point = points[k]
        new_i += point[0]
        new_j += point[1]
    new_i /= len(points)
    new_j /= len(points)
    return (int(new_i), int(new_j))


def middle_point(point_one, point_two):
    new_i = int((point_one[0] + point_two[0]) / 2.)
    new_j = int((point_one[1] + point_two[1]) / 2.)
    return (new_i, new_j)


def skeletonize(one_component_img, skeleton_value=127):
    #
    # SKELETONIZE IN LOCAL OTSU ONE COMPONENT
    # 
    skel_img = skimage.morphology.skeletonize(one_component_img == 255)

    skel_img = skel_img.astype("uint8")*skeleton_value
    #save_img(one_component_local_otsu - skel_img, results_folder + "skel/" + fname)

    return skel_img

def euclidean_distance(p, q):
    assert len(p) == len(q)
    total = 0
    for i in range(len(p)):
        total += np.power(q[i] - p[i], 2)
    return np.sqrt(total)

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


def distance_to_border(img, center, kernel):
    # position i, j
    # kernel must be 3x3 with
    # only 1 direction marked

    assert kernel.shape[0] == 3 and kernel.shape[1] == 3
    assert np.sum(kernel) == 1
    assert kernel[1, 1] == 0

    i, j = center
    values = cut_patch(img, center, (3, 3))
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
        new_center = (new_i, new_j) 
        return 1 + distance_to_border(img, new_center, kernel)


def position_to_border_distances(bin_img, position, directions=range(9)):
    # Single process
    # 0 1 2
    # 3 - 5
    # 6 7 8
    assert 0 < len(directions) < 9
    distances = np.ones(9, dtype="int32") * -1
    for i in directions:
        if i == 4:
            continue
        elif 0 <= i <= 8:
            kernel = np.zeros(9, dtype=np.uint32)
            kernel[i] = 1
            kernel = kernel.reshape((3, 3))
            distances[i] = 1 + distance_to_border(bin_img, position, kernel)
        else:
            raise ValueError("Direction not defined")
    return distances

def pixel_to_border_distances(bin_img, position, distances):
    # Multiprocess
    # distances is mp.Array

    for i in range(9):
        if i == 4:
            continue
        kernel = np.zeros(9, dtype=np.uint32)
        kernel[i] = 1
        kernel = kernel.reshape((3, 3))
        dist = distance_to_border(bin_img, position, kernel)
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
    labeled_skel_img[labeled_skel_img == 0] = "0"
    all_labels = np.unique(labeled_skel_img)
    relevant_labels = [l for l in all_labels if l != "0"]
    final_labeled_skel_img = np.zeros(skel_img.shape, dtype="uint8")
    for i, label in enumerate(relevant_labels):
        final_labeled_skel_img[labeled_skel_img == label] = 255-i
    return final_labeled_skel_img

def prune_skeleton(skel_img, max_no_branches=5):
    assert len(np.unique(skel_img)) == 2
    assert np.unique(skel_img)[1] == 1
    # Skeleton to (i, j) positions
    skel_positions = find_positions(skel_img)
    # Find the lowest position in i
    skel_positions.sort(reverse=True)
    lowest_skel_positions = [point for point in skel_positions if point[0] == skel_positions[0][0]]
    # Find the lowest position in j
    lowest_skel_positions.sort()
    initial_position = lowest_skel_positions[0]
    # Label skeleton branches starting in lowest position (i, j)
    labeled_skel_img = label_skel_branches(skel_img, initial_position)
    # Check length of resulting branches
    unique_labels = np.unique(labeled_skel_img, return_counts=True)
    zipped_labels = [l for l in zip(unique_labels[0], unique_labels[1])]
    zipped_labels.sort(key=lambda x: x[1], reverse=True)
    # Only use the max_no_branches longest branches
    for label, counts in zipped_labels[max_no_branches + 1 : ]:
        labeled_skel_img[labeled_skel_img == label] = 0
    return labeled_skel_img

def check_gaps_per_row(bin_img, rows=None):
    height, width = bin_img.shape
    if rows == None:
        rows = range(height)

    all_filled = []
    all_gaps = []
    for i in rows:
        last_position = (i, 0)
        final_position = (i, width-1)
        filled = []
        gaps = []

        if bin_img[last_position] != 0:
            is_filled = True
        else:
            is_filled = False

        for j in range(1, width):
            current_position = (i, j)
            if is_filled:
                if bin_img[current_position] != 0:
                    continue
                else:
                    filled.append((last_position, (i, j-1)))
                    last_position = current_position
                    is_filled = False
            else:
                if bin_img[current_position] == 0:
                    continue
                else:
                    gaps.append((last_position, (i, j-1)))
                    last_position = current_position
                    is_filled = True

        if last_position != final_position:
            assert last_position[1] < final_position[1]
            if is_filled:
                filled.append((last_position, final_position))
            else:
                gaps.append((last_position, final_position))
        all_filled.append(filled)
        all_gaps.append(gaps)
    return all_filled, all_gaps

def check_gaps_per_row_alt(bin_img, rows=None):
    height, width = bin_img.shape
    if rows == None:
        rows = range(height)
    filled_positions = [p for p in zip(*np.where(bin_img != 0))]

    intervals = []
    for i in rows:
        last_position = (i, 0)
        final_position = (i, width-1)
        filled = []
        gaps = []

        if last_position in filled_positions:
            is_filled = True
            row.pop(0)
        else:
            is_filled = False

        for j in range(1, width):
            current_position = (i, j)
            if is_filled:
                if current_position in filled_positions:
                    continue
                else:
                    filled.append((last_position, (i, j-1)))
                    last_position = current_position
                    is_filled = False
            else:
                if current_position not in filled_positions:
                    continue
                else:
                    gaps.append((last_position, (i, j-1)))
                    last_position = current_position
                    is_filled = True

        if last_position != final_position:
            assert last_position[1] < final_position[1]
            if is_filled:
                filled.append((last_position, final_position))
            else:
                gaps.append((last_position, final_position))
        intervals.append((filled, gaps))
    return intervals

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
            strong_skel_img = (skel_img != 0).astype("uint8") * 255
            save_img(one_component_img - strong_skel_img, results_folder + "skel/" + fname)
            
            # Find position of skeleton pixels
            #skel_positions = find_positions(skel_img)
            #print("\tCalculated skeleton pixel positions")

            # Prune skeleton
            print("\tPrunning skeleton")
            labeled_skel_img = prune_skeleton(skel_img)
            #colored_skel_img = cv2.applyColorMap(labeled_skel_img, cv2.COLORMAP_JET)
            #show_img(colored_skel_img, "labeled img")
            save_img(labeled_skel_img, results_folder + "prune_skel/" + fname)
            ## Find distance to border of each skeleton pixel
            #distances = skel_to_border_distances(one_component_img, skel_positions, 8)
            #print("\tCalculated skeleton to border distances")

            #d_max, d_min, d_std, d_mean, d_median = distances_stats(distances)
            ## Find center of possible palm
            #center, distance = d_min[0]
            #dic_mean = OrderedDict(d_mean)
            #dic_median = OrderedDict(d_median)
            #print("\tCenter: ", center)
            #print("\tDistance: ", distance)
            #print("\tMean radius: ", dic_mean[center])
            #print("\tMedian radius: ", dic_median[center])
            #no_palm =  cv2.circle(one_component_img, center[::-1], 
            #                      int(distance),
            #                      #int(min(dic_mean[center], dic_median[center])), 
            #                      (0), -1)
            #save_img(no_palm - skel_img, results_folder + "no_palm/" + fname)
            #import ipdb; ipdb.set_trace()

        #break

if __name__ == "__main__" :
    main()

