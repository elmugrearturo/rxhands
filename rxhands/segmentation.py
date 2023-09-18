import os
import cv2
import numpy as np

from rxhands.auxiliary import *
from rxhands.preprocessing import preprocess_image
import skimage

def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            raw_img = load_gray_img(data_folder + fname)
            img = preprocess_image(raw_img)
            #img = cv2.equalizeHist(raw_img)
            img_color = load_color_img(data_folder + fname)
            raw_blur = cv2.medianBlur(raw_img, 7)
            blur = cv2.medianBlur(img, 7)
            #norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

            # 
            # SOBEL FILTER
            #
            sobelxy = cv2.Sobel(blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

            #
            # ADAPTIVE THRESHOLD
            #
            adaptive_thr_img = cv2.adaptiveThreshold(blur, 255, 
                                            cv2.ADAPTIVE_THRESH_MEAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
            # Remove small objects
            no_small_obj = skimage.morphology.remove_small_objects(adaptive_thr_img==255, 32)
            no_small_holes = skimage.morphology.remove_small_holes(no_small_obj, 500)
            clean_adaptive_thr_img = no_small_holes.astype("uint8") * 255
            save_img(clean_adaptive_thr_img, results_folder + "adaptive/" + fname)
            
            #
            # COMBINE ADAPTIVE AND ORIGINAL
            #
            reinforced_edges = (img - sobelxy).astype("uint8")
            save_img(reinforced_edges, results_folder + "reinforced/" + fname)
            #show_img(cv2.add(img, clean_adaptive_thr_img), "Combined")
            
            
            #
            # 4-REGION OTSU THRESHOLD
            #
            patches = divide_image(blur, 4)
            thresholds = []
            for patch in patches:
                otsu_thr, otsu_img = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                thresholds.append(otsu_thr)
            min_threshold = min(thresholds)
            _, patch_threshold_img = cv2.threshold(blur, min_threshold, 255, cv2.THRESH_BINARY)
            dilate_img = cv2.morphologyEx(patch_threshold_img, cv2.MORPH_DILATE, eight_neighbors, iterations=2)
            erode_img = cv2.morphologyEx(dilate_img, cv2.MORPH_ERODE, eight_neighbors, iterations=2)
            
            # Clone result
            erode_img_original = erode_img.copy()
            
            # Remove 1/8 of the image
            eighth_division_h = int(erode_img.shape[0]/8)
            erode_img[erode_img.shape[0] - eighth_division_h:erode_img.shape[0], :] = 0
            save_img(erode_img, results_folder + "patch_thresh/" + fname)

            #
            # GLOBAL OTSU THRESHOLD
            #

            otsu_thr, otsu_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            print(fname, ": ", otsu_thr)

            # Clone result
            otsu_img_original = otsu_img.copy()
            
            # Remove 1/8 of the image
            eighth_division_h = int(otsu_img.shape[0]/8)
            otsu_img[otsu_img.shape[0] - eighth_division_h:otsu_img.shape[0], :] = 0

            # Closing/Opening
            closing = cv2.morphologyEx(otsu_img, cv2.MORPH_CLOSE, eight_neighbors, iterations=2)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, eight_neighbors, iterations=2)
            #show_img(opening, "Opening: %s" % (fname))

            # Calculate connected components of global Otsu th
            # and keep only one
            ret, marked_img = cv2.connectedComponents(opening)
            center_y = int(marked_img.shape[0]/2)
            center_x = int(marked_img.shape[1]/2)
            selected_component = marked_img[center_y, center_x]

            if selected_component == 0:
                center_y += int(marked_img.shape[0]/4)
                selected_component = marked_img[center_y, center_x]
                
                if selected_component == 0:
                    print("Couldn't find hand: %s" % fname)
                    continue
            
            # Remove every other component, fill holes and save
            one_component = (marked_img == selected_component).astype("uint8") * 255
            one_component = cv2.morphologyEx(one_component, cv2.MORPH_DILATE, eight_neighbors, iterations=6)
            one_component = skimage.morphology.remove_small_holes(one_component == 255, 500)
            one_component = one_component.astype("uint8") * 255
            #show_img(one_component, "One component: %s" % (fname))
            save_img(one_component, results_folder + "otsu/" + fname)

            #
            # SKELETONIZE CORE COMPONENT
            # 
            to_skel_img = skimage.morphology.skeletonize(one_component == 255)
            to_skel_img = to_skel_img.astype("uint8")*127
            save_img(one_component - to_skel_img, results_folder + "skel/" + fname)

            ##
            ## SLIC SUPERPIXELS
            ##
            #sp_mask = cv2.morphologyEx(erode_img, cv2.MORPH_DILATE, eight_neighbors, iterations=3)
            ## Apply slic to reinforced image
            #m_slic = skimage.segmentation.slic(reinforced_edges, n_segments=500, compactness=.1, mask=sp_mask, start_label=1, channel_axis=None)
            #m_slic_boundaries = skimage.segmentation.mark_boundaries(img, m_slic)
            #save_img(skimage.img_as_ubyte(m_slic_boundaries), results_folder + "mslic/" + fname)
            


            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
