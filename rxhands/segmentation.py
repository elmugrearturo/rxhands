import os
import cv2
import numpy as np

from rxhands.auxiliary import *

def main(data_folder="./data/", results_folder="./results/"):
    kernel = np.ones((5,5), np.uint8)
    eight_neighbors = np.ones((3, 3), np.uint8)
    for fname in os.listdir(data_folder) :
        if fname.endswith(".png") or fname.endswith(".tiff") :
            img = load_gray_img(data_folder + fname)
            #norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            blur = cv2.medianBlur(img, 7)
            otsu_thr, otsu_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            print(fname, ": ", otsu_thr)
            
            #
            # REMOVE BOTTOM (BIN IMG)
            #
            
            # Clone result
            otsu_img_original = otsu_img.copy()
            
            # Remove 1/8 of the image
            eighth_division_h = int(otsu_img.shape[0]/8)
            otsu_img[otsu_img.shape[0] - eighth_division_h:otsu_img.shape[0], :] = 0

            # Closing/Opening
            closing = cv2.morphologyEx(otsu_img, cv2.MORPH_CLOSE, eight_neighbors, iterations=2)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, eight_neighbors, iterations=2)
            #show_img(opening, "Opening: %s" % (fname))

            # Calculate connected components
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
            
            # Remove every other component
            one_component = (marked_img == selected_component).astype("uint8") * 255
            #show_img(one_component, "One component: %s" % (fname))
            
            save_img(one_component, results_folder + "thres/" + fname)
            #import ipdb;ipdb.set_trace()
        #break

if __name__ == "__main__" :
    main()
