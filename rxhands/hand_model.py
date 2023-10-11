
from rxhands.geometry import *
from rxhands.ridge import normalized_ridge_img, sobelx_ridge_img
from random import randint


class Region(object):

    def __init__(self, bounding_box, area, center, bin_img=None, *args, **kwargs):
        assert isinstance(bounding_box, tuple)
        assert isinstance(center, tuple)
        self.bin_img = bin_img
        self.positions = find_positions(bin_img)
        self.positions.sort()
        self.bounding_box = bounding_box
        self.width = self.bounding_box[2]
        self.height = self.bounding_box[3]
        
        self.top_coordinate = \
                (bounding_box[0], bounding_box[1])
        self.bottom_coordinate = \
                (bounding_box[0] + bounding_box[2],
                 bounding_box[1] + bounding_box[3])

        self.top_right_coordinate = \
                (bounding_box[0] + bounding_box[2],
                 bounding_box[1]
                 )
        self.bottom_left_coordinate = \
                (bounding_box[0],
                 bounding_box[1] + bounding_box[3]
                 )

        self.area = area
        self.center = center # (i, j)

        c = euclidean_distance(self.top_right_coordinate,
                               self.bottom_left_coordinate)
        # Get internal angle
        self.angle_rad = np.arcsin(self.height / c)
        self.angle = np.degrees(self.angle_rad)

        # Get inclination direction
        higher_point = self.positions[0]
        lower_point = self.positions[-1]
        if higher_point[1] == lower_point[1]:
            # No inclination (90deg)
            self.inclination = 0
        elif higher_point[1] < lower_point[1]:
            # More than 90deg
            self.inclination = -1
        else:
            # Less than 90deg
            self.inclination = 1

        self.predicted = []

    def get_predicted(self):
        all_predicted = []
        for i in range(len(self.predicted)):
            prediction = self.predicted[i]
            if prediction == 1:
                all_predicted.append(self.positions[i])
        return all_predicted

    def paint_region(self, color_img):
        color = [randint(0, 255), randint(0, 255), randint(0, 255)]
        color_img = cv2.rectangle(color_img, 
                                  self.top_coordinate,
                                  self.bottom_coordinate,
                                  color,
                                  1)

        color_img = cv2.circle(color_img, 
                               self.center[::-1],
                               4,
                               color,
                               -1)

        color_img = cv2.putText(color_img, 
                                str(int(self.angle))+"deg",
                                self.bottom_coordinate,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                color,
                                4)

        if len(self.predicted) > 0:
            for i in range(len(self.positions)):
                if self.predicted[i] == 1:
                    color_img = cv2.circle(color_img, self.positions[i][::-1], 2, (0, 0, 255), -1)
        return color_img


class Thumb(Region):

    def paint_region(self, color_img):
        color_img = cv2.putText(color_img, 
                                "T",
                                self.center[::-1],
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                [255, 255, 255],
                                4)
        return super(Thumb, self).paint_region(color_img)


class Finger(Region):
    
    def paint_region(self, color_img):
        color_img = cv2.putText(color_img, 
                                "F",
                                self.center[::-1],
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                [255, 255, 255],
                                4)
        return super(Finger, self).paint_region(color_img)
    

class Knuckles(Region):
    
    def __init__(self, center, radius, *args, **kwargs):
        assert len(center) == 2
        self.center = center # (i, j)
        self.radius = int(radius)
        
        ## Calculate rect for region
        ## We took 45deg diagonals, so
        #diag_radians = 45 * (np.pi / 180)
        ## opposite cathetus 
        #region_height = int(np.sin(diag_radians) * radius)
        #region_width = int(np.cos(diag_radians) * radius)
        ## X,Y format
        #starting_corner = (thumb_cut_point[1] - region_width,
        #                   thumb_cut_point[0] - region_height
        #                   )
        #ending_corner = (thumb_cut_point[1] + region_width,
        #                 thumb_cut_point[0]
        #                 )

    def paint_region(self, color_img):
        color_img = cv2.circle(color_img, self.center[::-1], self.radius,
                   [255, 0, 0], 5)
        return color_img


class Hand(object):
    
    def __init__(self, raw_img, processing_img, segmented_img, *args, **kwargs):
        '''segmented_img has to have only one connected component''' 
        
        self.raw_img = raw_img
        self.processing_img = processing_img
        self.segmented_img = segmented_img

        #
        # RIDGE IMG (for classifier)
        #
        print(f"\t\tCalculating ridge img...")
        self.classifier_img = sobelx_ridge_img(processing_img)
        self.classifier_right_img = sobelx_ridge_img(rotate_img(processing_img, -30))
        self.classifier_left_img = sobelx_ridge_img(rotate_img(processing_img, 30))

        #
        # SKELETONIZE
        #
        print(f"\t\tCalculating hand skeleton...")
        skel_img = skeletonize(segmented_img)
        self.skel_img = skel_img
        
        #
        # PRUNE SKELETON
        #
        print(f"\t\tPrunning skeleton...")
        # Skeleton has to be marked with 1s
        prunned_skel_img = prune_skeleton((skel_img != 0).astype("uint8"))
        self.prunned_skel_img = prunned_skel_img

        print(f"\t\t\tFinding relevant positions in prunned skeleton...")
        # Find positions of the prunned skeleton
        prunned_skel_positions = find_positions(prunned_skel_img)
        self.prunned_skel_positions = prunned_skel_positions
        
        # 
        # FIND KNUCKLE REGION
        #

        # THUMB CUT POINT
        thumb_cut_point, _ = self.find_thumb_cut_point(segmented_img, 
                                                  prunned_skel_positions)
        # Find distance to border from cut point
        # Only calculate diagonals
        distances = position_to_border_distances(segmented_img, 
                                                 thumb_cut_point, 
                                                 [0, 2, 6, 8])
        proposed_radius = distances.max()

        self.knuckles = Knuckles(thumb_cut_point, proposed_radius)

        #
        # FIND POSSIBLE FINGERS
        # 
        self.thumb, self.fingers = self.find_possible_fingers()

    def paint_to_img(self, img):
        new_img = img.copy()
        # Paint prunned skeleton
        new_img = cv2.add(new_img, 
                          (self.prunned_skel_img != 0).astype("uint8") * 255)
        color_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR) 

        color_img = self.knuckles.paint_region(color_img)
        if self.thumb != None:
            color_img = self.thumb.paint_region(color_img)
        for finger in self.fingers:
            color_img = finger.paint_region(color_img)
        return color_img

    def find_possible_fingers(self):
        # Separate skeleton
        skel_img = self.prunned_skel_img.copy()
        skel_img = cv2.circle(skel_img, 
                              self.knuckles.center[::-1], 
                              self.knuckles.radius,
                              [0],
                              -1)
        # Ensure that it is a binary img
        skel_img = (skel_img != 0).astype("uint8")
        (no_labels, labeling, stats, centroids) = \
                cv2.connectedComponentsWithStats(skel_img, 
                                                 connectivity=8)
        # Identify possible thumb
        # Thumb will be systematically to the right
        thumb = None
        thumb_candidates = []
        k_center = self.knuckles.center
        k_radius = self.knuckles.radius
        for label in range(1, no_labels):
            x, y = centroids[label]
            # knuckle center is (i,j)
            if x > k_center[1]:
                if k_center[0] + k_radius > y > k_center[0] - k_radius:
                    thumb_candidates.append(label)

        # Only accept it if there is a single candidate
        if len(thumb_candidates) == 1:
            label = thumb_candidates[0]
            bounding_box = tuple(stats[label][:4])
            area = stats[label][4]
            centroid = centroids[label]
            centroid = (int(centroid[1]), int(centroid[0]))
            thumb = Thumb(bounding_box,
                          area,
                          centroid,
                          (labeling == label).astype("uint8"))

        # Identify possible fingers
        # Avoid zero
        fingers = []
        for label in range(1, no_labels):
            centroid = centroids[label]
            if label in thumb_candidates :
                continue
            elif k_center[0] > centroid[1] :
                bounding_box = tuple(stats[label][:4])
                area = stats[label][4]
                centroid = (int(centroid[1]), int(centroid[0]))
                fingers.append(Finger(bounding_box,
                                      area,
                                      centroid,
                                      (labeling == label).astype("uint8")))
        return thumb, fingers 

    def find_thumb_cut_point(self, bin_img, skel_positions):
        # Finds the point where a thumb may first be
        # identified: going upwards, a straight horizontal
        # line crosses two distinct regions (palm and thumb)
        relevant_lines = set([p[0] for p in skel_positions])
        relevant_lines = sorted(list(relevant_lines), reverse=True)
        optimal_point = None
        for i in relevant_lines:
            filled, gaps = check_gaps_per_row(bin_img, [i])
            if len(filled[0]) == 2:
                init_p, end_p = filled[0][0]
                length = end_p[1] - init_p[1]
                optimal_point = (i, int(init_p[1] + length/2))
                break
        skeleton_point = None
        relevant_skeleton_points = [p for p in skel_positions if p[0] == i]
        if len(relevant_skeleton_points) == 1:
            skeleton_point = relevant_skeleton_points[0]
        else:
            distances = []
            for j in range(len(relevant_skeleton_points)):
                distances.append(euclidean_distance(optimal_point, 
                                                    relevant_skeleton_points[j]))
            selected = np.argmin(distances)
            skeleton_point = relevant_skeleton_points[selected]
        return optimal_point, skeleton_point

    def classify_points(self, clf):
        # 
        # FIND FINGER POINTS IN RIDGE IMAGE
        #
        i = 0
        for finger in self.fingers:
            i += 1
            if finger.angle >= 75:
                # Almost straight finger
                print(f"\t\t Finger {i} (straight):")
                
                print(f"\t\t\tFinding skeleton positions in classifier img...")
                skel_patches = patches_from_positions(self.classifier_img,
                                                      finger.positions,
                                                      (51, 51),
                                                      0)

                # CLASSIFY
                print(f"\t\t\tClassifiying patches...")
                X = np.array([patch.flatten() for patch in skel_patches])
                y_pred = clf.predict(X)
                finger.predicted = y_pred

            else:
                print(f"\t\t Finger {i} (inclination):")
                pass

