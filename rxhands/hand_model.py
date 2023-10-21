
from rxhands.geometry import *
from rxhands.clustering import dbscan_clustering, kmeans_clustering
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

        self.predictions = []

    # Comparison on the horizontal axis
    def __gt__(self, b):
        if self.center[1] > b.center[1]:
            return True
        return False

    def __ge__(self, b):
        if self.center[1] >= b.center[1]:
            return True
        return False
    
    def __lt__(self, b):
        if self.center[1] < b.center[1]:
            return True
        return False

    def __le__(self, b):
        if self.center[1] <= b.center[1]:
            return True
        return False

    def __eq__(self, b):
        if b == None:
            return False
        if self.center[1] == b.center[1]:
            return True
        return False
    
    def __ne__(self, b):
        if b == None:
            return True
        if self.center[1] != b.center[1]:
            return True
        return False

    def get_predicted(self):
        all_predicted = []
        for i in range(len(self.predictions)):
            prediction = self.predictions[i]
            if prediction == 1:
                all_predicted.append(self.positions[i])
        return all_predicted
    
    def approximate_prediction_points(self, clustering_algorithm="kmeans"):
        predicted = self.get_predicted()
        no_of_points = 2
        # Check if two or three points
        if self.finger_id == Finger.INDEX or self.finger_id == Finger.LITTLE:
            if self.predictions[-1] or \
                    self.predictions[-2] or \
                    self.predictions[-3] or \
                    self.predictions[-4]:
                        no_of_points = 3

        if len(predicted) != 0:
            try:
                if clustering_algorithm == "kmeans":
                    return kmeans_clustering(predicted, no_of_points)
                elif clustering_algorithm == "dbscan":
                    return dbscan_clustering(predicted)
            except Exception as e:
                print(e)
        return {}

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

        if len(self.predictions) > 0:
            for i in range(len(self.positions)):
                if self.predictions[i] == 1:
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

    UNKNOWN = 0
    INDEX = 1
    MIDDLE = 2
    RING = 3
    LITTLE = 4

    def __init__(self, bounding_box, area, center, bin_img=None, *args, **kwargs):
        super(Finger, self).__init__(bounding_box, area, center, bin_img)
        self.finger_id = self.UNKNOWN
        self.distal_tip = None # Tip
        self.distal_interphalangeal = None # First joint after Tip
        self.proximal_interphalangeal = None # Second joint after Tip
        self.metacarpophalangeal = None
        self.middle_phalanx_length = None # Distance between joints
        self.proximal_phalanx_length = None

    def get_finger_label(self):
        if Finger.UNKNOWN:
            return "Finger"
        elif Finger.INDEX:
            return "Index"
        elif Finger.MIDDLE:
            return "Middle"
        elif Finger.RING:
            return "Ring"
        elif Finger.LITTLE:
            return "Little"
        raise ValueError("Bad finger label")

    def set_finger_id(self, new_id):
        if new_id not in \
                [self.UNKNOWN, self.INDEX, self.MIDDLE, self.RING, self.LITTLE]:
                    raise ValueError("Unknown finger value")
        self.finger_id = new_id

    def paint_region(self, color_img):
        if self.finger_id == Finger.UNKNOWN:
            label = "F"
        elif self.finger_id == Finger.LITTLE:
            label = "L"
        elif self.finger_id == Finger.RING:
            label = "R"
        elif self.finger_id == Finger.MIDDLE:
            label = "M"
        elif self.finger_id == Finger.INDEX:
            label = "I"
        else:
            return color_img
        color_img = cv2.putText(color_img, 
                                label,
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

        #
        # LABEL FINGERS
        #
        self.label_fingers()

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

    def paint_poi_to_img(self, img):
        new_img = img.copy()
        # Create color img
        color_img = cv2.cvtColor(new_img, cv2.COLOR_GRAY2BGR) 

        # Paint knuckles center
        color_img = cv2.circle(color_img, self.knuckles.center[::-1], 2, (255, 0, 0), -1)
        for finger in self.fingers:
            if finger.distal_tip != None :
                color_img = cv2.circle(color_img, finger.distal_tip[::-1], 5, (0, 0, 255), -1)
            if finger.distal_interphalangeal != None:
                color_img = cv2.circle(color_img, finger.distal_interphalangeal[::-1], 5, (0, 0, 255), -1)
            if finger.proximal_interphalangeal != None:
                color_img = cv2.circle(color_img, finger.proximal_interphalangeal[::-1], 5, (0, 0, 255), -1)
            if finger.metacarpophalangeal != None:
                color_img = cv2.circle(color_img, finger.metacarpophalangeal[::-1], 5, (0, 0, 255), -1)
        return color_img

    def label_fingers(self):
        self.fingers.sort()
        finger_labels = [0 for _ in range(len(self.fingers))]
        if len(self.fingers) == 4:
            # Found 4, assume they are correct
            finger_labels[0] = Finger.LITTLE
            finger_labels[1] = Finger.RING
            finger_labels[2] = Finger.MIDDLE
            finger_labels[3] = Finger.INDEX
        elif len(self.fingers) == 3:
            if self.fingers[-1] > self.knuckles:
                if self.fingers[-2] >= self.knuckles:
                    # Two last fingers right of center, then
                    # they must be index and middle
                    finger_labels[-1] = Finger.INDEX
                    finger_labels[-2] = Finger.MIDDLE
                    # Guess third based on angle
                    if self.fingers[-3].angle < 70:
                        finger_labels[-3] = Finger.LITTLE
                    else:
                        finger_labels[-3] = Finger.RING
                else:
                    # Just one finger right of center
                    # Could be either index or middle
                    # Decide with angle
                    if self.fingers[-1].angle >= 85:
                        finger_labels[-1] = Finger.MIDDLE
                        finger_labels[-2] = Finger.RING
                    else:
                        finger_labels[-1] = Finger.INDEX
                        if self.fingers[-2].angle >= 85:
                            finger_labels[-2] = Finger.MIDDLE
                        else:
                            finger_labels[-2] = Finger.RING
                    finger_labels[-3] = Finger.LITTLE
            else:
                # Three fingers left of center, assume 
                # little, ring, middle
                finger_labels[0] = Finger.LITTLE
                finger_labels[1] = Finger.RING
                finger_labels[2] = Finger.MIDDLE
        elif len(self.fingers) == 2:
            if self.fingers[-1] > self.knuckles:
                if self.fingers[-2] >= self.knuckles:
                    # Two fingers right of center, then
                    # they must be index and middle
                    finger_labels[-1] = Finger.INDEX
                    finger_labels[-2] = Finger.MIDDLE
                else:
                    # Only one finger to the right, 
                    # guess everything based on angle
                    if self.fingers[-1].angle > 85:
                        finger_labels[-1] = Finger.MIDDLE
                    else:
                        finger_labels[-1] = Finger.INDEX

                    if self.fingers[-2].angle < 70:
                        finger_labels[-2] = Finger.LITTLE
                    else:
                        finger_labels[-2] = Finger.RING
            else:
                # Two fingers left of center, one can be middle
                if self.fingers[-1].angle > 85:
                    finger_labels[-1] = Finger.MIDDLE
                    if self.fingers[-2].angle < 70:
                        finger_labels[-2] = Finger.LITTLE
                    else:
                        finger_labels[-2] = Finger.RING
                else:
                    finger_labels[-1] = Finger.RING
                    finger_labels[-2] = Finger.LITTLE
        elif len(self.fingers) == 1 :
            if self.fingers[0] >= self.knuckles:
                # Can be middle or index
                if self.fingers[0].angle > 85:
                    finger_labels[0] = Finger.MIDDLE
                else:
                    finger_labels[0] = Finger.INDEX
            else:
                # Can be middle, ring or little
                if self.fingers[0].angle < 70:
                    finger_labels[0] = Finger.LITTLE
                elif self.fingers[0].angle <= 85:
                    finger_labels[0] = Finger.RING
                else:
                    finger_labels[0] = Finger.MIDDLE

        for i in range(len(finger_labels)):
            self.fingers[i].set_finger_id(finger_labels[i])

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

    def find_points_in_finger(self, clf, clustering_algorithm="kmeans"):
        self.classify_internal_finger_points(clf)

        print("\t Calculating points of interest in fingers...")
        for finger in self.fingers:
            clusters = finger.approximate_prediction_points(clustering_algorithm)
            points = [clusters[k] for k in clusters.keys()]
            if len(points) != 2 and len(points) != 3:
                continue
            points.sort()
            
            finger.distal_interphalangeal = points[0]
            finger.proximal_interphalangeal = points[1]
            finger.middle_phalanx_length = euclidean_distance(points[0], points[1])
            if len(points) == 3:
                finger.metacarpophalangeal = points[2]
                finger.proximal_phalanx_length = euclidean_distance(points[1], points[2])
            
            ## Find finger tip
            ## From distal_interphalangeal
            slope_i = finger.distal_interphalangeal[0] - finger.proximal_interphalangeal[0]
            slope_j = finger.distal_interphalangeal[1] - finger.proximal_interphalangeal[1]
            vertical_line = False
            if slope_i == 0:
                # Horizontal line
                # Something is wrong
                continue

            if slope_j == 0:
                # Vertical line
                vertical_line = True
                forever_j = finger.distal_interphalangeal[1]
            else:
                slope = slope_i / slope_j
                b = finger.distal_interphalangeal[0] - (slope*finger.distal_interphalangeal[1])
            initial_i = finger.distal_interphalangeal[0]
            visited = []
            border_found = False
            for k in range(1, int(finger.middle_phalanx_length)):
                # Formula (i-b)/m = j
                new_i = int(initial_i - k)
                if not vertical_line :
                    new_j = int((new_i - b) / slope)
                else:
                    new_j = forever_j
                visited.append((new_i, new_j))
                try:
                    if self.segmented_img[(new_i, new_j)] == 0:
                        border_found = True
                        break
                except:
                    break
            
            if border_found :
                # Search tip
                center_i, center_j = visited[-1]
                patch = cut_patch(self.raw_img, visited[-1], (99, 99))
                thresholds = skimage.filters.threshold_multiotsu(patch)
                patch = (patch >= thresholds[1]).astype("uint8") * 255
                patch_j = 49
                for patch_i in range(99):
                    filled, gaps = check_gaps_per_row(patch, [patch_i])
                    if len(filled[0]) > 0:
                        init_interval, end_interval = filled[0][0]
                        if init_interval[1] == end_interval[1]:
                            patch_j = init_interval[1]
                        else:
                            patch_j = init_interval[1] + (end_interval[1] - init_interval[1]) / 2
                            patch_j = int(patch_j)
                        #print(patch_i)
                        #show_img(patch, "parche")
                        break

                if patch_i < 49:
                    # Final i is a substraction
                    final_i = center_i - (49 - patch_i)
                elif patch_i > 49:
                    # Final i is an addition
                    final_i = center_i + (patch_i - 49)
                else:
                    final_i = center_i
                
                if patch_j < 49:
                    # Final i is a substraction
                    final_j = center_j - (49 - patch_j)
                elif patch_i > 49:
                    # Final i is an addition
                    final_j = center_j + (patch_j - 49)
                else:
                    final_j = center_j
                
                finger.distal_tip = (final_i, final_j)

    def classify_internal_finger_points(self, clf):
        # 
        # FIND FINGER POINTS IN RIDGE IMAGE
        #
        i = 0
        for finger in self.fingers:
            i += 1
            if finger.angle > 75:
                # Almost straight finger
                print(f"\t\t Finger {i} (straight):")
                
                print(f"\t\t\tFinding skeleton positions in classifier img...")
                skel_patches = patches_from_positions(self.classifier_img,
                                                      finger.positions,
                                                      (51, 51),
                                                      0)
            else:
                print(f"\t\t Finger {i} (inclination):")
                if finger.inclination < 0 :
                    # left inclination, rotate right
                    rotation_matrix = get_rotation_matrix(self.raw_img, -30)
                    classifier_img = self.classifier_right_img
                    #show_img(self.classifier_right_img, "right")
                else :
                    # right inclination, rotate left
                    rotation_matrix = get_rotation_matrix(self.raw_img, 30) 
                    classifier_img = self.classifier_left_img
                    #show_img(self.classifier_left_img, "left")
                
                # Get transformed positions
                print(f"\t\t\tFinding skeleton positions in classifier img...")
                rotated_positions = transform_positions(finger.positions, rotation_matrix)
                skel_patches = patches_from_positions(classifier_img,
                                                      rotated_positions,
                                                      (51, 51),
                                                      0)
                #for patch in skel_patches:
                #    show_img(patch, "patch")
            # CLASSIFY
            print(f"\t\t\tClassifiying patches...")
            X = np.array([patch.flatten() for patch in skel_patches])
            y_pred = clf.predict(X)
            finger.predictions = y_pred

            self.classifier_img.shape

    def to_dictionary(self):
        # Only fingers (x, y)
        df = {}
        height, width = self.classifier_img.shape
        for finger in self.fingers:
            if finger.finger_id == Finger.INDEX:
                # tip 5, joint points 6,7,8
                if finger.distal_tip != None:
                    df['5_X'] = finger.distal_tip[1]
                    df['5_Y'] = height - finger.distal_tip[0]
                else:
                    df['5_X'] = None
                    df['5_Y'] = None
                if finger.distal_interphalangeal != None:
                    df['6_X'] = finger.distal_interphalangeal[1]
                    df['6_Y'] = height - finger.distal_interphalangeal[0]
                else:
                    df['6_X'] = None
                    df['6_Y'] = None
                if finger.proximal_interphalangeal != None:
                    df['7_X'] = finger.proximal_interphalangeal[1]
                    df['7_Y'] = height - finger.proximal_interphalangeal[0]
                else:
                    df['7_X'] = None
                    df['7_Y'] = None
                if finger.metacarpophalangeal != None:
                    df['8_X'] = finger.metacarpophalangeal[1]
                    df['8_Y'] = height - finger.metacarpophalangeal[0]
                else:
                    df['8_X'] = None
                    df['8_Y'] = None
            elif finger.finger_id == Finger.MIDDLE:
                # tip 10, points 11,12,13
                if finger.distal_tip != None:
                    df['10_X'] = finger.distal_tip[1]
                    df['10_Y'] = height - finger.distal_tip[0]
                else:
                    df['10_X'] = None
                    df['10_Y'] = None
                if finger.distal_interphalangeal != None:
                    df['11_X'] = finger.distal_interphalangeal[1]
                    df['11_Y'] = height - finger.distal_interphalangeal[0]
                else:
                    df['11_X'] = None
                    df['11_Y'] = None
                if finger.proximal_interphalangeal != None:
                    df['12_X'] = finger.proximal_interphalangeal[1]
                    df['12_Y'] = height - finger.proximal_interphalangeal[0]
                else:
                    df['12_X'] = None
                    df['12_Y'] = None
                if finger.metacarpophalangeal != None:
                    df['13_X'] = finger.metacarpophalangeal[1]
                    df['13_Y'] = height - finger.metacarpophalangeal[0]
                else:
                    df['13_X'] = None
                    df['13_Y'] = None
            elif finger.finger_id == Finger.RING:
                # tip 15, points 16,17,18
                if finger.distal_tip != None:
                    df['15_X'] = finger.distal_tip[1]
                    df['15_Y'] = height - finger.distal_tip[0]
                else:
                    df['15_X'] = None
                    df['15_Y'] = None
                if finger.distal_interphalangeal != None:
                    df['16_X'] = finger.distal_interphalangeal[1]
                    df['16_Y'] = height - finger.distal_interphalangeal[0]
                else:
                    df['16_X'] = None
                    df['16_Y'] = None
                if finger.proximal_interphalangeal != None:
                    df['17_X'] = finger.proximal_interphalangeal[1]
                    df['17_Y'] = height - finger.proximal_interphalangeal[0]
                else:
                    df['17_X'] = None
                    df['17_Y'] = None
                if finger.metacarpophalangeal != None:
                    df['18_X'] = finger.metacarpophalangeal[1]
                    df['18_Y'] = height - finger.metacarpophalangeal[0]
                else:
                    df['18_X'] = None
                    df['18_Y'] = None
            elif finger.finger_id == Finger.LITTLE:
                # tip 20, points 21,22,23
                if finger.distal_tip != None:
                    df['20_X'] = finger.distal_tip[1]
                    df['20_Y'] = height - finger.distal_tip[0]
                else:
                    df['20_X'] = None
                    df['20_Y'] = None
                if finger.distal_interphalangeal != None:
                    df['21_X'] = finger.distal_interphalangeal[1]
                    df['21_Y'] = height - finger.distal_interphalangeal[0]
                else:
                    df['21_X'] = None
                    df['21_Y'] = None
                if finger.proximal_interphalangeal != None:
                    df['22_X'] = finger.proximal_interphalangeal[1]
                    df['22_Y'] = height - finger.proximal_interphalangeal[0]
                else:
                    df['22_X'] = None
                    df['22_Y'] = None
                if finger.metacarpophalangeal != None:
                    df['23_X'] = finger.metacarpophalangeal[1]
                    df['23_Y'] = height - finger.metacarpophalangeal[0]
                else:
                    df['23_X'] = None
                    df['23_Y'] = None
        return df
