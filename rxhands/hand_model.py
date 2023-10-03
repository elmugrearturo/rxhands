
from rxhands.geometry import *


def find_thumb_cut_point(bin_img, skel_positions):
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
