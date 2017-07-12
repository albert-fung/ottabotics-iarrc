from Constants import *
import cv2
import numpy as np
import math
import Utility


# Functions accessed by main


def compute_turn_angle(image):
    # TODO: segment the frame horizontally
    subframe_list = split_frame(image, 5)


    # TODO: get line segments

    # TODO: cluster lines by angle

    # TODO: apply weighted sum to dominant line angles

    pass


# Helper functions

def get_line_segments(image):
    """
    Find the line segments in the frame using HoughLinesP
    :return: 
    """
    mask_edges = cv2.Canny(image, 0, 200, apertureSize=3)

    # variables for HoughLinesP

    # max_line_gap = 20 # default/previous/initial value
    max_line_gap = 5
    min_line_length = 50
    threshold = 20

    lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, threshold, min_line_length, max_line_gap)

    return lines


def cluster_by_angle(line_list):
    clusters = []
    last_angle = -180

    for line in line_list:
        if abs(last_angle - line.angle) > 5:
            clusters.append([line])
        else:
            clusters[-1].append(line)
        last_angle = line.angle
    return clusters


def cluster_by_proximity(image, line_list):
    clusters = []

    for line in line_list:
        clusters.append([line])
        for query_line in line_list:
            distance = compute_point_distance(line.midpoint, query_line.midpoint)
            if distance <= 100 and abs(line.angle - query_line.angle) < 10:
                clusters[-1].append(query_line)
                cv2.line(image, line.midpoint, query_line.midpoint, (128, 128, 128), 1)

    filtered_clusters = []
    for cluster in clusters:
        if len(cluster) >= CLUSTER_MIN_SIZE:
            filtered_clusters.append(cluster)

    return image, filtered_clusters


def compute_point_distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    return abs(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))


def compute_cluster_position(cluster):
    """

    :param cluster: 
    :return: 
    """
    x_total = 0
    y_total = 0

    for line in cluster:
        x_total += line.midpoint[0]
        y_total += line.midpoint[1]

    return x_total / len(cluster), y_total / len(cluster)


def compute_cluster_magnitude(cluster):
    """

    :param cluster: 
    :return: 
    """
    length = 0

    for line in cluster:
        length += line.length

    return length


def compute_cluster_direction(cluster):
    angle_sum = 0
    for line_segment in cluster:
        angle_sum += line_segment.angle

    return angle_sum / len(cluster)


def generate_weight_list(number_of_weights):
    weight_list = []
    for i in range(number_of_weights):
        weight_list.append(1 / 2**(i+1))

    weight_sum = 0
    for w in weight_list:
        weight_sum += w

    weight_diff = 1 - weight_sum
    weight_comp = weight_diff / number_of_weights

    for i, w in enumerate(weight_list):
        weight_list[i] += weight_comp

    weight_sum = 0
    for w in weight_list:
        weight_sum += w
    print "DEBUG: weight sum: %s" % weight_sum

    return weight_list


def split_frame(image, number_of_subimages):
    """
    Split an image into a list of subimages
    :param image: 
    :param number_of_subimages:
    :return: 
    """
    initial_x = image.shape[1]
    initial_y = image.shape[0]

    final_y = int(initial_y / number_of_subimages)

    subimage_list = []

    start_index = 0
    end_index = final_y

    for i in range(number_of_subimages):
        try:
            subimage_list.append(image[:][start_index:end_index])
            start_index = end_index + 1
            end_index += final_y
        except Exception as e:
            Utility.get_stacktrace()

    return subimage_list
