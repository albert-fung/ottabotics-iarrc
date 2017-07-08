from Constants import *
import cv2
import numpy as np
from Utility import *
from math import hypot

"""


"""


# Functions accessed by main

def ready_to_start():
    """
    Check whether the start condition is satisfied (traffic light is found
    and the light is green). Once the start condition is met, the output 
    becomes constant and addition clock cycles are not wasted on 
    analyzing the frame.
    
    :return: bool
    """


# Helper functions


def match_features(reference, query, min_match_count = 10):
    """
    Returns list of destination points
    """
    try:
        ref_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    except:
        ref_grey = reference
    try:
        query_grey = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    except:
        query_grey = query

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(query_grey,None)
    kp2, des2 = sift.detectAndCompute(ref_grey,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matches_mask = mask.ravel().tolist()

        h,w = query_grey.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1,h-1], [w-1, 0]]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        dst_list = dst.ravel().tolist()
    else:
        dst_list = []

    for i,pnt in enumerate(dst_list):
        dst_list[i] = int(pnt)

    # print dst_list

    return dst_list


def find_circles(img):
    try:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        grey = img
    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1.2, 20)
    try:
        circles = np.uint16(np.around(circles))
        circle_list = []
        for circle in circles[0,:]:
            circle_list.append(circle.ravel().tolist())
    except:
        circle_list = []

    return circle_list


def draw_circle(img, point, radius=25, colour=(0,255,0)):
    cv2.circle(img, point, radius, colour, -1)
    return img


def draw_all_circles(img, circles, colour=(0,255,0)):
    for circle in circles:
        cv2.circle(img, (circle[0],circle[1]), circle[2], colour,1)


def mask_all_circles(img, circles):
    """
    Returns image where the area outside of each circle is cropped out
    """
    img_copy = img.copy()
    rows, cols, _ = img_copy.shape

    print("Total pixels: %d" % (rows*cols))

    for i in range(cols):
        for j in range(rows):
            point_is_out = True
            for circle in circles:
                if hypot(i-circle[0], j-circle[1]) < circle[2]:
                    point_is_out = False
                    break
            if point_is_out:
                img_copy[j][i] = 0

    return img_copy
