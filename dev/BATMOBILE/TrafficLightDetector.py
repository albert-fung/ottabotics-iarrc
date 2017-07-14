from Constants import *
import cv2
import numpy as np
# from Utility import *
import Utility
from math import hypot

"""


"""


# Functions accessed by main

def ready_to_start(query_image, reference_image_file="SampleImages/card_reference_resized.jpg"):
    """
    Check whether the start condition is satisfied (traffic light is found
    and the light is green). Once the start condition is met, the output 
    becomes constant and addition clock cycles are not wasted on 
    analyzing the frame.
    
    :return: bool
    """
    # reference_image = cv2.imread(reference_image_file, 1)
    # reference_image = cv2.imread("SampleImages/stoplight2.jpg", 1)
    traffic_light_test = cv2.imread("SampleImages/stoplight2.jpg", 1)
    reference_image = Utility.resize_image(cv2.imread("SampleImages/ottabotics_bunny.jpg", 1), 480)
    query_image = Utility.resize_image(cv2.imread("SampleImages/ottabotics_bunny_sample_3.jpg", 1), 480)

    dst_points = match_features(query_image, reference_image)
    print ">>> dst_points:"
    print dst_points
    print dst_points[0]

    """
    for i in range(0, len(dst_points), 4):
        try:
            cv2.line(query_image, (dst_points[i], dst_points[i+1]), (dst_points[i+2], dst_points[i+3]), (0, 255, 0), 2)
        except:
            Utility.get_stacktrace()
    """
    for i in range(0, len(dst_points), 2):
        try:
            cv2.circle(query_image, (dst_points[i], dst_points[i+1]), 3, (128, 255, 128), 2)
        except:
            Utility.get_stacktrace()

    roi = Utility.get_roi(query_image, dst_points)
    circles = find_circles(roi)
    # circles = find_circles(traffic_light_test)

    # for debugging
    # isolate the circles by masking everything else out
    # mask = mask_all_circles(query_image, circles)
    # mask = draw_all_circles(mask, circles)

    # TODO: get list of points for the centers of circles
    print "Number of circles found: %s" % len(circles)
    # a circle is [x, y, radius]
    circle_point_list = []
    for circle in circles:
        circle_point_list.append((circle[0], circle[1]))
        cv2.circle(traffic_light_test, (circle[0], circle[1]), 5, (0, 255, 0), 3)
    # TODO: get the colour at each center
    circle_colour_list = []
    for point in circle_point_list:
        circle_colour_list.append(get_colour_hsv(roi, point))
        # circle_colour_list.append(get_colour_hsv(traffic_light_test, point))
        # print point
    # print circle_colour_list
    # TODO: compare each center colour to a reference green or red colour
    for colour in circle_colour_list:
        if colours_match_hsv(colour, (120, 0, 0)):
            print "Colours match!"
            break

    # cv2.imshow("masked", traffic_light_test)
    cv2.imshow("asfadsf", reference_image)
    cv2.imshow("masked", roi)
    cv2.imshow("query", query_image)
    cv2.waitKey(0)

# Helper functions


def match_features(reference, query, min_match_count=10):
    """
    Returns list of destination points
    """
    try:
        ref_grey = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    except:
        Utility.get_stacktrace()
        ref_grey = reference
    try:
        query_grey = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
    except:
        Utility.get_stacktrace()
        query_grey = query

    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(query_grey, None)
    kp2, des2 = sift.detectAndCompute(ref_grey, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        h, w = query_grey.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts,M)

        dst_list = dst.ravel().tolist()
    else:
        dst_list = []

    for i, pnt in enumerate(dst_list):
        dst_list[i] = int(pnt)

    # print dst_list

    return dst_list


def find_circles(img):
    try:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        Utility.get_stacktrace()
        grey = img
    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1.2, 20)
    try:
        circles = np.uint16(np.around(circles))
        circle_list = []
        for circle in circles[0, :]:
            circle_list.append(circle.ravel().tolist())
    except:
        Utility.get_stacktrace()
        circle_list = []

    return circle_list


def draw_circle(img, point, radius=25, colour=(0, 255, 0)):
    cv2.circle(img, point, radius, colour, -1)
    return img


def draw_all_circles(img, circles, colour=(0, 255, 0)):
    for circle in circles:
        cv2.circle(img, (circle[0], circle[1]), circle[2], colour, 1)
    return img


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


def get_colour_bgr(img, point):
    """
    Returns the colour in BGR format for an image at a specified point (x,y)
    """
    colour = img[point[0]][point[1]].ravel().tolist()
    return colour


def get_colour_hsv(img, point):
    """
    Returns the colour in BGR format for an image at a specified point (x,y)
    """
    try:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except:
        img_hsv = img
    colour = img[point[0]][point[1]].ravel().tolist()
    colour[0] = int(colour[0] * 2)
    colour[1] = int(colour[1] / 255.0 * 100.0)
    colour[2] = int(colour[2] / 255.0 * 100.0)
    return colour


def colours_match_bgr(query, reference):
    pass


def colours_match_hsv(query, reference):
    # blue: h=170
    # green: h=85
    # red: h=0
    # yellow: h=43
    if query[0] > reference[0] - 15 and query[0] < reference[0] + 15:
        return True
    else:
        return False


def main():
    # camera = cv2.VideoCapture(0)
    # ret, frame = camera.read()
    # image = Utility.apply_preprocessing(frame)

    # image = Utility.apply_preprocessing(cv2.imread("SampleImages/card_query_resized.jpg", 1))
    image = cv2.imread("SampleImages/traffic_light_bunny_2.jpg", 1)
    image = Utility.resize_image(image, 480)

    # cv2.imshow("asfd", image)
    # cv2.waitKey(0)

    ready_to_start(image)

if __name__ == "__main__":
    main()
