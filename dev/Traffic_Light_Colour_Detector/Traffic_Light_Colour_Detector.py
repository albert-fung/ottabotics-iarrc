import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from math import hypot


def resize_image(img, height):
    """
    Resizes an image to the specified height
    """
    ratio = float(height) / img.shape[0]
    dim = (int(img.shape[1]*ratio),height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def get_ROI(img, points):
    """
    Returns image after being cropped to coordinates defined by the list points
    """
    x_list = []
    y_list = []
    for i in range(0,len(points),2):
        x_list.append(points[i])
    for i in range(1,len(points),2):
        y_list.append(points[i])

    tl = (min(x_list),min(y_list))
    br = (max(x_list),max(y_list))

    roi = img[tl[1]:br[1], tl[0]:br[0]]

    # print("top left coords: " + str(tl))
    # print("bottom right coords: " + str(br))

    return roi

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
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>min_match_count:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = query_grey.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
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
    colour[0] = int(colour[0]*2)
    colour[1] = int(colour[1]/255.0*100.0)
    colour[2] = int(colour[2]/255.0*100.0)
    return colour


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
            pointIsOut = True
            for circle in circles:
                if hypot(i-circle[0], j-circle[1]) < circle[2]:
                    pointIsOut = False
                    break
            if pointIsOut:
                img_copy[j][i] = 0

    return img_copy


def apply_perspective_transform(img, points):
    """
    using the points list, extract the four points and label as top-left, top-right, bottom-left, bottom-right

    source: http://www.pyimagesearch.com/2014/05/05/building-pokedex-python-opencv-perspective-warping-step-5-6/
    """
    pass

def main():
    reference = cv2.imread('card_reference_resized.jpg',1)
    query = cv2.imread('card_query_5_resized.jpg',1)
    traffic_light = cv2.imread('img_sample_2.jpg',1)
    traffic_light = resize_image(traffic_light,300)

    dst_points = match_features(query, reference)
    roi = get_ROI(query, dst_points)

    circles = find_circles(traffic_light)

    print circles


    mask = mask_all_circles(traffic_light, circles)
    draw_all_circles(mask,circles)

    cv2.imshow("Original Traffic Light",traffic_light)
    cv2.imshow("Masked Traffic Light",mask)
    # cv2.imshow("Original query image",query)
    # cv2.imshow("ROI of query",roi)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
