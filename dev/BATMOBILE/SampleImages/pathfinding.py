import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math


def get_lines():
    """
    Return list of lines (point, point) found by houghlinesp
    :return: 
    """
    pass


def compute_line_length(line):
    """
    assumptions: line is an array of two points where each point is an array if two integers (x,y)
    :param line: 
    :return: 
    """
    length = math.sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2)
    return length


def compute_line_angle(line):
    """
    angle with vertical
    :param line: 
    :return: 
    """
    dx = line[1][0] - line[0][0]
    dy = line[1][1] - line[0][1]
    rads = math.atan2(-dy,dx)
    rads %= 2*math.pi
    degs = -math.degrees(rads)
    if degs <= -180: 
        degs = degs + 180
        
    degs = degs + 90
    return degs


def cluster_lines():
    """
    
    :return: 
    """
    pass


def apply_perspective_transform():
    """
    
    :return: 
    """
    pass


def compute_direction(img):
    """
    - extract lines from image
    - create list of lines sorted by line length
    - exception/edge cases for longest line in the image
        - line angled towards outside edge: can be ignored in some cases
        - slope close to inf (horizontal line): can be ignored in most cases
    
    :param img: 
    :return: 
    """
    pass


def main():
    """
    - read image from camera
    - create two regions of interest: left and right halves split vertically
    - in each half:
    - get all line segments using houghlinesp
    - for each line segment, compute line length and angle with horizontal
    - create clusters for lines of similar angle
    - after getting two angles (one for each half of the frame), compute average
    :return: 
    """
    pass


def resize_image(img, height):
    """
    Resizes an image to the specified height
    :param img: 
    :param height: 
    :return: 
    """
    ratio = float(height) / img.shape[0]
    dim = (int(img.shape[1]*ratio),height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def test_get_line_segments_from_curve():
    """
    
    :return: 
    """
    camera = cv2.VideoCapture(0)
    #ret, frame = camera.read()
    frame = cv2.imread("img_sample_1.jpg",1)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = resize_image(img,480)
    img = cv2.GaussianBlur(img,(3,3),0)

    ## finding edges 
    maskedges=cv2.Canny(img,0,200,apertureSize=3)
    #Cutting away the small edges while increasing the edges that are visable
    # maskedges=cv2.morphologyEx(maskedges,cv2.MORPH_OPEN,(15,15))

    """
    #Showing the image
    cv2.imshow("frame",frame)
    cv2.imshow("res",res)
    cv2.imshow("mask",mask)
    cv2.imshow("maskedges",maskedges)
    #cv2.imshow("opening",opening)
    """


    #variables for houghlines
    maxLineGap= 1
    minLineLength = 3
    #The P at the end of HoughLines causes the output to become (x1,y1,x2,y2) instead of ro and phi 
    lines = cv2.HoughLinesP(maskedges,1,np.pi/180,15,minLineLength,maxLineGap)

    print("Length of lines: %d" % len(lines))

    for line in lines:
        for x1,y1,x2,y2 in line:
            print("(%d,%d) (%d,%d)" % (x1,y1,x2,y2))
            # print x1, y1, x2, type(y2)
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0))

            print compute_line_length(((x1,y1),(x2,y2)))
            print compute_line_angle(((x1,y1),(x2,y2)))
            #break
        #break

    cv2.imshow("asdf",img)
    cv2.waitKey(0)
    camera.release()
    cv2.destroyAllWindows()
    

test_get_line_segments_from_curve()


