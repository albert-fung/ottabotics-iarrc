import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math
from collections import deque

# POC: get all line segments from an image and display the line clustere of greatest length


class Line:
    def __init__(self,coords):
        self.coords = coords # ((x1,y1),(x2,y2))
        self.length = self.compute_line_length(coords)
        self.angle = -self.compute_line_angle(coords)

    def __str__(self):
        return str(self.angle)

    def __repr__(self):
        return self.__str__()

    def compute_line_length(self,line):
        """
        assumptions: line is an array of two points where each point is an array of two integers (x,y)
        """
        length = math.sqrt((line[0][0] - line[1][0])**2 + (line[0][1] - line[1][1])**2)
        return length

    def compute_line_angle(self,line):
        """
        angle with vertical
        vertical is 0, horizontal is 90
        left is positive, right is negative
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

    def get_length(self):
        pass

    def get_angle(self):
        pass

    def get_coords(self):
        return self.coords


def get_lines():
    """
    return list of lines (point, point) found by houghlinesp
    """
    pass


def cluster_lines(line_data):
    clusters = []
    last_angle = -180
    for line in line_data:
        if abs(last_angle - line.angle) > 5:
            clusters.append([line])
        else:
            clusters[-1].append(line)
        last_angle = line.angle
    return clusters


def apply_perspective_transform():
    pass


def compute_direction(img):
    # extract lines from image

    # create list of lines sorted by line length

    ### exception/edge cases for longest line in the image
        # line angled towards outside edge: can be ignored in some cases
        # slope close to inf (horizontal line): can be ignored in most cases
    pass


def sort_line_list(line_list_):
    newlist = sorted(line_list_, key=lambda x: x.angle, reverse=True)
    return newlist


def main():
    # read image from camera
    # create two regions of interest: left and right halves split vertically
    # in each half:
    # get all line segments using houghlinesp
    # for each line segment, compute line length and angle with horizontal
    # create clusters for lines of similar angle
    # after getting two angles (one for each half of the frame), compute average
    pass


def resize_image(img, height):
    """
    Resizes an image to the specified height
    """
    ratio = float(height) / img.shape[0]
    dim = (int(img.shape[1]*ratio),height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def draw_output_line(img,angle):
    start = (320,475)
    if angle < 0:
        endx = 640
    elif angle > 0:
        endx = 0
    else:
        endx = 320

    # TODO: check if endy is greater than 480

    endy = 320.0 / np.tan(angle)
    if endy > 480 or endy < 0:
        endy = 0
        endx = 480.0 * np.tan(angle)
        if angle < 0:
            endx = 320.0 + endx
        elif angle > 0:
            endx = 320.0 - endx            
        else:
            endx = 320
    else:
        endy = 480 - endy

    end = (int(endx), int(endy))

    cv2.line(img, start, end, (255,255,255), 5)
    cv2.circle(img,start,5,(0,255,0),-1)
    cv2.circle(img,end,5,(0,255,0),-1)

    # print end
    vis_line = Line((start,end))
    print "(vis | actual | diff): %d | %d | %d" % (int(vis_line.angle),int(angle),abs(int(vis_line.angle - angle)))

    return img


def draw_line_ex(img,theta):
    #img = img.clone()
    a = np.cos(theta)
    b = np.sin(theta)
    #x0 = a*rho
    #y0 = b*rho
    x0 = img.shape[1]/2
    y0 = img.shape[0]/2
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)

    return img


def test_get_line_segments_from_curve():
    camera = cv2.VideoCapture(0)
    output_buffer = deque(maxlen=10)

    while True:
        line_list = []
        ret, frame = camera.read()
        #frame = cv2.imread("img_sample_15.jpg",1)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = resize_image(img,480)
        img = cv2.GaussianBlur(img,(3,3),0)

        ##print "image shape: %s" % str(img.shape)

        ## finding edges 
        maskedges=cv2.Canny(img,0,200,apertureSize=3)
        #Cutting away the small edges while increasing the edges that are visable
        # maskedges=cv2.morphologyEx(maskedges,cv2.MORPH_OPEN,(15,15))

        #cv2.imshow("fawe",maskedges)
        #cv2.waitKey(0)

        # cv2.circle(img,(640,480),20, (255,255,0),-1)

        """
        #Showing the image
        cv2.imshow("frame",frame)
        cv2.imshow("res",res)
        cv2.imshow("mask",mask)
        cv2.imshow("maskedges",maskedges)
        #cv2.imshow("opening",opening)
        """


        #variables for houghlines
        maxLineGap= 20
        minLineLength = 50
        #The P at the end of HoughLines causes the output to become (x1,y1,x2,y2) instead of ro and phi 
        lines = cv2.HoughLinesP(maskedges,1,np.pi/180,15,minLineLength,maxLineGap)

        try:
            ##print("Length of lines: %d" % len(lines))
        

            for line in lines:
                for x1,y1,x2,y2 in line:
                    #print("(%d,%d) (%d,%d)" % (x1,y1,x2,y2))
                    # print x1, y1, x2, type(y2)
                    cv2.circle(img,(x1,y1),5,(255,0,255))
                    cv2.circle(img,(x2,y2),5,(0,255,255))
                    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
                    line_list.append(Line(((x1,y1),(x2,y2))))
                    #print "line length: %s" % compute_line_length(((x1,y1),(x2,y2)))
                    #print "line angle: %s" % compute_line_angle(((x1,y1),(x2,y2)))
                    #break
                #break

            line_list_sorted = sort_line_list(line_list)

            ##for line in line_list_sorted:
                ##print "a: %d | l: %d" % (int(line.angle), int(line.length))

            line_list_clustered = cluster_lines(line_list_sorted)
            ##print "length of clustered list: %d" % len(line_list_clustered)

            cluster_data = []
            for cluster in line_list_clustered:
                cluster_length = 0
                cluster_angle = 0
                for line in cluster:
                    cluster_length += line.length
                    cluster_angle += line.angle
                cluster_angle = cluster_angle / len(cluster)
                cluster_data.append((int(cluster_length), int(cluster_angle)))

            #print cluster_data

            cluster_data_sorted = sorted(cluster_data, key=lambda x: x[0], reverse=True)
           
            ##print "=== sorted cluster data ==="
            ##print cluster_data_sorted

            """
            angle_average = 0
            for i in range(1):
                theta = cluster_data_sorted[i][1]
                a = np.cos(theta)
                b = np.sin(theta)
                #x0 = a*rho
                #y0 = b*rho
                x0 = img.shape[1]/2
                y0 = img.shape[0]/2
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(img,(x1,y1),(x2,y2),(0,0,255),5)

                angle_average += theta

            angle_average = angle_average / 1
            """




            """
            print "angle average: %s" % angle_average

            # draw average line of n

            theta = angle_average
            a = np.cos(theta)
            b = np.sin(theta)
            #x0 = a*rho
            #y0 = b*rho
            x0 = img.shape[1]/2
            y0 = img.shape[0]/2
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            """

            """
            # draw longest line

            theta_ = cluster_data_sorted[0][1]
            print "output angle: %d" % int(theta_)
            a = np.cos(theta_)
            b = np.sin(theta_)
            #x0 = a*rho
            #y0 = b*rho
            x0 = img.shape[1]/2
            y0 = img.shape[0]/2
            x1 = int(x0 + 400*(-b))
            y1 = int(y0 + 400*(a))
            x2 = int(x0 - 400*(-b))
            y2 = int(y0 - 400*(a))

            cv2.line(img,(x1,y1),(x2,y2),(255,255,255),1)
            """

            output_buffer.append(cluster_data_sorted[0][1])
            output_angle = 0
            for angle in output_buffer:
                output_angle += angle
            output_angle = output_angle / len(output_buffer)

            img = draw_output_line(img, output_angle)

            #img = draw_line_ex(img,output_angle)

            ##print output_buffer

            ##print "output angle: %d" % int(output_angle)

            ##print "output line: (%s,%s),(%s,%s)" % (x1,y1,x2,y2)

            cv2.putText(img, str(output_angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)

        except Exception as e:
            print "Exception: %s" % e
            cv2.putText(img, "nope", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 255), 3)

        cv2.imshow("asdf",img)
        if cv2.waitKey(30) != 255:
            break
    camera.release()
    cv2.destroyAllWindows()
    
test_get_line_segments_from_curve()

#l = Line(((0,0),(1,0)))
#print l.length
#print l.angle
#print l.coords
#print l





