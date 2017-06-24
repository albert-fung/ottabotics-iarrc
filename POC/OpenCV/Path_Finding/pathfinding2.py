import cv2
import numpy as np
import sys
import math

"""
PSUDOCODE AND PLANNING

Functions to add:
- get line segments
- image preprocessing
- segment frame into vertical sections
- perspective transform to get bird's eye view
- camera calibration
- line clustering based on angle
- line clustering based on proximity
- find dominant line direction
- angle filtering
- weighted turning angles (higher toward bottom, less towards top)
"""

IMAGE_HEIGHT = 480
MIN_CLUSTER_SIZE = 10


class Line:
    def __init__(self, x1, y1, x2, y2):
        # set start and end points
        self.p1 = (x1, y1)
        self.p2 = (x2, y2)

        # set length
        self.length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # compute line angle
        dx = x2 - x1
        dy = y2 - y1
        rads = math.atan2(-dy, dx)
        rads %= 2 * math.pi
        degs = -math.degrees(rads)
        if degs <= -180:
            degs += 180
        degs += 90
        self.angle = degs

        # set midpoint
        self.midpoint = (((x1 + x2) / 2), ((y1 + y2) / 2))

    def __str__(self):
        return str(self.angle)

    def __repr__(self):
        return self.__str__()

    def get_coords(self):
        return self.p1, self.p2


class Cluster:
    def __init__(self, cluster):
        self.cluster = cluster
        self.magnitude = compute_cluster_magnitude(cluster)
        self.position = compute_cluster_position(cluster)


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


def resize_image(img, height):
    """
    Resizes an image to the specified height
    :param img: 
    :param height: 
    :return: 
    """
    ratio = float(height) / img.shape[0]
    dim = (int(img.shape[1]*ratio), height)

    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return image


def apply_preprocessing(image):
    """
    Apply suite of basic image processing filters like blur and colour conversion
    :param image:
    :return: 
    """
    # convert colour space from BGR to grey to BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # resize the image to height 480
    image = resize_image(image, IMAGE_HEIGHT)
    # apply gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)

    return image


def perspective_transform_calibration():
    """
    
    :return: 
    """
    pass


def perspective_transform():
    """
    
    :return: 
    """
    pass


def camera_calibration():
    pass


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
        if len(cluster) >= MIN_CLUSTER_SIZE:
            filtered_clusters.append(cluster)

    return image, filtered_clusters


def compute_point_distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    return abs(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))


def find_dominant_line_angle():
    pass


def angle_filtering():
    pass


def weighted_turn_angles():
    pass


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


def main():
    pass


def test_proximity(use_webcam=False):
    camera = cv2.VideoCapture(0)

    import time

    while True:
        start_time = time.time()
        line_list = []
        if use_webcam:
            ret, frame = camera.read()
        else:
            frame = cv2.imread("test_sample_3.png", 1)
        image = apply_preprocessing(frame)
        line_segments = get_line_segments(image)

        try:
            for line in line_segments:
                for x1, y1, x2, y2 in line:
                    # print "(%d,%d) (%d,%d)" % (x1, y1, x2, y2)
                    # cv2.circle(image, (x1, y1), 5, (255, 0, 255))
                    # cv2.circle(image, (x2, y2), 5, (0, 255, 255))
                    # cv2.circle(image, ((x1 + x2) / 2, (y1 + y2) / 2), 5, (255, 255, 0))
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    line_list.append(Line(x1, y1, x2, y2))

            image, clusters = cluster_by_proximity(image, line_list)

            for cluster in clusters:
                cv2.circle(image, compute_cluster_position(cluster), 5, (255, 0, 255))

            print "CLUSTERS: %s" % len(clusters)
            print "LINE COUNT: %s" % len(line_list)

            cv2.imshow("test", image)
            if cv2.waitKey(30) != 255:
                break

            cv2.imwrite("output.jpg", image)
        except Exception as e:
            print ">>> EXCEPTION: %s" % e

        end_time = time.time()
        print "Framerate: %s" % str(int(1 / (end_time - start_time)))

    if use_webcam:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 1:
        if args[0] == "test":
            test_proximity()
    else:
        main()
