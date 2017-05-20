import cv2
import numpy as np

def find_and_draw_corners():
    img_path = ""
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # manually define the corner points for a piece of paper
    corners = np.array([(57,39), (110,39), (122,75), (39, 75)], dtype="float32")

    return corners

def draw_corners(img, corners):
    # draw points on the image
    img2 = img.copy()
    for c in corners:
        cv2.circle(img2, tuple(c), 3, (0, 255, 0), -1)

    cv2.imshow(img2)
    cv2.waitKey(1)

def four_point_transform(pts):
    max_width, max_height = 300, 300
    hw_ratio = 11/8.5 # height/width ratio of a 8.5/11 piece of paper
    scale = int(max_width/12)

    center_x = 150
    center_y = 250

    dst = np.array([
    [center_x - scale, center_y - scale*hwratio], #top left
    [center_x + scale, center_y - scale*hwratio], #top right
    [center_x + scale, center_y + scale*hwratio], #bottom right
    [center_x - scale, center_y + scale*hwratio], #bottom left
    ], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts, dst)

    return M
