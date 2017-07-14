import cv2
import numpy as np
import Utility


def test_perspective_transformation():
    image = Utility.resize_image(cv2.imread("SampleImages/stoplight2.jpg", 1), 480)
    cv2.imshow("asdfsafd", Utility.apply_perspective_transformation(image))
    cv2.waitKey(0)


def test_get_image_size():
    image = cv2.imread("SampleImages/stoplight.jpg", 1)
    print image.shape
    print "height: %s" % image.shape[0]
    print "width: %s" % image.shape[1]


def test_four_point_transform():
    image = Utility.resize_image(cv2.imread("SampleImages/stoplight2.jpg", 1), 480)
    M = Utility.four_point_transform()
    print M
    warped = cv2.warpPerspective(image, M, (300, 300))
    cv2.imshow("original", image)
    cv2.imshow("afsdfas", warped)
    cv2.waitKey(0)


if __name__ == "__main__":
    test_four_point_transform()
