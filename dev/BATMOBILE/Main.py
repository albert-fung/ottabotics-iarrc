"""
Main.py

The controller of the entire system. All other scripts are called
from the main function within this script.

"""

import Constants
import Utility
import sys
import getopt
import cv2
import numpy as np
import TrafficLightDetector
import ArduinoController
import ArduinoEStop
import Pathfinder


def main(test_image="", use_webcam=True, debug_messages=False):
    # TODO: initialize camera
    camera = cv2.VideoCapture(0)

    while True:
        if use_webcam:
            try:
                ret, frame = camera.read()
            except:
                print "Camera is not attached. Exiting"
                Utility.get_stacktrace()
                return -1
        else:
            try:
                frame = cv2.imread(test_image, 1)
            except:
                print "Test image file was not set. Exiting"
                Utility.get_stacktrace()
                return -1

        image = Utility.apply_preprocessing(frame)

        # TODO: check traffic light detector status

        if TrafficLightDetector.ready_to_start(image):
            break

        break

    while True:
        """
        # TODO: check estop status
        if ArduinoEStop.stop_triggered():
            return 0
        """

        # TODO: get turn angle from pathfinder
        turn_angle = Pathfinder.compute_turn_angle(Utility.apply_perspective_transformation(image))
        # TODO: send angle to arduino controller
        ArduinoController.move(angle=turn_angle)


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv, "d", ["speed=", "starttraffic", "startcircuit", "startdrag"])
    except getopt.GetoptError:
        # TODO: print usage
        sys.exit(2)

    # TODO: declare and initialize flags for main
    debug = False
    # this isn't a good solution, maybe store a list of flags as they are set?

    for opt, arg in opts:
        if opt == "-h":
            # TODO: print help
            sys.exit()
        if opt == "-d":
            # set debug flag
            pass
        if opt == "--speed":
            # set speed variable
            pass
