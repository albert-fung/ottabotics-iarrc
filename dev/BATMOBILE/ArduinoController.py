"""
Provides communication between the Arduino board and Python.
"""

import Utility
import serial


# Functions accessed by main
def set_turn_angle(turn_angle):
    pass


def set_speed(speed):
    pass


def move(angle=47, speed=50):
    """
    
    :param angle: 32/47/62 - left/straight/right
    :param speed: 32/47/62 - back/neutral/forward
    :return: 
    """
    try:
        ser = serial.Serial('/dev/ttyUSB0')

        ser.write(chr(255))
        ser.write(chr(speed))

        ser.write(chr(angle))
    except:
        Utility.get_stacktrace()

# Helper functions
