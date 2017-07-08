"""
Check if the wireless stop has been triggered. The output
remains constant after the trigger is detected.
"""


WIRELESS_STOP_TRIGGERED = False


# Functions accessed by main
def stop_triggered():
    """
    Return the state of the wireless stop
    :return: bool
    """
    return WIRELESS_STOP_TRIGGERED


# Helper functions


def poll_arduino():
    """
    Check one of the internal variables of the Arduino board
    connected to the wireless RC receiver and set WIRELESS_STOP_TRIGGERED
    to that value
    :return: None
    """
    pass
