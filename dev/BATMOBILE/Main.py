import Constants
import Utility
import sys
import getopt

def main(debug=False):
    pass


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(argv, "d", ["speed=", "starttraffic", "startcircuit", "startdrag"])
    except getopt.GetoptError
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