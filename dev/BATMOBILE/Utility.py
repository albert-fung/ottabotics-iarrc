from Constants import *
import cv2


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


def write_to_log(message, log_file="LOG.txt"):
    with open(log_file, "a") as f:
        f.write(message)


def save_image(image, file_name):
    cv2.imwrite("SavedImages/" + file_name, image)


def record_video():
    pass


def apply_perspective_transformation():
    pass


def generate_perspective_transformation_params():
    pass


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


def split_frame(image, number_of_subimages):
    """
    Split an image into a list of subimages
    :param image: 
    :param number_of_subimages:
    :return: 
    """
    initial_x = image.shape[1]
    initial_y = image.shape[0]

    final_y = int(initial_y / number_of_subimages)

    subimage_list = []

    start_index = 0
    end_index = final_y

    for i in range(number_of_subimages):
        try:
            subimage_list.append(image[:][start_index:end_index])
            start_index = end_index + 1
            end_index += final_y
        except Exception as e:
            print ">>> Exception: %s" % e

    return subimage_list


def get_roi(img, points):
    """
    Returns image after being cropped to coordinates defined by the list points
    :param img:
    :param points:
    :return: 
    """
    x_list = []
    y_list = []
    for i in range(0, len(points),2):
        x_list.append(points[i])
    for i in range(1, len(points),2):
        y_list.append(points[i])

    tl = (min(x_list), min(y_list))
    br = (max(x_list), max(y_list))

    roi = img[tl[1]:br[1], tl[0]:br[0]]

    # print("top left coords: " + str(tl))
    # print("bottom right coords: " + str(br))

    return roi


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
    colour[0] = int(colour[0] * 2)
    colour[1] = int(colour[1] / 255.0 * 100.0)
    colour[2] = int(colour[2] / 255.0 * 100.0)
    return colour


def get_stacktrace():
    import traceback
    return traceback.format_exc()
