# import os
# import re

# import tensorflow as tf
# import tensorflow.python.platform
# from tensorflow.python.platform import gfile
import numpy as np
# import pandas as pd
# import sklearn
# from sklearn import cross_validation
# from sklearn.metrics import accuracy_score, confusion_matrix
# from sklearn.svm import SVC, LinearSVC
# import matplotlib.pyplot as plt
# %matplotlib inline
# import pickle

import time
from subprocess import call, check_output
# import tarfile
# import os.path
# from six.moves import urllib
# import sys
import cv2

source_dir = "test_images"
output_dir = "resized_test_output"

list_of_images = check_output("ls " + source_dir,shell=True).split("\n")[:-1]

print list_of_images

if len(list_of_images) > 0:
	for image_name in list_of_images:
		image = cv2.imread(source_dir + "/" + image_name)
		# cv2.imshow("adf",image)
		# cv2.waitKey(0)

		ratio = 100.0 / image.shape[0]
		dim = (int(image.shape[1] * ratio),100)
		resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
		print("size of resized image: " + str(resized.shape))
		# cv2.imshow("2",resized)
		# cv2.waitKey(0)
		cv2.imwrite(output_dir + "/" + image_name, resized)

