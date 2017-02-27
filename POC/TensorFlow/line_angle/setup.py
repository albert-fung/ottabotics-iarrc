import os
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
# %matplotlib inline
import pickle

import time
from subprocess import call
import tarfile
import os.path
from six.moves import urllib
import sys

model_dir = 'model/'
# images_dir = 'images/'
# list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = "model/"
    print("model directory: ", dest_directory)
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (
              filename, float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    
maybe_download_and_extract()

