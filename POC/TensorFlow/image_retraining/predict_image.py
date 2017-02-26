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


start_time = time.time()

model_dir = 'imagenet/'
images_dir = 'images/'
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]

features = pickle.load(open('features'))
labels = pickle.load(open('labels'))

def create_graph():
    with gfile.FastGFile(os.path.join(
    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.5, random_state=42)

def extract_features_for_one(image_name):
    nb_features = 2048
    features = np.empty((1,nb_features))
    create_graph()
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        image_data = gfile.FastGFile(image_name, 'rb').read()
        predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0' : image_data})
        features[0,:] = np.squeeze(predictions)

        return features

def predict_image(image_name):
    features = extract_features_for_one(image_name)
    clf_ = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
    clf_.fit(X_train, y_train)
    prediction = clf_.predict(features)

    return prediction

prediction = predict_image("images/" + sys.argv[1])
print("=======================================")
print("image prediction:")
print(prediction[0])
print("=======================================")

end_time = time.time()

print(">>> elapsed time: " + str(end_time - start_time))