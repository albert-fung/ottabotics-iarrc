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
from subprocess import call, check_output
import tarfile
import os.path
from six.moves import urllib
import sys

list_files = check_output("ls resized_test_output/",shell=True).split("\n")
list_files = list_files[:-1]

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

create_graph()

def extract_features_for_one(image_name):
    # start_time_graph = time.time()
    nb_features = 2048
    features = np.empty((1,nb_features))
    # create_graph()
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        image_data = gfile.FastGFile(image_name, 'rb').read()
        predictions = sess.run(next_to_last_tensor, {'DecodeJpeg/contents:0' : image_data})
        features[0,:] = np.squeeze(predictions)
	
        # end_time_graph = time.time()
	# print(">>> setup time: " + str(end_time_graph - start_time_graph))
        return features

clf_ = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
clf_.fit(X_train, y_train)

def predict_image(image_name):
    features = extract_features_for_one(image_name)
    # start_time_predict = time.time()
    # clf_ = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2', multi_class='ovr')
    # clf_.fit(X_train, y_train)
    prediction = clf_.predict(features)
    # end_time_predict = time.time()
    # print(">>> prediction time: " + str(end_time_predict - start_time_predict))
    return prediction

def run_tests():
    passed_count = 0
    failed_count = 0
    detected_count = 0
    average_time = 0

    for image in list_files:
        start_time_predict_list = time.time()
        file_name = image
        prediction = predict_image("test_images/" + file_name)
        print("=======================================")
        # print("image prediction:")
        print(file_name)
        output_list = file_name.split(".")[0].split("_")[:-2]
        output_str = ""
        for i in output_list:
            output_str += i + "_"
        output_str = output_str[:-1]
        # print(output_str + " :: " + prediction[0])
        if output_str == prediction[0]:
            print("PASSED")
            passed_count += 1
        elif prediction[0] in output_str:
            print("DETECTED")
            detected_count += 1
        else:
            print("FAILED")
            failed_count += 1
        end_time_predict_list = time.time()
        print("prediction time: " + str(end_time_predict_list - start_time_predict_list))
        average_time += (end_time_predict_list - start_time_predict_list)
    print("=======================================")
    print("PASSED: " + str(passed_count))
    print("FAILED: " + str(failed_count))
    print("DETECTED: " + str(detected_count))
    test_count = passed_count + failed_count + detected_count
    average_time = average_time / test_count
    print("AVERAGE: " + str(average_time))

def run_tests_2():
    passed_count = 0
    failed_count = 0
    detected_count = 0
    average_time = 0

    for image in list_files:
        start_time_predict_list = time.time()
        file_name = image
        prediction = predict_image("test_images/" + file_name)
        # print("=======================================")
        # print("image prediction:")
        # print(file_name)
        # output_list = file_name.split(".")[0].split("_")[:-2]
        # output_str = ""
        # for i in output_list:
            # output_str += i + "_"
        # output_str = output_str[:-1]
        # print(output_str + " :: " + prediction[0])
        # if output_str == prediction[0]:
            # print("PASSED")
            # passed_count += 1
        if prediction[0] in file_name:
            print("DETECTED")
            detected_count += 1
        else:
            print("FAILED")
            failed_count += 1
        end_time_predict_list = time.time()
        # print("prediction time: " + str(end_time_predict_list - start_time_predict_list))
        average_time += (end_time_predict_list - start_time_predict_list)
    print("=======================================")
    print("PASSED: " + str(passed_count))
    print("FAILED: " + str(failed_count))
    print("DETECTED: " + str(detected_count))
    test_count = passed_count + failed_count + detected_count
    average_time = average_time / test_count
    print("AVERAGE: " + str(average_time))

run_tests()
