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


start_time_total = time.time()

model_dir = 'imagenet/'
# images_dir = 'images/'
images_dir = 'resized_training_output/'

# TODO: sort object classes by their parent folder instead of their file names
list_images = [images_dir+f for f in os.listdir(images_dir) if re.search('jpg|JPG', f)]

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

# print(">>> list of image:",list_images)


def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = "imagenet/"
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




def create_graph():
    with gfile.FastGFile(os.path.join(
    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
    nb_features = 2048
    features = np.empty((len(list_images),nb_features))
    labels = []

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            # print(">>> ind:",ind)
            # print(">>> image:",image)
            # if (ind%100 == 0):
            print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
            {'DecodeJpeg/contents:0': image_data})
            features[ind,:] = np.squeeze(predictions)
            labels.append(re.split('_\d+',image.split('/')[1])[0])
    print(">>> features:",len(features[0]))
    print(">>> labels:",labels)
    return features, labels

features,labels = extract_features(list_images)
print(">>> done extracting features")


pickle.dump(features, open('features', 'wb'))
pickle.dump(labels, open('labels', 'wb'))

features = pickle.load(open('features'))
labels = pickle.load(open('labels'))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size=0.2, random_state=42)

clf = LinearSVC(C=1.0, loss='squared_hinge', penalty='l2',multi_class='ovr')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(">>> X_test[0]:",len(X_test[0]))  ### X_test is a list of image features, each 2048 long
# print(">>> X_test[0][0]",X_test[0][0])
# print(">>> X train:", X_train)
# print(">>> X test:", X_test)
# print(">>> y train:", y_train)
# print(">>> y test:", y_test)

print(">>> prediction finished")

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
    prediction = clf.predict(features)

    print(prediction)

print("=======================================")
print("image prediction:")
predict_image("images/water_bottle_12.jpg")
print("=======================================")

def plot_confusion_matrix(y_true,y_pred):
    print(">>> y_true:",y_true)
    print(">>> y_pred:", y_pred)
    cm_array = confusion_matrix(y_true,y_pred)
    print(">>> type cm array:")
    print(type(cm_array))
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    plt.imshow(cm_array[:-1,:-1], interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix", fontsize=16)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.set_label('Number of images', rotation=270, labelpad=30, fontsize=12)
    xtick_marks = np.arange(len(true_labels))
    ytick_marks = np.arange(len(pred_labels))
    plt.xticks(xtick_marks, true_labels, rotation=90)
    plt.yticks(ytick_marks,pred_labels)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 12
    fig_size[1] = 12
    plt.rcParams["figure.figsize"] = fig_size

    print(">>> cm array:",cm_array)

# print(">>> accuracy_score:",accuracy_score(y_test,y_pred))

print("Accuracy: {0:0.1f}%".format(accuracy_score(y_test,y_pred)*100))
plot_confusion_matrix(y_test,y_pred)
print(">>> plot finished")

end_time_total = time.time()

print(">>> elapsed time:", end_time_total - start_time_total)

