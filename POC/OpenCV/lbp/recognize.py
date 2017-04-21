# import the necessary packages
from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import cv2
from subprocess import call,check_output

# initialize the local binary patterns descriptor along with
# the data and label lists
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# name of training directory is also the label for the files it contains
TRAINING_DIR = "training_samples/"
TESTING_DIR = "testing_samples/"

def get_training_file_paths():
	path = TRAINING_DIR
	file_list = []
	label_list = check_output("ls " + path,shell=True).split("\n")[:-1]
	for label in label_list:
		file_list_temp = check_output("ls " + path + label,shell=True).split("\n")[:-1]
		for i in range(len(file_list_temp)):
			file_list_temp[i] = path + label + "/" + file_list_temp[i]
		file_list = file_list + file_list_temp
	return file_list


def get_testing_file_paths():
	path = TESTING_DIR
	file_list = check_output("ls " + path,shell=True).split("\n")[:-1]
	for i in range(len(file_list)):
		file_list[i] = path + file_list[i]
	return file_list

# loop over the training images
for imagePath in get_training_file_paths():
	# load the image, convert it to grayscale, and describe it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)

	print hist

	# extract the label from the image path, then update the
	# label and data lists
	labels.append(imagePath.split("/")[-2])
	data.append(hist)

	print("path: %s\nlabel: %s\n=====" % (imagePath,imagePath.split("/")[-2]))

print("=====\nlabels: %s\ndata: %s\n=====\n" % (labels,data))

# train a Linear SVM on the data
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

# loop over the testing images
for imagePath in get_testing_file_paths():
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist)[0]

	# display the image and the prediction
	#cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
	#	1.0, (0, 0, 255), 3)
	# cv2.imshow("Image", image)
	# cv2.waitKey(0)

	print("=====\nfile: %s\nprediction: %s\n" % (imagePath,prediction))
