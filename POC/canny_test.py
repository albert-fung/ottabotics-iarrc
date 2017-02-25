import cv2
import numpy as np
from matplotlib import pyplot as plt

camera = cv2.VideoCapture(0)

# img = cv2.imread("test_capture1.png",0)
# edges = cv2.Canny(img,100,200)

"""
while(True):
	cv2.imshow('name',edges)
	cv2.waitKey(0)
"""

# plt.ion()

while(True):
	ret, frame = camera.read()
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(img,100,200)

	"""
	plt.subplot(121),plt.imshow(img,cmap='gray')
	plt.title('Original Image'),plt.xticks([]),plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap='gray')
	plt.title('Edge Image'),plt.xticks([]),plt.yticks([])

	plt.pause(0.05)
	"""

	cv2.imshow("edges",edges)
	cv2.imshow("original",img)
	if cv2.waitKey(30) >= 0:
		break

camera.release()
cv2.destroyAllWindows()
