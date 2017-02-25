import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
	ret, frame = camera.read()
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(img,50,150,apertureSize=3)

	lines = cv2.HoughLines(edges,1,np.pi/180,200)
	for rho,theta in lines[0]:
		print rho
		print theta
		a = np.cos(theta)
		b.np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

	cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

	cv2.imshow("adf",img)

	if cv2.waitKey(30) >= 0:
		break

camera.release()
cv2.destroyAllWindows()
