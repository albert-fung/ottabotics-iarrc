import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while True:
	ret, frame = camera.read()
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	edges = cv2.Canny(img,50,150,apertureSize=3)

	#cv2.imshow("grea",edges)
	#cv2.waitKey(0)

	#print "edges: %s" % edges

	lines = cv2.HoughLines(edges,1,np.pi/180,120)
	
	#print "lines: %s" % lines
	#break	

	#print len(lines)
	#break

	try:
		for line in lines:
			for rho,theta in line:
				print "rho: %s" % rho
				print "theta: %s" % theta
				a = np.cos(theta)
				b = np.sin(theta)
				x0 = a*rho
				y0 = b*rho
				x1 = int(x0 + 1000*(-b))
				y1 = int(y0 + 1000*(a))
				x2 = int(x0 - 1000*(-b))
				y2 = int(y0 - 1000*(a))

			cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

		cv2.imshow("adf",img)
		#cv2.waitKey(0)
		#print "waitkey: %s" % cv2.waitKey(30)
		if cv2.waitKey(30) != 255:
			break
			#print "waitkey triggered!!!"
	except TypeError:
		print "no lines detected"

camera.release()
cv2.destroyAllWindows()
