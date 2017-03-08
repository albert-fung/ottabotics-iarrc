import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):

   ret, frame = cap.read()
   picture = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   detected_circle = cv2.HoughCircles(picture, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
   
   for i in detected_circle[0, :]:
      cv2.circle(hsv, (i[0], i[1]), i[2], (0, 255, 0), 1)


   cv2.imshow("detected circles", picture)
   
   k = cv2.waitKey(5) &0xFF
   if k == 27:
      break
cap.release()
cv2.destroyAllWindows()
   




