import cv2
import numpy

cap = cv2.VideoCapture(0)
#green = numpy.uint8([[[0, 255, 0]]])
#hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
#print hsv_green

while(True):
   ret, frame = cap.read()
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   median = cv2.medianBlur(hsv, 7)
 

   upper_green = numpy.array([80, 250, 205])
   lower_green = numpy.array([40, 100, 80])
   
   hsv_mask = cv2.inRange(hsv, lower_green, upper_green)
   median_mask = cv2.inRange(median, lower_green, upper_green)
   

   cv2.imshow("frame", frame)
   cv2.imshow("no blurring", hsv_mask)
   cv2.imshow("with blurring", median_mask)
   
   k = cv2.waitKey(5) &0xFF
   if k == 27:
      break
cap.release()
cv2.destroyAllWindows()
