import cv2
import numpy as np

cap = VideoCapture(0)

while(True)
   ret, frame = cap.read()
   hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   median = cv2.medianBlur(hsv, 3)
   
   
