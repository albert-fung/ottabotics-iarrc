import cv2
import numpy as np

#Bounds for colors detection 
lower_green= np.array([30,70,90])
upper_green=np.array([90,130,150])

lower_yellow=np.array([0,90,160])
upper_yellow=np.array([75,220,300])

lower_blue=np.array([120,130,10])
upper_blue=np.array([160,170,80])

#starting camera up -- 0 is webcam , 1 is a usb port 
cap=cv2.VideoCapture(1)

while True:
    ##green left lane
    ##Allowing the information from the camera to be a variable
    ret , frame = cap.read()
    ## creating a region of interest for a 640:640 frame 
    frame = frame[200:500,0:640]
    ##imageprocessing 
    frameblur = cv2.GaussianBlur(frame,(3,3),0)
    HSV=cv2.cvtColor(frameblur,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(HSV,lower_green,upper_green)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    
    ## finding edges 
    maskedges=cv2.Canny(res,0,200,apertureSize=3)
    #Cutting away the small edges while increasing the edges that are visable
    opening=cv2.morphologyEx(maskedges,cv2.MORPH_OPEN,(15,15))


    #Showing the image
    cv2.imshow("frame",frame)
    cv2.imshow("res",res)
    cv2.imshow("mask",mask)
    cv2.imshow("maskedges",maskedges)
    #cv2.imshow("opening",opening)

    #variables for houghlines
    minLineLength = 100
    maxLineGap= 7
    #The P at the end of HoughLines causes the output to become (x1,y1,x2,y2) instead of ro and phi 
    greenlines = cv2.HoughLinesP(maskedges,1,np.pi/180,15,minLineLength,maxLineGap)


    #Variables for calculating the lines
    slope= []
    lowestpoint=[0,0,0,0]
    highestpoint=[640,640,640,640]
    totalslope=0

    #Dont run if there are no lines detected 
    if greenlines is not None:
        
        for line in greenlines:
            for x1,y1,x2,y2 in line:
                ##calculating slopes of each line 
                slopeofline=((y1-y2)/(x1-x2))
                slope.append(slopeofline)
                ##calculating a total slope 
                totalslope=totalslope+slopeofline
                ##Figuring out the lowest point detected and the highest point detected 
                if y1 > lowestpoint[1] :
                    lowestpoint[0]=x1
                    lowestpoint[1]=y1
                if y1 < highestpoint[1]:
                    highestpoint[0] =x1
                    highestpoint[1]=y1
        #You can use this to see each hough lines drawn
        #for x in range(0,len(greenlines)):
            #for x1,y1,x2,y2 in greenlines[x]:
                #hough=cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    #cv2.imshow("hough",hough)

    #Validation that there is a line to be drawn              
    if totalslope > 0 and len(slope) > 0:
        avgslope=totalslope/len(slope)
        
        print(avgslope)
        line=cv2.line(frame,(lowestpoint[0],lowestpoint[1]),(highestpoint[0],highestpoint[1]),(255,0,0),5)
        cv2.imshow("line",line)
    if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()




