import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

img1 = cv2.imread('card_reference_resized.jpg',0)          # queryImage
img2 = cv2.imread('card_query_5_resized.jpg',0) # trainImage
img2_colour = cv2.imread('card_query_5_resized.jpg',1)
stoplight_colour = cv2.imread('stoplight.jpg',1)
stoplight = cv2.imread('stoplight.jpg',0)
stoplight_out = cv2.cvtColor(stoplight,cv2.COLOR_GRAY2BGR)

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    print "Sufficient good matches found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    roi_list_of_coords = dst.ravel().tolist()
    print(roi_list_of_coords)
    for i,pnt in enumerate(roi_list_of_coords):
        roi_list_of_coords[i] = int(pnt)
    print(roi_list_of_coords)
    x_list = []
    y_list = []
    for i in range(0,len(roi_list_of_coords),2):
        x_list.append(roi_list_of_coords[i])
    for i in range(1,len(roi_list_of_coords),2):
        y_list.append(roi_list_of_coords[i])
    print("xlist:", x_list)
    print("ylist:", y_list)

    print("min x: %d\nmax x: %d\nmin y: %d\nmax y: %d\n" % (min(x_list),max(x_list),min(y_list),max(y_list)))

    img_roi = img2_colour[min(y_list):max(y_list),min(x_list):max(x_list)]

    cv2.imshow("agr",img_roi)
    cv2.waitKey(0)

    circles = cv2.HoughCircles(stoplight, cv2.HOUGH_GRADIENT, 1.2, 20)
    print(circles)
    circles = np.uint16(np.around(circles))
    print("Number of circles found: %d" % (len(circles[0,:])))
    for i in circles[0,:]:
        cv2.circle(stoplight_out,(i[0],i[1]),i[2],(255,0,0),2)
        # cv2.circle(stoplight_out,(i[0],i[1]),2,(0,0,255),3)

        print("Center point: (%d,%d)" % (i[0],i[1]))

        cv2.rectangle(stoplight_out,(i[0]-25,i[1]-25),(i[0]+25,i[1]+25),(0,255,0))


        print(stoplight_out[0][0][0])


        center_colour = stoplight_colour[i[1]][i[0]].ravel().tolist()

        cv2.circle(stoplight_out,(i[0],i[1]),2,(center_colour[0],center_colour[1],center_colour[2]),30)

        print("Colour value at (%d,%d): %s" % (i[0],i[1],str(center_colour)))

    cv2.imshow("stoplight",stoplight_out)
    cv2.waitKey(0)





    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    cv2.imshow("podfe",img2)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()

cv2.imshow("adsff",img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

