import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.cvtColor(cv2.imread(r"C:\Users\BKY\Downloads\searchImage.png"),cv2.COLOR_BGR2GRAY)          # queryImage
img2 = cv2.cvtColor(cv2.imread(r"D:\findPlan images\positiv\flucht-und-rettungsplan.jpg"),cv2.COLOR_BGR2GRAY) # trainImage

# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp2 = orb.detect(img2,None)

# compute the descriptors with ORB
kp1, des1 = orb.compute(img1, kp1)
kp2, des2 = orb.compute(img1, kp2)

# find the keypoints and descriptors with SIFT
#kp1, des1 = sift.detectAndCompute(img1,None)
#kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
#-- Draw matches
img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img_matches,flags=2)



plt.imshow(img3),plt.show()