# This is a sample Python script.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

from findwindows import findwindows
from hough import findlines
from detectors import sift_detector, ORB_detector, BRIEF_detectorBF, Surf_detector, BRIEF_detector, sift_detectorBF, \
    ORB_detectorBF, Surf_detectorBF
from takewindow import getwindows

img = plt.imread('test5.jpg')
temp = plt.imread('test6.jpg')

img = cv2.resize(img, (640, 480), interpolation = cv2.INTER_AREA)
temp = cv2.resize(temp, (640, 480), interpolation = cv2.INTER_AREA)

# If the system throws error, please change these two variables instead
width = 80
length = 80

start = time.time()
windows, tempwindow,tempwindow2, tempwindow3 = getwindows(img, width, length)

imges = findwindows(windows, tempwindow,tempwindow2, tempwindow3,length, True)

x1, x2, x3, x4 = imges.shape
end = time.time()
print("Running time：", end-start)

lines,cdst = findlines(img, True)
cv2.imshow("Source", img)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
cv2.waitKey(0)

for i in range(x1):
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", imges[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

start = time.time()
matchesMask, image1, image2, keypoints_1, keypoints_2, matches = sift_detector(img, temp)
end = time.time()

print("The length of the matches: ", np.sum(matchesMask))
print("Running time", end-start)

draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask,
                   flags=cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatchesKnn(image1, keypoints_1, image2, keypoints_2, matches, None, **draw_params)
plt.imshow(img3, ), plt.show()

# start = time.time()
# image1, image2, keypoints_1, keypoints_2, matches = sift_detectorBF(img, temp)
# end = time.time()
#
# match_img = cv2.drawMatches(image1, keypoints_1, image2, keypoints_2, matches[:330], None,
#     matchColor=(0,255,0), singlePointColor=(0,0,255))
# plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
# plt.show()
#
# print("Running time", end-start)

# for i in range(x1):
#     matchesMask, image1, image2, keypoints_1, keypoints_2, matches = sift_detector(imges[i], temp)
#
#     draw_params = dict(matchColor=(0, 255, 0),
#                        singlePointColor=(255, 0, 0),
#                        matchesMask=matchesMask,
#                        flags=cv2.DrawMatchesFlags_DEFAULT)
#
#     img3 = cv2.drawMatchesKnn(image1, keypoints_1, image2, keypoints_2, matches, None, **draw_params)
#     plt.imshow(img3, ), plt.show()


