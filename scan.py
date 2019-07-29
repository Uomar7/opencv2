import cv2
import numpy as np

im = cv2.imread('sheet.jpg', cv2.IMREAD_GRAYSCALE)

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10;
params.maxThreshold = 200

params.filterByCircularity = True
params.minCircularity = 0.4

ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
    detector = cv2.SimpleBlobDetector(params)
else:
    detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(im)

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]),(80,100,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('keypoints', im_with_keypoints)
cv2.waitKey(0)