# # import the necessary packages
# from __future__ import print_function
# from skimage.feature import peak_local_max
# from skimage.morphology import watershed
# from scipy import ndimage
# import argparse
# import imutils
# import cv2
#
# import numpy as np
#
# # construct the argument parse and parse the arguments
# # ap = argparse.ArgumentParser()
# # ap.add_argument("-i", "--image", required=False,
# #                 help="path to input image")
# # args = vars(ap.parse_args())
#
# # load the image and perform pyramid mean shift filtering
# # to aid the thresholding step
# path = 'images/recognition/sircali_buyuk_height1' + "_ADAPTIVE_THRESHOLD_" + ".png"
# image = cv2.imread(path)
# # shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
# # cv2.imshow("Input", image)
#
# # convert the mean shift image to grayscale, then apply
# # Otsu's thresholding
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255,
# 	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#
# # kernel = np.ones((10, 10), np.float32) /225
# # thresh = cv2.filter2D(thresh, -1, kernel)
#
# # thresh = cv2.GaussianBlur(thresh, (15, 15), 0)
# # thresh = cv2.medianBlur(thresh, 15)
# # thresh = cv2.bilateralFilter(thresh, 15, 75, 75)
#
# cv2.imshow("Thresh", thresh)
#
# # find contours in the thresholded image
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
#                         cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# print("[INFO] {} unique contours found".format(len(cnts)))
#
# # loop over the contours
# for (i, c) in enumerate(cnts):
#     # draw the contour
#     ((x, y), _) = cv2.minEnclosingCircle(c)
#     cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
#
# # show the output image
# cv2.imshow("Image", image)
# cv2.waitKey(0)import numpy as np
import cv2

im = cv2.imread('images/recognition/ince_kose_height' + "_ADAPTIVE_THRESHOLD_" + ".png")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# area_sum = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if 2000< area< 25000:
        ((x, y), r) = cv2.minEnclosingCircle(contour)
        #cv2.circle(imgray, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(imgray, str(area), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # area_sum = area_sum + area
# print("total area equals " + str(area_sum))

imgray = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(imgray, contours, -1, (0,255,0), 2)

while True:
    cv2.imshow('frame', imgray)

    key = cv2.waitKey(1)
    if key == 27: #  'esc' key
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
