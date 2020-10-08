import cv2
import numpy as np

def nothing(x):
    pass
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    #  This is done using a 1D NumPy array (table)
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def ad_th(image, p1 = 115, p2 = 1):
    grayscaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_bgr = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, p1, p2)
    return img_bgr

examples = ["sircali_buyuk_height1", "ince_kose_height"]

path_strip = 'images/recognition/' + examples[1]
path = path_strip + ".png"
path_wr = path_strip + "_ADAPTIVE_THRESHOLD_" + ".png"
cv2.namedWindow("frame")
cv2.createTrackbar("threshold param1", "frame", 115, 200, nothing)
cv2.createTrackbar("threshold param2", "frame", 1, 5, nothing)
cv2.createTrackbar("gamma", "frame", 100, 300, nothing)
cv2.createTrackbar("filter", "frame", 0, 100, nothing)

while True:
    p1a = cv2.getTrackbarPos("threshold param1", "frame")
    p1 = (int(p1a / 2)) * 2 + 3
    p2 = cv2.getTrackbarPos("threshold param2", "frame")
    gamma = float(cv2.getTrackbarPos("gamma", "frame")) / 100
    filter = int(cv2.getTrackbarPos("filter", "frame") /2) * 2 +1

    frame = cv2.imread(path)
    frame = adjust_gamma(frame, gamma)
    frame = ad_th(frame, p1, p2)

    # kernel = np.ones((15, 15), np.float32) / 225
    # frame = cv2.filter2D(frame, -1, kernel)

    kernel = np.ones((filter,filter), np.uint8)
    # frame = cv2.erode(frame, kernel, iterations=1)
    # frame = cv2.dilate(frame, kernel, iterations=1)
    frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)

    # frame = cv2.GaussianBlur(frame, (15, 15), 0)
    # frame = cv2.medianBlur(frame, filter)
    # frame = cv2.bilateralFilter(frame, 15, 75, 75)

    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, ("p1 = " + str(p1a) + ", p2 = " + str(p2) + ", gamma = " +str(gamma)), (50, 50), font, 1, (0,0,255))

    cv2.imshow('frame', frame)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == 32:  # spacebar to save image
        cv2.imwrite(path_wr, frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
