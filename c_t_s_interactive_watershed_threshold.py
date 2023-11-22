import cv2

# im = cv2.imread('images/recognition/ince_kose_height' + "_ADAPTIVE_THRESHOLD_" + ".png")
im = cv2.imread("sourceimages/test_binary.png")
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
