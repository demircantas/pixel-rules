import cv2
import pickle

import pr_utility as pru

path = 'sourceimages/test_binary.png'
emb_coord = pru.embedding(path)
cv2.namedWindow("frame")
cv2.createTrackbar("test", "frame", 85, 100, pru.nothing)

prev_test = 0
while True:
    test = cv2.getTrackbarPos("test", "frame")

    if test != prev_test:
        thr = float(test) * 0.01
        frame = pru.corner_detect(path, emb_coord, thr)[0][0]

        points_corners = [pru.corner_detect(path, emb_coord, thr)[x][1] for x in range(4)]
        # print("corners " + str(len(points_corners)))

        bricks = pru.brick_lister(points_corners)
        brick_attr = []
        for brick in bricks:
            brick_attr.append(pru.brick_draw(frame, brick))

        print(len(brick_attr))
        for attr in brick_attr:
            print(attr)

        prev_test = test

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)
    if key == 27:  # this is the "esc" key
        break
    elif key == 32:  # spacebar to save image
        with open("pickle/bricks.pkl", "wb") as f:
            f.write(pickle.dumps(bricks, True))
        

cv2.waitKey(0)
cv2.destroyAllWindows()

# validation

im = cv2.imread(path)
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# area_sum = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if 2000 < area < 25000:
        ((x, y), r) = cv2.minEnclosingCircle(contour)
        # cv2.circle(imgray, (int(x), int(y)), int(r), (0, 255, 0), 2)

        center_coord = (int(x), int(y))
        try:
            for attr in brick_attr:
                # print(attr["center_point"])
                # print(center_coord)
                if pru.distance(attr["center_point"], center_coord) < 150:  # there is an issue with this parameter
                    print("match_found")
                    cv2.putText(imgray, ("area = " + str(int(area))), (int(x) - 50, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (127, 127, 127), 2)
                    error = int(1000 * (1 - (area / attr["area"])))
                    cv2.putText(imgray, "error = " + (str(error / 10)) + "%", (int(x) - 50, int(y) + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 127, 127), 2)
        except:
            print("coordinate error")
            pass

    # area_sum = area_sum + area
# print("total area equals " + str(area_sum))

imgray = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(imgray, contours, -1, (0, 255, 0), 2)

while True:
    cv2.imshow('frame', imgray)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
