import cv2
import numpy as np
import math
import pickle

def nothing(x):
    pass

def embedding(image):
    '''
    Takes an image, prompts user to draw a region of interest(ROI) 
    and returns a tuple with integer coordinates of its corners
    '''
    im = cv2.imread(image)
    rect = cv2.selectROI(im)
    corners = (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]))

    return corners

def corner_detect(image_path, embedding_coordinates, threshold):
    img_bgr = cv2.imread(image_path)  # image_path is a string variable
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    embedding = cv2.imread(image_path, 0)  # in cv2 images are stored as numpy ndarray

    (x, y, width, height) = embedding_coordinates  # list input taken from argument
    embedding = embedding[y:y + height, x:x + width]
    embeddings = [np.rot90(embedding.copy(), (i)) for i in range(4)]

    def template_matching(img_bgr, image_gray, embedding):
        # Slicing: [start:stop:step]. If step is negative, the list is reversed.
        # w, h = embedding.shape[::-1]
        w, h = embedding.shape  # new version. I don't understand why the list used to be reversed for w, h

        # ndarray.shape function returns dimensions as a list(width, height, number of channels (in order)).
        res = cv2.matchTemplate(image_gray, embedding, cv2.TM_CCOEFF_NORMED)  # returns an image

        # with match quality as pixel value. Inputs are the main image, embedding and method
        # methods are 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED' (used for thesis), 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
        loc = np.where(res >= threshold)  # np.where returns two arrays containing coordinates of x, y
        points = zip(*loc[::-1])  # zip takes lists with equal numbers of elements, combines their members
        # "*" operator converts the operation to unzip

        corner_count = 0
        points_filtered = []
        mask = np.zeros(img_bgr.shape[:2], np.uint8)  # mask is used to filter out duplicate matches
        for pt in points:
            if mask[pt[1] + int(h / 2), pt[0] + int(w / 2)] != 255:
                mask[pt[1]:pt[1] + h, pt[0]:pt[0] + w] = 255
                corner_count += 1

                cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (127, 127, 200), 2)
                pt_centered = (int(pt[0] + w / 2), int(pt[1] + h / 2))
                points_filtered.append(pt_centered)
        return [img_bgr, points_filtered]

    return [template_matching(img_bgr, img_gray, embedding) for embedding in embeddings]

def distance(point_a, point_b):
    '''
    Returns euclidean distance from tuple of two points
    '''
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

def brick_lister(unsorted_corners):
    bricks = []  # list of lists with 4 items for brick corners
    brick = []  # list wirh 4 corner coordinates, 2 element tuples each
    pts_ul, pts_ll, pts_lr, pts_ur = unsorted_corners[0], unsorted_corners[1], unsorted_corners[2], unsorted_corners[3]
    # corner point lists
    pix_thresh = 20  # length in pixels for horizontal/vertical matching *default = 50

    i = 0
    for point_ul in pts_ul:
        brick.append(point_ul)

        pix_dist = 500
        for point_ll in pts_ll:
            # iterate over the list of lower left points to locate the one that is closest to the upper left corner, but sits lower
            # proximity to a similar vertical axis and relative placement
            if abs(point_ll[0] - brick[0][0]) < pix_thresh and point_ll[1] > brick[0][1]:
                if distance(brick[0], point_ll) < pix_dist:
                    pix_dist = distance(brick[0], point_ll)
                    if len(brick) == 2:
                        del brick[1]
                    brick.append(point_ll)

        pix_dist = 500
        for point_ur in pts_ur:
            # proximity to a similar horizontal axis and relative placement
            if abs(point_ur[1] - brick[0][1]) < pix_thresh and point_ur[0] > brick[0][0]:
                if distance(brick[0], point_ur) < pix_dist:
                    pix_dist = distance(brick[0], point_ur)
                    if len(brick) == 3:
                        del brick[2]
                    brick.append(point_ur)

        pix_dist = 500
        for point_lr in pts_lr:
            # proximity to a similar horizontal axis and relative placement
            try:
                if abs(point_lr[1] - brick[1][1]) < pix_thresh and point_lr[0] > brick[1][0]:
                    if distance(brick[1], point_lr) < pix_dist:
                        pix_dist = distance(brick[1], point_lr)
                        if len(brick) == 2:
                            pass
                        else:
                            if len(brick) == 4:
                                del brick[3]
                            brick.append(point_lr)
            except:
                print("this used to be an error situation")
                pass

        if len(brick) == 4:
            brick[2], brick[3] = brick[3], brick[2]  # change point order for drawing
            bricks.append(brick.copy())
        brick.clear()
        i += 1
        if i > 500:  # max number of bricks
            break
    return bricks

def brick_draw(image, brick):
    pts = np.array([corner for corner in brick])
    # old method
    # cv2.polylines(image, [pts], True, (0, 0, 255), 2)
    brickAttr = ()

    area = cv2.contourArea(pts)
    if 2000 < area < 25000:
        cv2.drawContours(image, [pts], 0, (200, 127, 0), 2)
        ((x, y), r) = cv2.minEnclosingCircle(pts)
        # cv2.circle(imgray, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.putText(image, str(area), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        brickAttr = {"center_point": (int(x), int(y)), "area": area}
    return brickAttr

# Main
def find_emb(path):
    emb_coord = embedding(path)

    cv2.namedWindow("frame")
    cv2.createTrackbar("test", "frame", 85, 100, nothing)

    prev_test = 0
    while True:
        test = cv2.getTrackbarPos("test", "frame")

        if test != prev_test:
            thr = float(test) * 0.01
            frame = corner_detect(path, emb_coord, thr)[0][0]

            points_corners = [corner_detect(path, emb_coord, thr)[x][1] for x in range(4)]

            bricks = brick_lister(points_corners)
            brick_attr = []
            for brick in bricks:
                brick_attr.append(brick_draw(frame, brick))

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
    return brick_attr

# Main
def emb_val(path, brick_attr):

    im = cv2.imread(path)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 2000 < area < 25000:
            ((x, y), r) = cv2.minEnclosingCircle(contour)

            center_coord = (int(x), int(y))
            try:
                for attr in brick_attr:
                    # print(attr["center_point"])
                    # print(center_coord)
                    if distance(attr["center_point"], center_coord) < 150:  # there is an issue with this parameter
                        print("match_found")
                        cv2.putText(imgray, ("area = " + str(int(area))), (int(x) - 50, int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (127, 127, 127), 2)
                        error = int(1000 * (1 - (area / attr["area"])))
                        cv2.putText(imgray, "error = " + (str(error / 10)) + "%", (int(x) - 50, int(y) + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (127, 127, 127), 2)
            except:
                print("coordinate error")
                pass


    imgray = cv2.cvtColor(imgray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(imgray, contours, -1, (0, 255, 0), 2)

    while True:
        cv2.imshow('frame', imgray)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.waitKey(0)
    cv2.destroyAllWindows()
