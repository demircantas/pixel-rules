from typing import Tuple

import cv2
import numpy as np
import math
import pickle

def embedding(image):
    im = cv2.imread(image)
    # Select Region of Interest
    r = cv2.selectROI(im)

    corners = (int(r[0]), int(r[1]), int(r[2]), int(r[3]))

    return corners

def nothing(x):
    pass

def distance(point_a, point_b):
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

def corner_detect(image_path, embedding_coordinates, threshold):
    img_bgr = cv2.imread(image_path)  # image_path is a string variable
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    embedding = cv2.imread(image_path, 0)  # in cv2 images are stored as numpy ndarray
    (x, y, width, height) = embedding_coordinates  # list input taken from argument
    embedding = embedding[y:y + height, x:x + width]
    embeddings = [np.rot90(embedding.copy(), (i)) for i in range(4)]

    def template_matching(img_bgr, image_gray, embedding):
        # w, h = embedding.shape[::-1]  # [::-1] is used to reverse the order of a list.
        w, h = embedding.shape  # new version. I don't understand why the list is reversed for w, h
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
        # print(corner_count)
        return [img_bgr, points_filtered]

    return [template_matching(img_bgr, img_gray, embedding) for embedding in embeddings]


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