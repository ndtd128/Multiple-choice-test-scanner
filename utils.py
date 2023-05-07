import cv2
import numpy as np

SIMILAR_RATE = 0.002
def stack_images(scale, img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    rowsAvailable = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]

    if rowsAvailable:
        # resize by scale
        for x in range(0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape[:2]:
                    img_array[x][y] = cv2.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv2.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]),
                                                 None, scale, scale)

                if len(img_array[x][y].shape) == 2:
                    img_array[x][y] = cv2.cvtColor(img_array[x][y], cv2.COLOR_GRAY2BGR)

        # stacking image
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(img_array[x])

        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv2.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv2.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,
                                          scale, scale)

            if len(img_array[x].shape) == 2:
                img_array[x] = cv2.cvtColor(img_array[x], cv2.COLOR_GRAY2BGR)

        hor = np.hstack(img_array)
        ver = hor

    return ver


def rect_contour(contours):
    rectCon = []

    for contour in contours:
        area = cv2.contourArea(contour)
        # print("Area ", area)

        # Filter by area
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

            if len(approx) == 4:
                check = True
                for cnt in rectCon:
                    rate = cv2.matchShapes(contour, cnt, cv2.CONTOURS_MATCH_I1, 0.0)
                    if rate < 0.05:
                        check = False
                        break
                if check:
                    rectCon.append(contour)

    print(len(rectCon))

    # sort descending rectangle contours by area
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)


    return rectCon


def get_corner_points(contour):
    # contour.get

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    return approx


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
    # print(myPoints)
    # print(add)
    myPointNew[0] = myPoints[np.argmin(add)]  # [0,0]
    myPointNew[3] = myPoints[np.argmax(add)]  # [w,h]
    diff = np.diff(myPoints, axis=1)
    myPointNew[1] = myPoints[np.argmin(diff)]  # [w,0]
    myPointNew[2] = myPoints[np.argmax(diff)]  # [h,0]

    return myPointNew


def numberDetection(img):
    imgContours = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur, 20, 35)

    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    imgContours = cv2.drawContours(imgContours, contours, -1, (0, 255, 0)[::-1], 1)

    kernel = np.ones((5, 5), np.uint8)

    # Perform dilation on the image
    dilated = cv2.dilate(imgCanny, kernel, iterations=1)
    cv2.imshow("A", dilated)


def splitBoxes(img):
    a = 1
    # row = np.vsplit(img,6)
    # col =np.hsplit(row[4],5)
    # cv2.imshow("im1", row[1])
    # cv2.imshow("im2", col[2])
    # cv2.imshow("im3", col[3])
    # cv2.imshow("im4", col[4])
