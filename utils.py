import cv2
import numpy as np
import math
from constants import *
import imutils
from imutils.perspective import four_point_transform

def stackImages(scale, img_array):
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


def rectContour(contours):
    rectCon = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

            if len(approx) == 4:
                rectCon.append(contour)

    # Sort descending rectangle contours by area
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)

    uniqueRectCon = []
    uniqueRectCon.append(rectCon[0])
    for i in range(1, len(rectCon)):
        if cv2.contourArea(rectCon[i]) / cv2.contourArea(rectCon[i - 1]) > 1.05 or cv2.contourArea(rectCon[i]) / cv2.contourArea(rectCon[i - 1]) < 0.95:
            uniqueRectCon.append(rectCon[i])

    return uniqueRectCon

def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    return approx

# Deprecated
def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    myPointNew= np.zeros((4,1,2),np.int32)
    add = myPoints.sum(1)
    # print(myPoints)
    # print(add)
    myPointNew[0]  = myPoints[np.argmin(add)] # [0,0]
    myPointNew[3] = myPoints[np.argmax(add)]   #[w,h]
    diff = np.diff(myPoints,axis=1)
    myPointNew[1] = myPoints[np.argmin(diff)]  # [w,0]
    myPointNew[2] = myPoints[np.argmax(diff)]  # [h,0]
    return myPointNew

# Deprecated
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
    cv2.imshow("A",dilated)

def extractAnswerColumns(answerRegion):
    # Load the image
    image = answerRegion.copy()

    # Define the coordinates of the ROI (x, y, width, height)
    w = math.ceil(image.shape[1] * 0.78)
    h = math.ceil(image.shape[0] * 0.85)
    x = round((image.shape[1] - w) / 2)-10
    y = round((image.shape[0] - h) / 2)

    # Crop the image based on the ROI coordinates
    roi = image[y:y+h, x:x+w-15]
    roi_resized = cv2.resize(roi, (0,0), fx=0.8, fy=0.8)

    # Calculate the width of each column
    column_width = math.ceil(w / 3)

    # Define the width to be cropped from each column
    crop_width = math.ceil(column_width / 6)

    # Extract column 1 and crop the first 1/6 width
    column1 = roi[:, :column_width]
    column1_cropped = column1[:, crop_width:]

    # Extract column 2 and crop the first 1/6 width
    column2 = roi[:, column_width:2*column_width]
    column2_cropped = column2[:, crop_width:]

    # Extract column 3 and crop the first 1/6 width
    column3 = roi[:, 2*column_width:]
    column3_cropped = column3[:, crop_width:]

    croppedColumns = [column1_cropped, column2_cropped, column3_cropped]
    return croppedColumns

def getAnswerSheetInfo(answerSheetImage):
    image = answerSheetImage
    # Crop "info" region
    info = image[INFO_Y:INFO_Y+INFO_H, INFO_X:INFO_X+INFO_W]

    # Crop "score" region
    score = image[SCORE_Y:SCORE_Y+SCORE_H, SCORE_X:SCORE_X+SCORE_W]

    answerSheetInfo = {}

    imgGray = cv2.cvtColor(info, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur, 20, 50)

    contours1, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rectCon = rectContour(contours1)

    candNumberCorners = getCornerPoints(rectCon[0])
    candNumberArea = four_point_transform(info, candNumberCorners.reshape(4, 2))

    testCodeCorners = getCornerPoints(rectCon[1])
    testCodeArea = four_point_transform(info, testCodeCorners.reshape(4, 2))

    answerSheetInfo["infoImage"] = info
    answerSheetInfo["candidateNumber"] = candNumberArea
    answerSheetInfo["testCode"] = testCodeArea

    return answerSheetInfo

def getBubbles(thresholdedImage):
    cnts = cv2.findContours(thresholdedImage, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if w >= 20 and h >= 20  and ar >= 0.7 and ar <= 1.3:
            is_overlapping = False
            (curr_x, curr_y), curr_radius = cv2.minEnclosingCircle(c)
            for existingCnt in questionCnts:
                (existing_x, existing_y), existing_radius = cv2.minEnclosingCircle(existingCnt)
                distance = np.sqrt((existing_x - curr_x)**2 + (existing_y - curr_y)**2)
                if distance < (existing_radius + curr_radius):
                    is_overlapping = True
                    break

            if not is_overlapping:
                questionCnts.append(c)
    return questionCnts
