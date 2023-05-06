import cv2
import numpy as np
import os
from utils import *
from GradedAnswerSheet import *
import imutils
from imutils import contours

height = 1021
width = 937


def preprocess(img):
    img = cv2.resize(img, (width, height))

    imgContours = img.copy()
    imgRectCon = img.copy()
    imgSelectedCon = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur, 20, 50)

    # Finding all contours
    contours1, hierarchy = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = cv2.drawContours(imgContours, contours1, -1, (0, 255, 0)[::-1], 1)

    # Find rects
    rectCon = rect_contour(contours1)
    contour1 = get_corner_points(rectCon[2])
    # contour2 = get_corner_points(rectCon[1])
    # contour3 = get_corner_points(rectCon[2])
    # contour4 = get_corner_points(rectCon[3])


    cv2.drawContours(imgSelectedCon, contour1, -1, (0, 255, 0)[::-1], 10)
    # cv2.drawContours(imgSelectedCon, contour2, -1, (0, 0, 255)[::-1], 10)
    # cv2.drawContours(imgSelectedCon, contour3, -1, (255, 0, 0)[::-1], 10)
    # cv2.drawContours(imgSelectedCon, contour4, -1, (0, 255, 255)[::-1], 10)
    # cv2.drawContours(imgSelectedCon, rectCon, -1, (0, 255, 255)[::-1], 1)

    biggestContour = reorder(contour1)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (width, height))

    # Threshold
    imgWarpgray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpgray, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # splitBoxes(imgWarpColored)
    thresh = cv2.threshold(imgWarpgray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= 1 and h >= 1  and ar >= 0.8 and ar <= 1.1:
            questionCnts.append(c)
        cv2.drawContours(imgWarpColored, questionCnts, -1, (0, 255, 0)[::-1], 3)

        # questionCnts = imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        # correct = 0
        # for (q, i) in enumerate(np.arange(0, len(questionCnts), 6)):
        #     # sort the contours for the current question from
        #     # left to right, then initialize the index of the
        #     # bubbled answer
        #     cnts = imutils.contours.sort_contours(questionCnts[i:i + 5])[0]
        #     bubbled = None
        #
        #     for (j, c) in enumerate(cnts):
        #         # construct a mask that reveals only the current
        #         # "bubble" for the question
        #         mask = np.zeros(thresh.shape, dtype="uint8")
        #         cv2.drawContours(mask, [c], -1, 255, -1)
        #
        #         # apply the mask to the thresholded image, then
        #         # count the number of non-zero pixels in the
        #         # bubble area
        #         mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        #         total = cv2.countNonZero(mask)
        #
        #         # if the current total has a larger number of total
        #         # non-zero pixels, then we are examining the currently
        #         # bubbled-in answer
        #         if bubbled is None or total > bubbled[0]:
        #             bubbled = (total, j)


    # imgArray = [img, imgGray, imgBlur, imgCanny, imgContours]
    # imgArray = [img, imgContours]
    imgArray = [imgContours, imgSelectedCon, imgWarpColored]
    imgStack = stack_images(0.5, imgArray)
    cv2.imshow("stacked image", imgStack)
    cv2.waitKey(0)


def process(img, gradedAnswerSheets):
    preprocess(img)

    # TODO: Process image and get candidate number, test code, score and result image
    candidateNumber = None
    testCode = None
    score = None
    resultImage = img

    # Create new object of class GradedAnswerSheet having the above info
    gradedAnswerSheet = GradedAnswerSheet(candidateNumber, testCode, score, resultImage)
    gradedAnswerSheets.append(gradedAnswerSheet)

    # Test 
    # Cách test xuất CSV: 
    # Comment từ #TODO tới trước #Test, bỏ comment từ #Start CSV Test tới #End CSV Test và chạy file main.py
    # Hai folder reports và results sẽ được tạo ra.
    # Folder reports chứa folder con có tên của bài kiểm tra. Trong đó chứa CSV report của từng mã đề.
    # Folder results chứa folder con có tên của bài kiểm tra. Trong đó chứa ảnh chấm điểm bài làm của từng SBD.

    # Start CSV test
    # resultImage = img
    # gradedAnswerSheet1 = GradedAnswerSheet(21020625, 123, 10, resultImage)
    # gradedAnswerSheet2 = GradedAnswerSheet(21020626, 321, 9, resultImage)
    # gradedAnswerSheet3 = GradedAnswerSheet(21020627, 123, 9, resultImage)
    # gradedAnswerSheets.append(gradedAnswerSheet1)
    # gradedAnswerSheets.append(gradedAnswerSheet2)
    # gradedAnswerSheets.append(gradedAnswerSheet3)
    # End CSV test

    return resultImage
