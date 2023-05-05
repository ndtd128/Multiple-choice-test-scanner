import cv2
import numpy as np
import os
from utils import *
from GradedAnswerSheet import *

height = 559
width = 393

def preprocess(img):
    img = cv2.resize(img, (width, height))

    imgContours = img.copy()
    imgRectCon = img.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 50)

    # Finding all contours
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    imgContours = cv2.drawContours(imgContours, contours, -1, (0, 255, 0)[::-1], 1)

    # Find rects
    rectCon = rect_contour(contours)
    imgRectCon = cv2.drawContours(imgRectCon, rectCon, -1, (0, 255, 0)[::-1], 1)

    # imgArray = [img, imgGray, imgBlur, imgCanny, imgContours]
    # imgArray = [img, imgContours]
    imgArray = [imgContours, imgRectCon]
    imgStack = stack_images(1.4, imgArray)
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
    # Cách test xuất CSV: Comment dòng 37-45 và bỏ comment dòng 53-59.
    # Hai folder reports và results sẽ được tạo ra.
    # Folder reports chứa folder con có tên của bài kiểm tra. Trong đó chứa CSV report của từng mã đề.
    # Folder results chứa folder con có tên của bài kiểm tra. Trong đó chứa ảnh chấm điểm bài làm của từng SBD.

    # resultImage = img
    # gradedAnswerSheet1 = GradedAnswerSheet(21020625, 123, 10, resultImage)
    # gradedAnswerSheet2 = GradedAnswerSheet(21020626, 321, 9, resultImage)
    # gradedAnswerSheet3 = GradedAnswerSheet(21020627, 123, 9, resultImage)
    # gradedAnswerSheets.append(gradedAnswerSheet1)
    # gradedAnswerSheets.append(gradedAnswerSheet2)
    # gradedAnswerSheets.append(gradedAnswerSheet3)

    return resultImage