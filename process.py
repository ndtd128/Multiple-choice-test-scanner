import cv2
import numpy as np
import os
from utils import *
from GradedAnswerSheet import *
import imutils
from imutils.perspective import four_point_transform
from imutils import contours
from constants import *

def getAnswerList(answerArea):
    imgWarpgray = cv2.cvtColor(answerArea, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(imgWarpgray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    answerList = []
    answerColumns = extractAnswerColumns(thresh)
    for columnIndex, column in enumerate(answerColumns):
        cnts = cv2.findContours(column, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if w >= 20 and h >= 20  and ar >= 0.7 and ar <= 1.3:
                # Check if the contour overlaps with any existing contour
                is_overlapping = False
                (curr_x, curr_y), curr_radius = cv2.minEnclosingCircle(c)
                for existingCnt in questionCnts:
                    (existing_x, existing_y), existing_radius = cv2.minEnclosingCircle(existingCnt)
                    distance = np.sqrt((existing_x - curr_x)**2 + (existing_y - curr_y)**2)
                    if distance < (existing_radius + curr_radius):
                        is_overlapping = True
                        break

                # If the contour is not overlapping with any existing contour, add it to questionCnts
                if not is_overlapping:
                    questionCnts.append(c)
                    
        questionCnts = imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]

        for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
            cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
            bubbled = None
            filledThreshold = 400
            for j, c in enumerate(cnts):
                # construct a mask that reveals only the current
                # "bubble" for the question
                mask = np.zeros(column.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
        
                # apply the mask to the thresholded image, then
                # count the number of non-zero pixels in the
                # bubble area
                mask = cv2.bitwise_and(column, column, mask=mask)
                # cv2.imshow("mask", mask)
                # cv2.waitKey(0)
                total = cv2.countNonZero(mask)
                # if the current total has a larger number of total
                # non-zero pixels, then we are examining the currently
                # bubbled-in answer
                if bubbled is None or total > bubbled[0]:
                    bubbled = (total, j)
            
            if bubbled[0] < filledThreshold:
                answerList.append(-1)
            else:
                answerList.append(bubbled[1])
    
    return answerList

def getTestCode(answerSheetImage):
    answerSheetInfo = getAnswerSheetInfo(answerSheetImage)
    testCodeArea= answerSheetInfo["testCode"]
    
    # Threshold
    imgWarpgray = cv2.cvtColor(testCodeArea, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(imgWarpgray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cv2.imshow("Test code", thresh)
    # cv2.waitKey(0)
    threshH, threshW = thresh.shape[:2]
    crop_width = int(0.25 * threshW)
    crop_height = int(0.1 * threshH)

    cropped_thresh = thresh[crop_height:, crop_width:]

    testCode = ""
    cnts = cv2.findContours(cropped_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
    cv2.drawContours(cropped_thresh, questionCnts, -1, (0, 255, 0)[::-1], 10)

    questionCnts = imutils.contours.sort_contours(questionCnts, method="left-to-right")[0]

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 10)):

        cnts = contours.sort_contours(questionCnts[i:i + 10], method="top-to-bottom")[0]
        bubbled = None
        
        for j, c in enumerate(cnts):
            mask = np.zeros(cropped_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
    
            mask = cv2.bitwise_and(cropped_thresh, cropped_thresh, mask=mask)

            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        testCode += str(bubbled[1])

    return testCode

def getCandidateNumber(answerSheetImage):
    answerSheetInfo = getAnswerSheetInfo(answerSheetImage)
    candidateNumberArea= answerSheetInfo["candidateNumber"]
    imgWarpgray = cv2.cvtColor(candidateNumberArea, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(imgWarpgray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    threshH, threshW = thresh.shape[:2]
    crop_width = int(0.825 * threshW)
    crop_height = int(0.1 * threshH)

    cropped_thresh = thresh[crop_height:threshH-crop_height, threshW - crop_width:]

    candidateNumber = ""
    cnts = cv2.findContours(cropped_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)

        if w >= 20 and h >= 20  and ar >= 0.7 and ar <= 1.3:
            # Check if the contour overlaps with any existing contour
            is_overlapping = False
            (curr_x, curr_y), curr_radius = cv2.minEnclosingCircle(c)
            for existingCnt in questionCnts:
                (existing_x, existing_y), existing_radius = cv2.minEnclosingCircle(existingCnt)
                distance = np.sqrt((existing_x - curr_x)**2 + (existing_y - curr_y)**2)
                if distance < (existing_radius + curr_radius):
                    is_overlapping = True
                    break

            # If the contour is not overlapping with any existing contour, add it to questionCnts
            if not is_overlapping:
                questionCnts.append(c)
    cv2.drawContours(cropped_thresh, questionCnts, -1, (0, 255, 0)[::-1], 10)
    questionCnts = imutils.contours.sort_contours(questionCnts, method="left-to-right")[0]

    for (q, i) in enumerate(np.arange(0, len(questionCnts), 10)):

        cnts = contours.sort_contours(questionCnts[i:i + 10], method="top-to-bottom")[0]
        bubbled = None
        
        for j, c in enumerate(cnts):
            mask = np.zeros(cropped_thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(cropped_thresh, cropped_thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)
        candidateNumber += str(bubbled[1])

    return candidateNumber

def calculateGrade(answerList, answerKeys, testCode):
    correctAnswerList = []
    wrongAnswerList = []
    for index, key in enumerate(answerKeys[testCode]):
        # if answerList[index] == -1:
        #     wrongAnswerList.append(index)
        if answerList[index] == key:
            correctAnswerList.append(index)

        else:
            wrongAnswerList.append(index)
    grade = round((len(correctAnswerList) / float(len(answerKeys[testCode]))) * 10, 2)
    return grade

def getAnswerArea(img):
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
    rectCon = rectContour(contours1)

    answerAreaCorners = getCornerPoints(rectCon[1])

    cv2.drawContours(imgSelectedCon, answerAreaCorners, -1, (0, 255, 0)[::-1], 10)

    imgWarpColored = four_point_transform(img, answerAreaCorners.reshape(4, 2))

    # imgArray = [imgContours, imgSelectedCon, imgWarpColored]
    # imgStack = stackImages(0.3, imgArray)
    # cv2.imshow("stacked image", imgStack)
    # cv2.waitKey(0)

    return imgWarpColored

def getResult(answerArea, answerKeys, testCode, answerList, grade ,img):
    count = 0

    #Find answer area and store its location
    location_info = []
    answerColumns = extractAnswerColumns(answerArea)
    for columnIndex, column in enumerate(answerColumns):
        w = column.shape[1]
        h = column.shape[0]
        res = cv2.matchTemplate(img, column, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        location_info.append([top_left[1], top_left[1] + h, top_left[0], top_left[0] + w])
        print(location_info[columnIndex])

    imgWarpgray = cv2.cvtColor(answerArea, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(imgWarpgray, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    answerColumns = extractAnswerColumns(thresh)
    for columnIndex, column in enumerate(answerColumns):
        cnts = cv2.findContours(column, cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)

            if w >= 20 and h >= 20  and ar >= 0.7 and ar <= 1.3:
                # Check if the contour overlaps with any existing contour
                is_overlapping = False
                (curr_x, curr_y), curr_radius = cv2.minEnclosingCircle(c)
                for existingCnt in questionCnts:
                    (existing_x, existing_y), existing_radius = cv2.minEnclosingCircle(existingCnt)
                    distance = np.sqrt((existing_x - curr_x)**2 + (existing_y - curr_y)**2)
                    if distance < (existing_radius + curr_radius):
                        is_overlapping = True
                        break

                # If the contour is not overlapping with any existing contour, add it to questionCnts
                if not is_overlapping:
                    questionCnts.append(c)
                    
        questionCnts = imutils.contours.sort_contours(questionCnts, method="top-to-bottom")[0]

        #Fill answers with color
        #Green: Key answer
        #Red: False answer
        #Blue: No answer
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 4)):
            cnts = contours.sort_contours(questionCnts[i:i + 4])[0]
            cv2.drawContours(img[location_info[columnIndex][0]:location_info[columnIndex][1], location_info[columnIndex][2]:location_info[columnIndex][3]], 
                             cnts, answerKeys[testCode][count], color=(50, 193, 99), thickness=cv2.FILLED)
            if (answerList[count] != answerKeys[testCode][count] and answerList[count] != -1):
                cv2.drawContours(img[location_info[columnIndex][0]:location_info[columnIndex][1], location_info[columnIndex][2]:location_info[columnIndex][3]], 
                             cnts, answerList[count], color=(80,127,255), thickness=cv2.FILLED)
            elif (answerList[count] == -1):
                cv2.drawContours(img[location_info[columnIndex][0]:location_info[columnIndex][1], location_info[columnIndex][2]:location_info[columnIndex][3]], 
                             cnts, answerKeys[testCode][count], color=(208,224,64), thickness=cv2.FILLED)
            count += 1

        cv2.putText(img[SCORE_Y + round(SCORE_H/12):SCORE_Y+SCORE_H, SCORE_X:SCORE_X+SCORE_W], str(grade), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
    # cv2.imshow("filled", img)
    # cv2.waitKey()
    return img


def process(img, answerKeys, gradedAnswerSheets):
    answerArea = getAnswerArea(img)
    answerList = getAnswerList(answerArea)
    candidateNumber = getCandidateNumber(img)
    testCode = getTestCode(img)
    grade = calculateGrade(answerList, answerKeys, testCode)
    print("Candidate number: " + candidateNumber)
    print("Test code: " + testCode)
    print("Grade: " + str(grade))
    print("Answer list: ", answerList)

    # TODO: Create result image (ban Hung lam nhe)
    resultImage = img

    # Create new object of class GradedAnswerSheet having the above info
    gradedAnswerSheet = GradedAnswerSheet(candidateNumber, testCode, grade, resultImage, answerList)
    gradedAnswerSheets.append(gradedAnswerSheet)
    getResult(answerArea, answerKeys, testCode, answerList, grade, img)

    return resultImage
