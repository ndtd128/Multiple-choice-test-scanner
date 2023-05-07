import cv2
import math
from process import *

def getAnswerSheetInfo(answerSheetImage):
    image = answerSheetImage

    INFO_X_RATIO = 0.52
    INFO_Y_RATIO = 0.09
    INFO_W_RATIO = 0.35
    INFO_H_RATIO = 0.25
    SCORE_W_RATIO = 0.12
    SCORE_H_RATIO = 0.15
    SCORE_X_RATIO = 0.15
    SCORE_Y_RATIO = 0.32
    WIDTH = image.shape[1]
    HEIGHT = image.shape[0]

    INFO_X = round(INFO_X_RATIO * WIDTH)
    INFO_Y = round(INFO_Y_RATIO * HEIGHT)
    INFO_W = round(INFO_W_RATIO * WIDTH)
    INFO_H = round(INFO_H_RATIO * HEIGHT)
    SCORE_W = round(SCORE_W_RATIO * WIDTH)
    SCORE_H = round(SCORE_H_RATIO * HEIGHT)
    SCORE_X = round(SCORE_X_RATIO * WIDTH)
    SCORE_Y = round(SCORE_Y_RATIO * HEIGHT)

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
    answerSheetInfo["candidateNumber"] = rectCon[0]
    answerSheetInfo["testCode"] = rectCon[1]

    return answerSheetInfo




