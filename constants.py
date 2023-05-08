import cv2

# HARDCODED ANSWER SHEET HEIGHT AND WIDTH
HEIGHT = 2048
WIDTH = 1466

# ANSWER AREA
ANSWER_WIDTH = 1013
ANSWER_HEIGHT = 934

# INFO AREA
INFO_X_RATIO = 0.52
INFO_Y_RATIO = 0.09
INFO_W_RATIO = 0.35
INFO_H_RATIO = 0.25
SCORE_W_RATIO = 0.12
SCORE_H_RATIO = 0.15
SCORE_X_RATIO = 0.15
SCORE_Y_RATIO = 0.32

INFO_X = round(INFO_X_RATIO * WIDTH)
INFO_Y = round(INFO_Y_RATIO * HEIGHT)
INFO_W = round(INFO_W_RATIO * WIDTH)
INFO_H = round(INFO_H_RATIO * HEIGHT)
SCORE_W = round(SCORE_W_RATIO * WIDTH)
SCORE_H = round(SCORE_H_RATIO * HEIGHT)
SCORE_X = round(SCORE_X_RATIO * WIDTH)
SCORE_Y = round(SCORE_Y_RATIO * HEIGHT)

CANDIDATE_NUMBER_HEIGHT = 446
CANDIDATE_NUMBER_WIDTH = 270

FILLED_THRESHOLD = 400
FILLED_THRESHOLD_2 = 100