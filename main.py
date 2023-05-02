import cv2
import numpy as np
from utils import *

#################
path = "images/mcq.png"
height = 559
width = 393
#################

# Preprocessing
img = cv2.imread(path)
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

# Test
# imgArray = [img, imgGray, imgBlur, imgCanny, imgContours]
# imgArray = [img, imgContours]
imgArray = [imgContours, imgRectCon]
imgStack = stack_images(1.4, imgArray)
cv2.imshow("stacked image", imgStack)
cv2.waitKey(0)