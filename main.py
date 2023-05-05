import cv2
import os
from process import *
from GradedAnswerSheet import *
import csvUtils

# Path to answer sheet images and answer key CSV files
answerSheetPath = 'answerSheets'
answerKeyPath = 'answerKeys'

# Path to result folder (= name of test) (sau nay se lay input nguoi dung)
# testName = input("Test name: ").trim()
testName = "midterm"

# Create result folder if not yet existed
resultFolder = os.path.join('results', testName)
if not os.path.exists(resultFolder):
    os.makedirs(resultFolder)

# Push all answer sheet images to a list
answerSheetsImages = []
answerSheetFiles = os.listdir(answerSheetPath)
for filename in answerSheetFiles:
    img = cv2.imread(os.path.join(answerSheetPath, filename))
    answerSheetsImages.append(img)

# Extract answer key for each test code from CSV and push to answerKeys dictionary 
# answerKeys dictionary format: key = test code, value = list of answer keys for that test code
answerKeyFiles = os.listdir(answerKeyPath)
answerKeys = {}
for answerKeyFilePath in answerKeyFiles:
    answerKeyFullFilePath = os.path.join(answerKeyPath, answerKeyFilePath)
    csvUtils.makeAnswerKeyListFromCSV(answerKeyFullFilePath, answerKeys)

# List of graded answer sheets
gradedAnswerSheets = []

# Process each image
for img in answerSheetsImages:
    process(img, gradedAnswerSheets)

# Output image files in result folder
for i in range(len(gradedAnswerSheets)):
    outputPath = os.path.join(resultFolder, f'{gradedAnswerSheets[i].candidateNumber}.jpg')
    cv2.imwrite(outputPath, gradedAnswerSheets[i].resultImage)

# Create CSV report
csvUtils.createCSVReport(gradedAnswerSheets, testName)