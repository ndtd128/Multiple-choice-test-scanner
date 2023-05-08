import csv
import os
from GradedAnswerSheet import *

def makeAnswerKeyListFromCSV(csvFilePath, answerKeys):
    with open(csvFilePath, 'r') as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            if row[0] == 'Test code':
                testCode = row[1]
                answerKeys[testCode] = []

            elif row[0].isdigit() and len(row[1]) == 1:
                # Map answer key to int
                key = ord(row[1]) - 65
                answerKeys[testCode].append(key)

def createCSVReport(gradedAnswerSheets, testName):
    testResults = {}
    for index, sheet in enumerate(gradedAnswerSheets):
        if sheet.testCode != "N/A" and sheet.testCode not in testResults:
            testResults[sheet.testCode] = []
            testResults[sheet.testCode].append((sheet.candidateNumber, sheet.score))

    reportsFolder = "reports"
    testFolderPath = os.path.join(reportsFolder, testName)
    if not os.path.exists(testFolderPath):
        os.makedirs(testFolderPath)

    for testCode, results in testResults.items():
        reportFilename = f"test_report_{testCode}.csv"
        reportFilePath = os.path.join(testFolderPath, reportFilename)
        print(reportFilePath)
        with open(reportFilePath, 'w', newline='') as reportFile:
            writer = csv.writer(reportFile)
            writer.writerow(['TEST REPORT'])
            writer.writerow(['Test code', testCode])
            writer.writerow(['Candidate number', 'Score'])
            for result in results:
                writer.writerow(result)
