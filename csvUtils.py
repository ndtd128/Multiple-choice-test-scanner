import csv
import os
from GradedAnswerSheet import *
import statistics

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
        if sheet.testCode != "NA" and sheet.testCode not in testResults:
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
            
            writer.writerow([])  # Blank line
            
            # Calculate additional metrics
            scores = [result[1] for result in results]
            averageScore = sum(scores) / len(scores)
            medianScore = statistics.median(scores)
            modeScore = statistics.mode(scores)
            lowestGrade = min(scores)
            highestGrade = max(scores)
            
            # Calculate grade distribution
            gradeDistribution = []
            totalStudents = len(scores)
            for i in range(10):
                gradeRange = f"{i}-{i + 1}"
                count = sum(i <= score < i + 1 for score in scores)
                percentage = count / totalStudents * 100
                gradeDistribution.append(f"{count} ({percentage:.2f}%)")
            
            # Write additional fields to the CSV
            writer.writerow(['ANALYTICS'])
            writer.writerow(['Average Score', averageScore])
            writer.writerow(['Median Score', medianScore])
            writer.writerow(['Mode Score', modeScore])
            writer.writerow(['Lowest Grade', lowestGrade])
            writer.writerow(['Highest Grade', highestGrade])
            writer.writerow(['Grade Ranges'] + [f"{str(i)}->{str(i + 1)}" for i in range(10)])
            writer.writerow(['Grade Distribution'] + gradeDistribution)


