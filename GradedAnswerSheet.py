class GradedAnswerSheet:
    def __init__(self, candidateNumber, testCode, score, resultImage, answerList):
        self.candidateNumber = candidateNumber
        self.answerList = answerList
        self.testCode = testCode
        self.score = score
        self.resultImage = resultImage