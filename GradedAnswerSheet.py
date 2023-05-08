class GradedAnswerSheet:
    def __init__(self, candidateNumber, testCode, score, resultImage, answerList, wrongAnswerList, correctAnswerList):
        self.candidateNumber = candidateNumber
        self.answerList = answerList
        self.wrongAnswerList = wrongAnswerList
        self.correctAnswerList = correctAnswerList
        self.testCode = testCode
        self.score = score
        self.resultImage = resultImage