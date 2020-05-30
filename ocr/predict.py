# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np

class DigitNumberRecognizer:
    def __init__(self):
        print("create object...")
        self.svm = cv.ml.SVM_load("svm.yml")
    def predict(self, dataset):
        result = self.svm.predict(dataset)[1]
        text = ""
        for i in range(len(result)):
            text += str(np.int32(result[i][0]))
        print(text)
        return text