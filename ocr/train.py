# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np
import os

def load_data():
    images = []
    labels = []
    path = "./digits"
    files = os.listdir(path)
    count = len(files)
    sample_data = np.zeros((count, 28*48), dtype=np.float32)
    index = 0
    for filename in files:
        file_path = os.path.join(path, filename)
        if os.path.isfile(file_path) is True:
            images.append(file_path)
            labels.append(filename[:1])
            img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, (28, 48))
            row = np.reshape(img, (-1, 28*48))
            sample_data[index] = row
            index +=1
    return sample_data, np.asarray(labels, np.int32)

train_data, train_labels = load_data()

# train stage
svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.83)
svm.train(train_data, cv.ml.ROW_SAMPLE, train_labels)
svm.save("svm.yml")

# test
n_test = 3
svm = cv.ml.SVM_load("svm.yml")
result = svm.predict(train_data[:n_test])[1]
print(result)
print(train_labels[:n_test])