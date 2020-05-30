# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np
from .utils import get_data_set, split_lines
from predict import DigitNumberRecognizer


class TextAreaDetector:
    def __init__(self, modelfile):
        self.net = cv.dnn.readNet(modelfile)
        names = self.net.getLayerNames()
        for name in names:
            print(name)
        self.threshold = 0.5
        self.layerNames = ["feature_fusion/Conv_7/Sigmoid",
                           "feature_fusion/concat_3"]

    def detect(self, image):
        H, W, _ = image.shape
        rH = H / float(320)
        rW = W / float(320)
        blob = cv.dnn.blobFromImage(image, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False)
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layerNames)
        print(scores)

        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # start to decode the output
        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]

            # loop over the number of columns
            for x in range(0, numCols):
                if scoresData[x] < self.threshold:
                    continue

                # compute the offset factor as our resulting feature maps will be 4x smaller than the input image
                (offsetX, offsetY) = (x * 4.0, y * 4.0)

                # extract the rotation angle for the prediction and then compute the sin and cosine
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)

                # use the geometry volum to derive the width and height of the bounding box
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]

                # compute both the starting and ending (x,y)-coordinates for the text prediction bounding box
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)

                # add the bounding box coordinates and probalility score to our respective lists
                rects.append([startX, startY, endX, endY])
                confidences.append(float(scoresData[x]))

        # 非极大值抑制, NMS
        boxes = cv.dnn.NMSBoxes(rects, confidences, self.threshold, 0.8)
        result = np.zeros(image.shape[:2], dtype=np.uint8)
        for i in range(boxes):
            i = i[0]
            box = rects[i]
            startX = box[0]
            startY = box[1]
            endX = box[2]
            endY = box[3]
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            # draw the bounding box on the image
            cv.rectangle(result, (startX, startY), (endX, endY), (0,0,255), 2)
        return result




