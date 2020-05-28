# *_*coding:utf-8 *_*
import numpy as np
import cv2 as cv

def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)