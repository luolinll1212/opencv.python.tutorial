# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np

def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)

# 二值化+形态学变化+霍夫直线检测+透视变换

#读取图片
path = "images/5.jpg"
img = cv.imread(path)
imshow("original", img)

# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imshow("gray", gray)

# 二值化
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)
imshow("binary", binary)

# 图像形态学操作，闭
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
dst = cv.morphologyEx(binary, cv.MORPH_CLOSE,kernel)
imshow("dst", dst)

#
#
#
#
#
#

cv.waitKey(0)
cv.destroyAllWindows()
