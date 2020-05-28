# *_*coding:utf-8 *_*
import numpy as np
import cv2 as cv

def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)

# 边缘检测+轮廓发现

path = "./images/1_0.jpg"
img = cv.imread(path)
cv.imshow("original", img)

# 灰度图片
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imshow("gray", img_gray)

canny = cv.Canny(img_gray, 100, 100 * 2) # 阈值用createTrackbar获取
imshow("Canny", canny)

# 轮廓发现
contours, heriachy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 　找到的轮廓画在原图
for i, contour in enumerate(contours):
    cv.drawContours(img, contours, i, (0, 255, 0), 2)
imshow("drawcontours", img)

rate = 0.75
imgH, imgW, _ = img.shape
minH, minW = imgH * rate, imgW * rate
# 　根据轮廓找到最大外接矩形
for i, contour in enumerate(contours):
    rect = cv.minAreaRect(contours[i]) # 外接矩形，有方向
    (rectX, rectY), (rectH, rectW), angle = rect
    if rectW > minW and rectH > minH and rectW < (imgW - 20) and rectH < (imgH - 20):
        pts = cv.boxPoints(rect)  # 矩形->点,4个点
        bbox = cv.boundingRect(pts)  # 点->框，bbox,左上角,x,y,w,h


x, y, w, h = bbox
cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
imshow("rectimg", img)

# 截取
x1,y1,x2,y2 = x,y,x+w,y+h
newimg = img[y1:y2,x1:x2]
imshow("newimg", newimg)


cv.waitKey(0)
cv.destroyAllWindows()
