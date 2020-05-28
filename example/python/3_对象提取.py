# *_*coding:utf-8 *_*
import numpy as np
import cv2 as cv


def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)


# 二值化+形态学变化+轮廓发现

# 读取图片
path = "./images/3.jpg"
img = cv.imread(path)
imshow("orignal", img)

# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imshow("gray", gray)

# 二值化
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
imshow("binary", binary)

# 形态学，开
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
imshow("morphologyEx", dst)

# 轮廓发现
contours, heriachy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# 　找到的轮廓画在原图
resultImg = img.copy()
for i, contour in enumerate(contours):
    area = cv.contourArea(contour)
    if area < 100:  # 如果面积小于100，跳出
        continue
    rect = cv.boundingRect(contour)  # 外接矩形，没有方向
    x, y, w, h = rect
    ratio = float(w) / float(h)
    if 0.9 < ratio and ratio < 1.1:
        cv.rectangle(resultImg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.drawContours(resultImg, contours, i, (0, 0, 255), 2)
        # 计算外接圆的周长和面积
        print("circle area: %f" % (area))
        print("circle length: %f " % (cv.arcLength(contour, True)))
        cx = int(x + w / 2)
        cy = int(y + h / 2)
        cv.circle(resultImg, (cx, cy), 2, (255, 0, 0), 2, 8, 0)

imshow("boundingRect", resultImg)

cv.waitKey(0)
cv.destroyAllWindows()
