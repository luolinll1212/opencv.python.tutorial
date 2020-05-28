# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np


def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)


# 二值化+形态学处理(可选)+距离变换+连通区域计算

# 读取图片
path = "images/4.jpg"
img = cv.imread(path)
imshow("original", img)

# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imshow("gray", gray)

# 二值化，单峰－三角阈值
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
imshow("binary", binary)

# 图像形态学，开
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
imshow("dst", dst)

# 取反
binarynot = cv.bitwise_not(dst)
imshow("binarynot", binarynot)

# 计算距离
dist = cv.distanceTransform(binarynot, cv.DIST_L2, cv.DIST_MASK_3)
distnorm = cv.normalize(dist, 0.0, 1.0, norm_type=cv.NORM_MINMAX)
imshow("distnorm", distnorm)

# 阈值分割
ret, distbinary = cv.threshold(distnorm, 0.6, 1.0, cv.THRESH_BINARY)
imshow("distbinary", distbinary)

dist8u = (distbinary * 255).astype(np.uint8) # (0,1) -> (0,255)
imshow("dist8u", dist8u)


# 自适应阈值
binaryadaptive = cv.adaptiveThreshold(dist8u, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 85, 0.0)
imshow("binaryadaptive", binaryadaptive)

# 膨胀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilate = cv.dilate(binaryadaptive, kernel)
imshow("dilate", dilate)

# 计算联通局域
resultImg = dilate.copy()
contours, heriachy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
    np.random.seed()
    b,g,r = np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)
    cv.drawContours(resultImg, contours, i, (b,g,r), 2)
imshow("resultImg", resultImg)

print("总数：", len(contours))

cv.waitKey(0)
cv.destroyAllWindows()
