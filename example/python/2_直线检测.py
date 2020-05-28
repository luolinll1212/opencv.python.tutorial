# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np

# 二值化+形态学变化＋霍夫直线检测

def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)

# 读取图像
path = "./images/2_0.jpg"
img = cv.imread(path)
imshow("original", img)

# 灰度图
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imshow("gray", gray)

# 高斯模糊
gauss = cv.GaussianBlur(gray, (3,3), 0)
imshow("gauss", gauss)

# 二值化
ret, binary = cv.threshold(gauss, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_TRIANGLE)
imshow("binary",binary)

# 形态学操作,开，膨胀->腐蚀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 1), (-1,-1))
dst = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)
imshow("morphologyEx", dst)

# 图像形态学操作，腐蚀
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3),(-1,-1))
dilate = cv.dilate(dst, kernel)
imshow("dilate", dilate)

resultImg = img.copy()
minLINELENGTH=100
# 霍夫直线检测
lines = cv.HoughLinesP(dilate, 1, np.pi/180, 50, 10)
print(lines.shape)
for i, line in enumerate(lines):
    cv.line(resultImg, (line[:,0], line[:,1]), (line[:,2], line[:,3]), (0,0,255), 2, 8, 0)
imshow("result", resultImg)



cv.waitKey(0)
cv.destroyAllWindows()


