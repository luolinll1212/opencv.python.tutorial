# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np


def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)


# 二值化+形态学变化+霍夫直线检测+透视变换

# 读取图片
path = "/home/rose/git/opencv.python.tutorial/example/python/images/5.jpg"
img = cv.imread(path)
imshow("original", img)

# 灰度
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imshow("gray", gray)

# 二值化
ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
imshow("binary", binary)

# 图像形态学操作，闭
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
imshow("dst", dst)

# 轮廓发现
contours, heriachy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

imgH, imgW, _ = img.shape
draw = np.zeros(img.shape, dtype=np.uint8)
# 找外接矩形
for i, contour in enumerate(contours):
    rect = cv.boundingRect(contour)
    x, y, w, h = rect
    if w > imgW / 2 and h > imgH / 2 and w < (imgW - 10):
        cv.drawContours(draw, contours, i, (0, 0, 255), 2)
imshow("draw", draw)

# 霍夫直线检测
drawBin = cv.cvtColor(draw, cv.COLOR_BGR2GRAY)  # 灰度
acc = int(min(imgH * 0.5, imgW * 0.5))
linesImg = np.zeros(img.shape, dtype=np.uint8)
lines = cv.HoughLinesP(drawBin, 1, np.pi / 180, acc, acc)  # 霍夫直线检测
lines = lines.squeeze(1)
for i, line in enumerate(lines):
    cv.line(linesImg, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2, 8, 0)
imshow("linesImg", linesImg)

# 寻找直线,定位坐标
drawlinesImg = np.zeros(img.shape, dtype=np.uint8)

topline, bottonline, leftline, rightline = None, None, None, None
for i, line in enumerate(lines):  # 过滤线
    deltaX = np.abs(line[2] - line[0])
    deltaY = np.abs(line[3] - line[1])
    if line[3] < imgH / 2.0 and line[1] < imgH / 2.0 and deltaX > acc * 0.5:
        topline = line
        cv.line(drawlinesImg, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2, 8, 0)
    elif line[3] > imgH / 2.0 and line[1] > imgH / 2.0 and deltaX > acc * 0.5:
        bottonline = line
        cv.line(drawlinesImg, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2, 8, 0)
    elif line[0] < imgW / 2.0 and line[2] < imgW / 2.0 and deltaY > acc * 0.5:
        leftline = line
        cv.line(drawlinesImg, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2, 8, 0)
    elif line[0] > imgW / 2.0 and line[2] > imgW / 2.0 and deltaY > acc * 0.5:
        rightline = line
        cv.line(drawlinesImg, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2, 8, 0)
imshow("drawlinesImg", drawlinesImg)

# 拟合四条直线, y=kx+c
newlines = [topline, bottonline, leftline, rightline]
k = []
c = []
for line in newlines:
    k1 = float(line[3] - line[1]) / float(line[2] - line[0])
    c1 = line[1] - k1 * line[0]
    k += [(k1)]
    c += [(c1)]

# 寻找四个顶点
# 左上
x1 = int((c[0] - c[2]) / (k[2] - k[0]))
y1 = int((k[0] * x1 + c[0]))
# 右上
x2 = int((c[0] - c[3]) / (k[3] - k[0]))
y2 = int(k[0] * x2 + c[0])
# 左下
x3 = int((c[1] - c[2]) / (k[2] - k[1]))
y3 = int(k[1] * x3 + c[1])
# 右下
x4 = int((c[1] - c[3]) / (k[3] - k[1]))
y4 = int(k[1] * x4 + c[1])

# 画点
cv.circle(drawlinesImg, (x1, y1), 2, (255, 255, 0))
cv.circle(drawlinesImg, (x2, y2), 2, (255, 255, 0))
cv.circle(drawlinesImg, (x3, y3), 2, (255, 255, 0))
cv.circle(drawlinesImg, (x4, y4), 2, (255, 255, 0))
imshow("drawlinesImg", drawlinesImg)

# 透视变换
pts1 = np.float32([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) # 左上，右上，左下，右下
pts2 = np.float32([[0,0],[imgW, 0],[0, imgH],[imgW, imgH]])
M = cv.getPerspectiveTransform(pts1, pts2)
imgWrap = cv.warpPerspective(img, M, (imgW, imgH))
imshow("imgWrap", imgWrap)

cv.waitKey(0)
cv.destroyAllWindows()
