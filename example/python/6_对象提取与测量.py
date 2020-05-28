# *_*coding:utf-8 *_*
import cv2 as cv


def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)


# 二值化+形态学变化+轮廓发现

# 读取图片
path = "./images/6.jpg"
img = cv.imread(path)
imshow("original", img)

# # 高斯模糊
# gauss = cv.GaussianBlur(img, (15, 15), 0)
# imshow("gauss", gauss)
#
# # 灰度
# gray = cv.cvtColor(gauss, cv.COLOR_BGR2GRAY)
# imshow("gray", gray)
#
# # 二值化
# ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
# imshow("binary", binary)
#
# # 形态学操作，闭
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
# dst = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
# imshow("dst", dst)
#
# # 轮廓发现
# contours, heriachy = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#
# # 提取面积
# imgH, imgW, _ = img.shape
# resultImg = img.copy()
# for i, contour in enumerate(contours):
#     rect = cv.boundingRect(contour)
#     x, y, w, h = rect
#     if w < imgW / 2: continue
#     if w > imgW - 20: continue
#     area = cv.contourArea(contour)
#     length = cv.arcLength(contour, True)
#     print("area", area)
#     print("length", length)
#     cv.drawContours(resultImg, contours, i, (0,0,255), 1)
# imshow("resultImg", resultImg)

cv.waitKey(0)
cv.destroyAllWindows()
