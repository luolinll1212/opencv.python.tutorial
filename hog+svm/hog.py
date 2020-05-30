# *_*coding:utf-8 *_*
import cv2 as cv

def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)

path = r"./images/train/pos/per00001.jpg"
img = cv.imread(path)
imshow("original", img)

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
(rects, weights) = hog.detectMultiScale(img,
                                        winStride=(4, 4),
                                        padding=(8, 8),
                                        scale=1.25,
                                        useMeanshiftGrouping=False)

for (x,y,w,h) in rects:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
imshow("hog-people", img)



cv.waitKey(0)
cv.destroyAllWindows()
