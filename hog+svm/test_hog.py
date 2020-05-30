# *_*coding:utf-8 *_*
import cv2 as cv

def imshow(name, img):
    cv.namedWindow(name)
    cv.imshow(name, img)

path = r"./images/train/pos/per00001.jpg"
img = cv.imread(path)
imshow("original", img)

winSize = (128,64) # w,h
blocksize = (8,8)
blockstride = (8,8)
cellsize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64
hog= cv.HOGDescriptor(winSize, blocksize, blockstride, cellsize, nbins, derivAperture, winSigma,
                      histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
winstride = (8,8)
padding = (8,8)
locations = ((10, 20),)
hist = hog.compute(img, winstride, padding, locations)
print(hist.shape) # 1152 = 8*16*9 -> 64*128 /(8*8) * 9 , 每个cell用9个表示方向

cv.waitKey(0)
cv.destroyAllWindows()




