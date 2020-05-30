# *_*coding:utf-8 *_*
import cv2 as cv
import numpy as np
import math

# class Hog_descriptor():
#     def __init__(self, img, cell_size=8, bin_size=9):
#         self.img = img
#         self.img = np.sqrt(img * 1.0 / float(np.max(img))) # gamma=0.5
#         self.img = self.img * 255 # 反归一化
#         self.cell_size = cell_size
#         self.bin_size = bin_size
#         self.angle_unit = 180 / self.bin_size # 直方图每列的宽度
#         assert type(self.bin_size) == int, "bin_size should be integer"
#         assert type(self.cell_size) == int, "cell_size should be integer"
#         assert 180%self.bin_size == 0, "bin_size should be divisible by 180"
#
#     # 主功能
#     def extract(self):
#         height, width = self.img.shape
#         # 1，计算图像每一个像素点的梯度幅度和方向
#         gradient_value, gradient_angle = self.global_gradient()
#
#     def global_gradient(self):



