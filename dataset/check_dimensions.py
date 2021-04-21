import cv2
import numpy as np
import sys

np.set_printoptions(threshold=sys.maxsize)
im = cv2.imread('masks/image_1.png',cv2.IMREAD_GRAYSCALE)

im2 = cv2.imread('../matlab/masks/img_1.png',cv2.IMREAD_GRAYSCALE)

print(np.unique(im))
print(np.unique(im2))