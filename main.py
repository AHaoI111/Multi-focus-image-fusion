import glob
import Image_blending
import cv2

img1 = cv2.imread('1.jpg')
img2 = cv2.imread('2.jpg')

img = Image_blending.merge_images([img1, img2])
cv2.imshow('img', img)
cv2.waitKey(0)