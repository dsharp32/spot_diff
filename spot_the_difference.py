#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:50:25 2020

@author: user
"""

from skimage.measure import compare_ssim
import cv2
import numpy as np
from PIL import Image

# Open the image form working directory
image = Image.open('cars.jpg')
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)
# show the image
image.show()

data = np.array(image)
# print(type(data))
# summarize shape
# print(data.shape)

# get number of columns in 2D numpy array
numOfColumns = data.shape[1]
left_width = int(numOfColumns/2)
right_width = int(numOfColumns/2)

print(left_width,right_width)

left = data[:,:left_width] 
right = data[:,-right_width:]

imLeft = Image.fromarray(left)
imRight = Image.fromarray(right)

imLeft = imLeft.save("left.jpg")
imRight = imRight.save("right.jpg")

before = cv2.imread('left.jpg')
after = cv2.imread('right.jpg')

# Convert images to grayscale
before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

# Compute SSIM between two images
(score, diff) = compare_ssim(before_gray, after_gray, full=True)
print("Image similarity", score)

# The diff image contains the actual image differences between the two images
# and is represented as a floating point data type in the range [0,1] 
# so we must convert the array to 8-bit unsigned integers in the range
# [0,255] before we can use it with OpenCV
diff = (diff * 255).astype("uint8")

# Threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

mask = np.zeros(before.shape, dtype='uint8')
filled_after = after.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(before, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(after, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, (0,255,0), -1)
        cv2.drawContours(filled_after, [c], 0, (0,255,0), -1)

cv2.imshow('before', before)
cv2.imshow('after', after)
cv2.imshow('diff',diff)
cv2.imshow('mask',mask)
cv2.imshow('filled after',filled_after)
cv2.waitKey(0)