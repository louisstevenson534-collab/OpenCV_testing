import numpy as np
import cv2 as cv

filename = 'chip.png'
img = cv.imread(filename)

img_tup = np.shape(img)
height = img_tup[0]
width = img_tup[1]

crop_height = height//2
crop_width = width//2
cropped_img = img[crop_height-50:crop_height+50, crop_width-50:crop_width+50]

gray = cv.cvtColor(cropped_img,cv.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
num_corners = print(cv.countNonZero(dst))

#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)

## Threshold for an optimal value, it may vary depending on the image.
cropped_img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',cropped_img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()