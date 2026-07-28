import numpy as np
import cv2 as cv

filename = 'chip.png' #sample photo for corner detection
img = cv.imread(filename)

'''crop the photo to isolate point of interest'''
img_tup = np.shape(img) #find resolution of image
height = img_tup[0] 
width = img_tup[1]

crop_height = height//2
crop_width = width//2
cropped_img = img[crop_height-50:crop_height+50, crop_width-50:crop_width+50] #convert original image into cropped 100x100 pixel frame
''''''

gray = cv.cvtColor(cropped_img,cv.COLOR_BGR2GRAY) #turn cropped image grayscale for more corner detection

gray = np.float32(gray)
dst = cv.cornerHarris(gray,2,3,0.04)
num_corners = print(cv.countNonZero(dst))

'''result is dilated for marking the corners, not important'''
dst = cv.dilate(dst,None)

'''Threshold for an optimal value, it may vary depending on the image'''
cropped_img[dst>0.01*dst.max()]=[0,0,255]

cv.imshow('dst',cropped_img) #display image with detections
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()
