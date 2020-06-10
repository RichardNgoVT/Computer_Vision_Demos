# CompVisHw2_2.py - Setting up puppies for watershed seperation
# Created on 10/21/19
# @author: Richard Ngo

import cv2
import numpy as np

img = cv2.imread('puppies.png',0)

#set bright boundaries to backround
ret,backround = cv2.threshold(img,105,255,cv2.THRESH_TOZERO_INV)#70, 105
#set dark boundaries between dogs to backround
ret,backround = cv2.threshold(backround,9,255,cv2.THRESH_BINARY)#9, 10

#cv2.imshow('backround', backround)

distIm = cv2.distanceTransform(backround,cv2.DIST_L2,5)

dst = np.zeros(shape=(len(distIm),len(distIm[0])))

#cv2.imshow('distanceT', np.uint8(cv2.normalize(distIm,dst,0,255,cv2.NORM_MINMAX)))

cv2.imshow('distance transform', cv2.equalizeHist(np.uint8(cv2.normalize(distIm,dst,0,255,cv2.NORM_MINMAX))))

ret, foreground = cv2.threshold(distIm,0.55*distIm.max(),255,0)
#cv2.imshow('foreground', foreground)

foreground = np.uint8(foreground)
backround = np.uint8(backround)


#code based on opencv watershed tutorial from here on out
#https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

#to be flooded
unknown = cv2.subtract(backround,foreground)

#cv2.imshow('unknown', unknown)

#markers
ret, markers = cv2.connectedComponents(foreground)

# make backround the shallowest segment
markers = markers+1

# set up region to be flooded
markers[unknown==255] = 0

#cv2.imshow('markers', np.uint8(np.uint8(cv2.normalize(markers,dst,0,255,cv2.NORM_MINMAX))))

img = cv2.imread('puppies.png')
cv2.imshow('Original Image', img)
#blur image for more effective segmentation
imgGB = cv2.GaussianBlur(img,(5,5),5)#9,9,1  5,5,5  11,11,3  11,11,1

markers = cv2.watershed(imgGB,markers)

"""
#layered watershed?
test = markers
unknownT = unknown

unknownT[test > 1] = 0
test[unknownT==255] = 0
test[test == -1] = 0

#test[unknown==255] = 0
#test[unknown<255] = 1
#unknownT = unknown
#test[markers == -1] = 0
#test[markers <= 1] = 0
#unknownT[test == 255] = 0
#test[unknownT==255] = 0

#ret, test = cv2.connectedComponents(test)

#test = cv2.watershed(imgGB,test)
#test = np.uint8(cv2.normalize(test,dst,0,255,cv2.NORM_MINMAX))

test = cv2.watershed(imgGB,test)


test = np.uint8(cv2.normalize(test,dst,0,255,cv2.NORM_MINMAX))
cv2.imshow('test', test)
"""

img[markers == -1] = [255,0,0]


cv2.imshow('Final Segmentations', np.uint8(cv2.normalize(markers,dst,0,255,cv2.NORM_MINMAX)))
cv2.imshow('Final Segmented Image', img)

print('number of segments found excluding backround',np.amax(markers)-1)

cv2.waitKey(0)
cv2.destroyAllWindows()
