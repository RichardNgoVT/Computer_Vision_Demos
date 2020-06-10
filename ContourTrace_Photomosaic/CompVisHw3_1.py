# -*- coding: utf-8 -*-
"""
ECE5554 FA19 HW3 part 1.py - contours
Created on Fri Oct 25 10:34:24 2019
@author: crjones4
"""

# CompVisHw3_1.py - implemented Pavlidis contour Extraction
# Created on 11/12/19
# @Updated: Richard Ngo
import numpy as np
import cv2
import math
#pathName = "C:\\Data\\" # change this for your own file structure
pathName = ""
MAXCONTOUR = 5000
doLogging = False
def showImage(img, name):
 cv2.imshow(name, img)
 return
################################

def saveImage(img, name):
    cv2.imwrite(name + ".png", img)
    return
################################
def GaussArea(pts):
    area = 0
    for i in range(len(pts)):
        if(i==len(pts)-1):
            P1 = pts[i]
            P2 = pts[0]
        else:
            P1 = pts[i]
            P2 = pts[i+1]
        area +=(P1[0]*P2[1]-P1[1]*P2[0])/2
    return abs(area);
################################
def onePassDCE(ctrIn):
    Kmin = 0
    imin = 0
    for i in range(len(ctrIn)):
        if(i==0):
            P1 = ctrIn[len(ctrIn)-1]
            P2 = ctrIn[i]
            P3 = ctrIn[i+1]
            #print(P1,P2,P3)
        elif(i==len(ctrIn)-1):
            P1 = ctrIn[i-1]
            P2 = ctrIn[i]
            P3 = ctrIn[0]
        else:
            P1 = ctrIn[i-1]
            P2 = ctrIn[i]
            P3 = ctrIn[i+1]
            #print(P1,P2,P3)
        #print(range(len(ctrIn)))
       # print(P1[1],P2[1],P3[1])
       # print(P1[0],P2[0],P3[0])
        
        L1 = ((P2[1]-P1[1])**(2)+(P2[0]-P1[0])**(2))**(1/2)
        L2 = ((P3[1]-P2[1])**(2)+(P3[0]-P2[0])**(2))**(1/2)
        
        
       
        if((P2[0]-P1[0]) == 0):
            ang1 = (P2[1]-P1[1])/abs(P2[1]-P1[1])*math.pi/2
        else:
            ang1 = (P2[1]-P1[1])/(P2[0]-P1[0])
        if((P3[0]-P2[0]) == 0):
            ang2 = (P3[1]-P2[1])/abs(P3[1]-P2[1])*math.pi/2
        else:
            ang2 = (P3[1]-P2[1])/(P3[0]-P2[0])
        angD = math.atan(ang1)-math.atan(ang2)
        K = abs(angD*L1*L2/(L1+L2))
        if(K<Kmin or i==0):
            #print(K,i)
            Kmin = K
            imin = i
    trimmedContour = np.append(ctrIn[0:imin],ctrIn[imin+1:len(ctrIn)], axis=0)
    """
    contourImage = cv2.imread('VAoutline.png')
    for i in range(len(trimmedContour)):
       contourImage[trimmedContour[i,1],trimmedContour[i,0]] = [255,0,0] 
    cv2.namedWindow('contour', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('contour', contourImage)
    cv2.resizeWindow('contour', (int(len(contourImage[0])*2), int(len(contourImage)*2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return trimmedContour
################################
def Pavlidis(img, start):
    contourImage = cv2.imread('VAoutline.png') 
    stuck = 0
    points = np.array([start])
    trace = np.array([start[0],start[1]])
    vert=1
    hor=0
    while 1:
        i = 0
        while i < 3:
            horP = hor
            vertP = vert
            #print(img[trace[1]+(0-1)*hor-1*vert,trace[0]+(0-1)*vert+1*hor], img[trace[1]+(1-1)*hor-1*vert,trace[0]+(1-1)*vert+1*hor], img[trace[1]+(2-1)*hor-1*vert,trace[0]+(2-1)*vert+1*hor], trace[1],trace[0], vert, hor, i)
            #if(img[trace[1]+(i-1)*hor-1*vert,trace[0]+(i-1)*vert+1*hor] != img[trace[1]-1*hor,trace[0]-1*vert]):
            if(img[trace[1]+(i-1)*hor-1*vert,trace[0]+(i-1)*vert+1*hor] > 0):
                #print(trace[1],trace[0], vert, hor)
                stuck=0
                trace[1]+=(i-1)*hor-1*vert
                trace[0]+=(i-1)*vert+1*hor
                
                contourImage[trace[1],trace[0]] = [255,0,0]
                
                if(trace[0] == start[0] and trace[1] == start[1]):
                    
                    cv2.namedWindow('contour', flags=cv2.WINDOW_NORMAL)
                    cv2.imshow('contour', contourImage)
                    #print(points)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    
                    return points
                points = np.append(points, [[trace[0],trace[1]]], axis=0)
                #hor = (i-1)*vertP+horP*(2-i)*(i)
                #vert = -(i-1)*horP+vertP*(2-i)*(i)
                if(i!=2):
                    hor = (i-1)*vertP+horP*i
                    vert = -(i-1)*horP+vertP*i
                i=-1
            i+=1
        stuck = stuck+1
        if(stuck == 3):
            points = np.append(points, [[trace[0],trace[1]]], axis=0)
            return points
        hor = vertP
        vert = -horP
################################
def showContour(ctr, img, name):
    contourImage = img
    length = ctr.shape[0]
    for count in range(length):
        contourImage[ctr[count, 1], ctr[count, 0]] = 0
        cv2.line(contourImage,(ctr[count, 0], ctr[count, 1]), \
                 (ctr[(count+1)%length, 0], ctr[(count+1)%length, 1]),(128,128,128),1)
    showImage(contourImage, name)
    saveImage(contourImage, name)
#################################
inputImage = cv2.imread(pathName + 'VAoutline.png', cv2.IMREAD_GRAYSCALE)
thresh = 70;
binary = cv2.threshold(inputImage, thresh, 255, cv2.THRESH_BINARY)[1]
(height, width) = binary.shape
# find a start point
ystt = np.uint8(height/2) # look midway up the image
for xstt in range(width): # from the left
    if (binary[ystt, xstt] > 0):
        break
contour = Pavlidis(binary, [xstt, ystt])
showContour(contour, inputImage, "CONTOUR")
print("Initial Significant Corners:",contour.shape[0], " Initial Gauss Area:",GaussArea(contour))
for step in range(6):
    numLoops = math.floor(contour.shape[0]/2)
    for idx in range(numLoops):
        contour = onePassDCE(contour)
    showContour(contour, np.zeros_like(inputImage), "STEP"+str(step))
    print("Step:",step," Significant Corners:",contour.shape[0], " Gauss Area:",GaussArea(contour))
cv2.waitKey(0)
cv2.destroyAllWindows()