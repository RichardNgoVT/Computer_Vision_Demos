
# CompVisHw3_1.py - Updated Code example code listed in lecture notes for image alignment
# https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/
# Created on 11/12/19
# @Updated: Richard Ngo

from __future__ import print_function
import cv2
import numpy as np
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.015
#############################################################
def showimg(img):
    cv2.namedWindow('showimg', flags=cv2.WINDOW_NORMAL)
    cv2.imshow('showimg', img)
    #cv2.resizeWindow('showimg', (int(len(img[0])*2), int(len(img)*2)))
    #print(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()    
#############################################################
def findMatches(im1, im2):
# Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]
    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    cv2.imwrite("matches.jpg", imMatches)
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    return points1, points2, matches[len(matches)-1].distance
#############################################################
def combineImages(points1, points2, im1,im2):
     # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
    print("here")
    print(h)
    # Use homography
    height1, width1, channels1 = im1.shape
    height2, width2, channels2 = im2.shape  
    if(h[0,2]<=0):
        h1 = np.copy(h)
        h1[0,2] = 0
        h2=np.identity(3)
        h2[0,2]=-h[0,2]
        width = width2
    else:
        h1 = np.copy(h)
        h2=np.identity(3)
        width = width1
    part1 = cv2.warpPerspective(im1, h1, (width+int(round(abs(h[0,2]))), height2))#width+math.ceil(h2[0,2])
    part2 = cv2.warpPerspective(im2, h2, (width+int(round(abs(h[0,2]))), height2))  
    #showimg(part1)
    #showimg(part2)
    #imReg = np.zeros(shape=(height,width))
    """
    if(h1[0,2]==0):
        #imReg[:math.floor(h2[0,2])] = part1[:math.floor(h2[0,2])]  
        imReg=part1
        imReg[:,int(round(h2[0,2])):,:] = part2[:,int(round(h2[0,2])):,:]  
    else:
        #imReg[:math.floor(h1[0,2])] = part2[:math.floor(h1[0,2])]  
        imReg=part2
        imReg[:,int(round(h1[0,2])):,:] = part1[:,int(round(h1[0,2])):,:]
    """
    """
    if(h1[0,2]==0):
        #imReg[:math.floor(h2[0,2])] = part1[:math.floor(h2[0,2])]  
        imReg=part1
        imReg[part2>0] = part2[part2>0]
    else:
        #imReg[:math.floor(h1[0,2])] = part2[:math.floor(h1[0,2])]  
        imReg=part2
        imReg[part1>0] = part1[part1>0]
    """
    imReg=part1
    imReg[part2>0] = part2[part2>0]
    #showimg(imReg)
    return imReg, h
#############################################################
def alignImages(im1in, im2in, im3in):
    #sorter = [im1in,im2in,im3in]
    #sorter.sort(key=lambda x: len(x), reverse=True)
    #[im1, im2, im3] = sorter
    
    points12, points21, MD12 = findMatches(im1in,im2in)
    points13, points31, MD13 = findMatches(im1in,im3in)
    
    points21, points12, MD21 = findMatches(im2in,im1in)
    points23, points32, MD23 = findMatches(im2in,im3in)
    
    points31, points13, MD31 = findMatches(im3in,im1in)
    points32, points23, MD32 = findMatches(im3in,im2in)
    
    compad = [MD12+MD13,MD21+MD23,MD31+MD32]
    center = compad.index(min(compad))
    
    if(center == 0):
        if(MD12<MD13):
            [im1,im2,im3]=[im2in,im1in,im3in]
            [points1,points2] = [points21, points12]
        else:
            [im1,im2,im3]=[im3in,im1in,im2in]
            [points1,points2] = [points31, points13]
    elif(center == 1):
        if(MD21<MD23):
            [im1,im2,im3]=[im1in,im2in,im3in]
            [points1,points2] = [points12, points21]
        else:
            [im1,im2,im3]=[im3in,im2in,im1in]
            [points1,points2] = [points32, points23]
    else:
        if(MD31<MD32):
            [im1,im2,im3]=[im1in,im3in,im2in]
            [points1,points2] = [points13, points31]
        else:
            [im1,im2,im3]=[im2in,im3in,im1in]
            [points1,points2] = [points23, points32]
    #[im1,im2]=[np.copy(im2),np.copy(im1)]
    #[points1,points2]=[np.copy(points2),np.copy(points1)]
    #[im1,im3]=[np.copy(im3),np.copy(im1)]
    #points1, points2, matches = findMatches(im1,im2)
    """
    print(matches2[len(matches2)-1].distance, len(matches2))
    print("here")
    print(matches3[len(matches3)-1].distance, len(matches3))
    if(matches2[len(matches2)-1].distance<=matches3[len(matches3)-1].distance):
    """
    halfReg, h1= combineImages(points1, points2,im1,im2)
    points1, points2, matches3 = findMatches(halfReg,im3)
    fullReg, h2 = combineImages(points1, points2,halfReg,im3)
    """
    else:
        halfReg, h1 = combineImages(points13, points31,im1,im3)
        points1, points2, matches2 = findMatches(halfReg,im2)
        fullReg, h2 = combineImages(points1, points2,halfReg,im2)
        """
    #showimg(fullReg[:,:(width1+width2+width3-abs(math.floor(h1[0,2]))-abs(math.floor(h2[0,2]))),:])
    return fullReg, h1, h2
#############################################################
if __name__ == '__main__':
    # # Read images to be aligned
    ImgNames = ["hobbit","goodwin","BigFour"]
    for I in ImgNames:
        im1R = I+"0.png"
        im2R = I+"1.png"
        im3R = I+"2.png"
        print("Reading ", im1R)
        im1 = cv2.imread(im1R, cv2.IMREAD_COLOR)
        print("Reading ", im2R)
        im2 = cv2.imread(im2R, cv2.IMREAD_COLOR)
        print("Reading ", im3R)
        im3 = cv2.imread(im3R, cv2.IMREAD_COLOR)
        
        print("Aligning images ...")
        # Registered image will be resotred in imReg.
        # The estimated homography will be stored in h.
        imReg, h1, h2 = alignImages(im1, im2, im3)
        # Write aligned image to disk.
        outFilename = I+"_aligned.jpg"
        print("Saving aligned image : ", outFilename);
        cv2.imwrite(outFilename, imReg)
        
        # Print estimated homography
        print("Estimated homographies for: "+I+"\n", h1)
        print(h2)