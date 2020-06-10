# CompVisHw2_1.py - Compares effectivness of template matching with diffrent levels of blur and noise
# Created on 10/20/19
# @author: Richard Ngo
import numpy as np, cv2

################################
#noisy - modified from Shubham Pachori on stackoverflow
def noisy(image, noise_type, sigma):
    if noise_type == "gauss":
        row,col = image.shape
        mean = 0
        gauss = np.random.normal(mean,sigma,(row,col))
        gauss = gauss.reshape(row,col)
        noisy = image + gauss
        return noisy
    elif noise_type == "s&p":
        row,col = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
            for i in image.shape]
        out[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
            for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        row,col = image.shape
        gauss = np.random.randn(row,col)
        gauss = gauss.reshape(row,col)
        noisy = image + image * gauss
    return noisy
################################
    
img = cv2.imread('motherboard-gray.png', cv2.IMREAD_GRAYSCALE)
temp = cv2.imread('template.png', cv2.IMREAD_GRAYSCALE)

heightI = len(img)
widthI = len(img[0])

heightT = len(temp)
widthT = len(temp[0])

maxnoise = 10
maxsig = 5
maxcorr = np.zeros((maxnoise+1, maxsig+1), dtype = "float32")
maxlocy = np.zeros((maxnoise+1, maxsig+1), dtype = "int64")
maxlocx = np.zeros((maxnoise+1, maxsig+1), dtype = "int64")
for N in range(maxnoise+1):
    for S in range(maxsig+1):
        no = np.uint8(noisy(img, 'gauss', N))
        if S == 0:
            nosm = no
        else:
            nosm = cv2.GaussianBlur(no,(S*6+1,S*6+1), S)
        matched = cv2.matchTemplate(nosm,temp,cv2.TM_CCORR_NORMED)
        max_val = np.amax(matched)
        loc = np.where(matched==max_val)       
        
        #for viewing
        maxcorr[N,S] = max_val
        maxlocy[N,S] = abs(loc[0][0]-382)
        maxlocx[N,S] = abs(loc[1][0]-438)
        
N = maxnoise
S = maxsig

heightM = len(matched)
widthM = len(matched[0])
dst = np.zeros(shape=(len(matched),len(matched[0])))
matched = np.uint8(cv2.normalize(matched,dst,0,255,cv2.NORM_MINMAX))
histEqmatched = cv2.equalizeHist(matched)
transmatch = np.zeros((heightM, widthM), dtype = "uint8")

#Piecewise-Linear Transformation
r1 = 160#input
s1 = 40#output
r2 = 245#input
s2 = 128#output

for i in range(heightM):
    for j in range(widthM):
        if matched[i,j] <= r1:
            transmatch[i,j] = s1/r1*matched[i,j]#if intensity of image is below or equal to r1
        elif matched[i,j] >= r2:
            transmatch[i,j] = (255-s2)/(255-r2)*(matched[i,j]-r2)+s2#if intensity of image is above or equal to r2
        else:
            transmatch[i,j] = (s2-s1)/(r2-r1)*(matched[i,j]-r1)+s1#if intensity of image is between r1 and r2

heightI = len(nosm)
widthI = len(nosm[0])

#show resulting images for N = 10, S = 5
cv2.namedWindow('Noise: N='+str(N)+' S='+str(S), flags=cv2.WINDOW_NORMAL)
cv2.imshow('Noise: N='+str(N)+' S='+str(S), no)
cv2.resizeWindow('Noise: N='+str(N)+' S='+str(S), (int(widthI/2), int(heightI/2)))

cv2.namedWindow('Smoothed: N='+str(N)+' S='+str(S), flags=cv2.WINDOW_NORMAL)
cv2.imshow('Smoothed: N='+str(N)+' S='+str(S), nosm)
cv2.resizeWindow('Smoothed: N='+str(N)+' S='+str(S), (int(widthI/2), int(heightI/2)))

cv2.namedWindow('Corr: N='+str(N)+' S='+str(S), flags=cv2.WINDOW_NORMAL)
cv2.imshow('Corr: N='+str(N)+' S='+str(S), matched)
cv2.resizeWindow('Corr: N='+str(N)+' S='+str(S), (int(widthM/2), int(heightM/2)))

cv2.namedWindow('HECorr: N='+str(N)+' S='+str(S), flags=cv2.WINDOW_NORMAL)
cv2.imshow('HECorr: N='+str(N)+' S='+str(S), histEqmatched)
cv2.resizeWindow('HECorr: N='+str(N)+' S='+str(S), (int(widthM/2), int(heightM/2)))

cv2.namedWindow('LTCorr: N='+str(N)+' S='+str(S), flags=cv2.WINDOW_NORMAL)
cv2.imshow('LTCorr: N='+str(N)+' S='+str(S), transmatch)
cv2.resizeWindow('LTCorr: N='+str(N)+' S='+str(S), (int(widthM/2), int(heightM/2)))


cv2.waitKey(0)
cv2.destroyAllWindows()
