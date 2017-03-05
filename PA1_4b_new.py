__author__ = 'khushboo_agarwal'

import math
from scipy import ndimage
from PIL import Image
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from pylab import *
import time

#definition of gaussian function
def gaussian_mask (x,sigma): 
	if sigma == 0:
		return 0
	else:
		gauss = (1/(math.sqrt(2*(math.pi))*sigma))*exp(-((x**2))/2/sigma**2)
	return gauss

#definition of gaussian second derivative function    
def gaussian_derv(x,y,sigma,z):
    if(z =='x'):
        gaussian_der = gaussian_mask(x,sigma)*(-x/(sigma**2))
    elif(z=='y'):
        gaussian_der = gaussian_mask(x,sigma)*(-y/(sigma**2))
    return gaussian_der

# definition of Harris matrix
def Harris_matrix(Img, alpha, threshold, sigma):
    start = time.clock()
    
    I = array(Image.open(Img).convert('L')) # read the input image
    
    G = []
    for i in range(-2,2+1):
        G.append(gaussian_mask(i,sigma)) # equating y to 0 since we need a 1D matrix
    
    Gx = []
    for i in range(-2,2+1):
        Gx.append(gaussian_derv(i,0,sigma,'x')) 
   	
    Gy = []
    for i in range(-2,2+1):
        Gy.append([gaussian_derv(0,i,sigma,'y')]) 
    
    I1 = []
    for i in range(len(I[:,0])):
   	I1.extend([convolve(I[i,:],G)]) # I*G in x direction
    I1 = array(matrix(I1))
    I11 = I1*I1
    
    Ix = []
    for i in range(len(I[:,0])):
   	Ix.extend([convolve(I1[i,:],Gx)]) # I*G in x direction
    Ix = array(matrix(Ix))
    
    I2 = []
    for i in range(len(I[0,:])):
   	I2.extend([convolve(I[:,i],G)]) # I*G in y direction
    I2 = array(matrix(transpose(I2))) 
    I22 = I2*I2
    
    Iy = []
    for i in range(len(I[0,:])):
   	Iy.extend([convolve(I2[:,i],Gx)]) # I*G in y direction
    Iy = array(matrix(transpose(Iy))) 
    
    I12 = []
    for i in range(len(I1[:,0])):    
        threshold = []
        for j in range(len(I2[0,:])):    
   	    threshold.append(I1[i,j]*I2[i,j])
   	if (j == len(I2[0,:])-1):
                I12.extend(array(matrix(threshold)))
    I12 = array(matrix(I12))
    
    
    Ixy = []
    for i in range(len(I12[:,0])):
   	Ixy.extend([convolve(I12[i,:],Gx)]) # I*G in x direction
    Ixy = array(matrix(Ixy))
    
    
    
    x = [] # this array stores the x vertex of corners
    y = [] # this array stores the y vertex of corners
    for i in range(len(I[:,0])):
        for j in range(len(I[0,:])):
            H = ([Ix[i,j]**2,Ix[i,j]*Iy[i,j]],[Ix[i,j]*Iy[i,j],Iy[i,j]**2]) # Harris Matrix
            if(abs(linalg.det(H)-(alpha*(trace(H)))) > threshold): # if a corner # if a corner                
                y.append(i-5)
                x.append(j-5)
    
    plt.figure()
    plt.imshow(I,cmap = cm.gray)
    plot(x,y,'r.')
    plt.axis([5,len(I[0,:]),len(I[:,0]),5])
    show()
    return time.clock() - start

time1 = Harris_matrix('input1.png',.03,16.5,1.5)
time2 = Harris_matrix('input2.png',.004,4.55,1.5)
time3 = Harris_matrix('input3.png',.004,1,1.5)
print 'The Accuracy is:\nInput Image 1: %.2fseconds\nInput Image 2: %.2fseconds\nInput Image 3: %.2fseconds)'%(time1,time2,time3)
'''=====================================================Conclusion start==================================================
Accuray of this algoritm is better than the next one.
========================================================Conclusion end=================================================='''