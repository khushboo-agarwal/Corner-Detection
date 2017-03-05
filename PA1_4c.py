__author__ = 'khushboo_agarwal'

#the header files
from numpy import *
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os,sys
import scipy
import time

#definition of gaussian function
def Gaussian_function(x, sigma):
	if sigma == 0:
		return 0
	else:
		g = math.exp(-x*x)/(2*sigma*sigma) 
		return g 									#this functions returns the value of the gaussian function as given by the formula based on x and sigma value
#gaussian mask
def Gaussian_mask(sigma):
	Gaussian = np.ones((1, 5), np.float32)
	G_1d = np.ones((1, 5), np.float32)
	G_2d = np.ones((1, 5), np.float32)
	for i in range(-2, 2+1):
		gaussian_1 = Gaussian_function(i, sigma)						#substituting y = 0, because we want the result in one dimension
		gaussian_2 = Gaussian_function(i-0.5, sigma)
		gaussian_3 = Gaussian_function(i+0.5, sigma)
		gauss_mean = (gaussian_1+gaussian_2+gaussian_3)/3
		Gaussian[0][i+2] = 	gauss_mean*(1/(math.sqrt(2.0 * math.pi * sigma * sigma)))									#gaussian[] will save the gaussian function value
		#calculating the derivative gaussian with formula (-x/pow(sigma,3)*sqrt(2*pi))*Gaussian_function
		gaussian_der = gauss_mean*((-i)/(sigma*sigma*sigma*(math.sqrt(2*math.pi))))
		G_1d[0][i+2] = gaussian_der
	return (Gaussian, G_1d)
#smoothing the image
def gaussian_conv(I, G, S):

	#Creating an empty array Iyy needed for transposing the result
	Iyy = [] 	
	S = np.shape(I)	
	#Creating an empty array Ix: I[i,:]*G
	Igx = []	

	#Creating an empty array Iy: I[:,i]*G
	Igy = []			

	#convolving in the x direction
	for i in range(S[0]):
		I1 = (np.convolve(I[i,:], G, 'same'))
		Igx.append(np.array(I1))

	#convolving in the y direction
	for i in range(S[1]):
		I2 = (np.convolve(I[:,i], G, 'same'))
		Iyy.append(np.array(I2))
		Igy = np.transpose(Iyy)
	
	return(Igx, Igy)

#1st derivative gaussian convolution
def gaussian_1derv_conv(Ix,Iy,Ixy,Gx,Gy, S):

	#Creating an empty array Iyy needed for transposing the result
	Iyy = []
	S = np.shape(Ix)
	I_x = []
	I_y = []
	I_xy = []
	#computing for I_x
	for i in range(0, S[0]):
		I1 = (np.convolve(np.array(Ix)[i,:], Gx, 'same'))
		I_x.append(np.array(I1))

	#Computing for I_y
	for i in range(0, S[1]):
		I2 = (np.convolve(np.array(Iy)[:,i], Gy, 'same'))
		Iyy.append(np.array(I2))
		I_y = np.transpose(Iyy)

	for i in range(0, S[0]):
		I3 = (np.convolve(np.array(Ixy)[i,:], Gx, 'same'))
		I_xy.append(np.array(I3))


	return(I_x, I_y, I_xy)
#defining the harris corner
def Harris(img, th):
	start = 0
	start = time.start()
	I = np.array(img)
	S = np.shape(I)
	Gaussian = np.ones((1, 5), np.float32)
	G_1d = np.ones((1, 5), np.float32)
	Gaussian, G_1d = Gaussian_mask(sigma)
	G = Gaussian.flatten()
	G1dx = G_1d.flatten()
	G1dy = np.transpose(G1dx)

	Igx = Igy = []
	Igx, Igy = gaussian_conv(I, G, S)
	Ix2 = np.square(Igx)
	Iy2 = np.square(Igy)
	Ixy = np.ones(S)
	for i in range(S[0]):
		for j in range(S[1]):
			Ixy[i][j] = Igx[i][j] * Igy[i][j]

	Lxx = Lyy = []
	Lxx, Lyy, Lxy = gaussian_1derv_conv(Ix2, Iy2, Ixy, G1dx, G1dy, S)

	#plt.imshow(I, cmap = cm.gray)
	for i in range(S[0]):
		for j in range(S[1]):
			Harris = ([Lxx[i][j], Lxy[i][j]],[Lxy[i][j], Lyy[i][j]])
			eigenv1, eigenv2 = np.linalg.eig(Harris)[0]
			C = eigenv1*eigenv2 - 0.04*(eigenv1+eigenv2)
			if(C>th):
				x.append[j]
				y.append[i]

	plt.figure()
	plt.imshow(I, cmap = cm.gray)
	plt.plot(x,y,'ro')
	plt.axis([0, S[1]], [S[0], 0])
	plt.show()
	return (time.start() - start)

sigma = 1.5
	#reading the image
Image_1 = Image.open("input1.png").convert('L')
time1 = Harris(Image_1, 20)
Image_2 = Image.open("input2.png").convert('L')
time2 = Harris(Image_2, 36)
Image_3 = Image.open("input3.png").convert('L')
time3 = Harris(Image_3, 15)
print("the time taken for image1, image2, image3 is respectively", time1, time2, time3)

