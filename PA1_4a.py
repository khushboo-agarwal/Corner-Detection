__author__ = 'khushboo agarwal'


#the header files
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os,sys
import scipy
import time

'''
Func: gaussian_function
param : x , sigma
Desc : implements e^((-x^2)/(2*sigma*sigma))
'''

def gaussian_function(x, sigma):
	if sigma == 0:
		return 0
	else:
		g = math.exp(-x*x)/(2*sigma*sigma)
	return g
'''
Func: gaussian_mask
param: sigma
Desc: returns the gaussian mask, gaussian 1st derivative mask and the gaussian 2nd derivative mask
'''

def gaussian_mask(sigma):
	g = np.ones((1,5), np.float32)
	g_1d = np.ones((1,5), np.float32)
	g_2d = np.ones((1,5), np.float32)
	
	for i in range(-2, 3):
		gaussian_1 = gaussian_function(i, sigma)
		gaussian_2 = gaussian_function(i-0.5, sigma)
		gaussian_3 = gaussian_function(i+0.5, sigma)
		gaussian = (gaussian_1+ gaussian_2+gaussian_3)/3
		g[0][i+2] = gaussian * (1/(math.sqrt(2.0 * math.pi * sigma * sigma)))
		g_1d[0][i+2] = gaussian * (-i) * (1/(math.sqrt(2.0 * math.pi * sigma**4)))
		g_2d[0][i+2] = gaussian * (-1 + (i*i)/(sigma*sigma))* (1/(math.sqrt(2.0 * math.pi * sigma**4)))
	return (g, g_1d, g_2d)
'''
Func: gaussian_conv
param:	I : image
		G : Gaussian mask
		S : shape of image
returns: convoluted image in both x and y direction individually which smooths the image: Igx and Igy 
'''
def gaussian_conv(I, G, S):

	#Creating an empty array Iyy needed for transposing the result
	Iyy = [] 
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
'''
Func: gaussian_1derv_conv
param: Ix : Igx - image convoluted in x- direction with G
	   Iy : Igy - image convoluted in y- direction with G
	   Gx : gaussian 1st derivative in x- direction 
	   Gy : gaussian 1st derivative in y- direction
Desc : returns I_x, I_y, convoluted images with 1st derivative of gaussian in x and y direction
'''

def gaussian_1derv_conv(Ix,Iy,Gx,Gy):

	#Creating an empty array Iyy needed for transposing the result
	Iyy = []
	S = np.shape(Ix)
	I_x = []
	I_y = []
	#computing for I_x
	for i in range(0, S[0]):
		I1 = (np.convolve(np.array(Ix)[i,:], Gx, 'same'))
		I_x.append(np.array(I1))

	#Computing for I_y
	for i in range(0, S[1]):
		I2 = (np.convolve(np.array(Iy)[:,i], Gy, 'same'))
		Iyy.append(np.array(I2))
		I_y = np.transpose(Iyy)

	return(I_x, I_y)
'''
Func: gaussian_2derv_conv
param: Ix: image convoluted with 1st derivative of gaussian
	   Iy: image convoluted with 1st derivative of gaussian
	   Gx: 2nd derivative gaussian kernel in x- direction
	   Gy: 2nd derivative gaussian kernel in y- direction
'''

def gaussian_2derv_conv(Ix,Iy,Gx,Gy):

	#Creating an empty array Iyy needed for transposing the result
	Iyy = Iyyy = []
	S = np.shape(Ix)
	I_x = []
	I_y = Ixy = []
	#computing for I_x
	for i in range(0, S[0]):
		I1 = (np.convolve(np.array(Ix)[i,:], Gx, 'same'))
		I_x.append(np.array(I1))

	#Computing for I_y
	for i in range(0, S[1]):
		I2 = (np.convolve(np.array(Iy)[:,i], Gy, 'same'))
		Iyy.append(np.array(I2))
		I_y = np.transpose(Iyy)

	#Computing for I_xy
	for i in range(0, S[1]):
		I3 = (np.convolve(np.array(Ix)[:,i], Gy, 'same'))
		Iyyy.append(np.array(I3))
		Ixy = np.transpose(Iyyy)

	return(I_x, I_y, Ixy)

def Hessian_matrix(Image):
	start = 0
	start = time.start()
	I = np.array(Image)
	S = np.shape(I)

	g = np.ones((1,5), np.float32)
	g_1d = np.ones((1,5), np.float32)
	g_2d = np.ones((1,5), np.float32)

	g, g_1d, g_2d = gaussian_mask(sigma)

	G = g.flatten()
	G1dx = g_1d.flatten()
	G1dy = np.transpose(G1dx)
	G2dx = g_2d.flatten()
	G2dy = np.transpose(G2dx)

	#to smoothen the image
	Igx = Igy = []
	Igx, Igy = gaussian_conv(I, G, S)

	#convoluting Image with 1st derivative of I
	Ix = Iy = []
	Ix, Iy = gaussian_1derv_conv(Igx, Igy, G1dx, G1dy)

	Ixx = Iyy= Ixy =[]
	Ixx, Iyy, Ixy = gaussian_2derv_conv(Ix, Iy, G2dx, G2dy)

	t = 87
	x=y=[]
	for i in range(S[0]):
		for j in range(S[1]):
			Hessian = ([Ixx[i][j], Ixy[i][j]], [Ixy[i][j], Iyy[i][j]])
			H = np.linalg.eigvals(Hessian)
			if(abs(H[1])>t and abs(H[0])>t):
				y.append[i]
				x.append[j]
	plt.figure()
	plt.imshow(I, cmap = cm.gray)
	plt.plot(x,y, 'ro')
	plt.show()
	return = time.start() - start
sigma =1.5
Image1 = Image.open("input1.png").convert('L')
time1 = Hessian_matrix(Image1)
Image2 = Image.open("input2.png").convert('L')
time2 = Hessian_matrix(Image2)
Image3 = Image.open("input3.png").convert('L')
time3 = Hessian_matrix(Image3)
print ("time for Image1", time1)
print("time for Image2", time2)
print("time for Image3", time3)