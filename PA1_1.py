'''
author      :	Khushboo Agarwal
date created:   9-14-2016
title		:	Canny Edge Detection 
version		:	1
'''

from __future__ import division						#this will always get real values for division

#import header files required
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os,sys

#creating sigma as a global variable which will be used for trial and error and also as a parameter for calling functions

L = 65
H = 75

#Part 2: Create a one-dimensional Gaussian mask G to convolve with I. The standard deviation(s) of this Gaussian is a parameter to the edge detector
#Part 3: Create a one-dimensional mask for the first derivative of the Gaussian in the x and y directions; call these Gx

'''
While making a Gaussian mask it is important to keep the size in mind. The size if too small would blur the edges also, and other important features which we 
may need. Also if it is too big then it will leave out blurring and also computations would be too much. Gaussian function is defined as a bell curve which usually 
has a mean around the 0. Also the standrd deviation defines how spread the function is around its mean, as in its width. The higher the standard deviation, 
the more it is spread [Source: Wikipedia, https://en.wikipedia.org/wiki/Gaussian_function]. 

To define the Gaussian mask, we use a threshold Th = e^(-x*x)/(2*sigma*sigma); with the figure as reference we conclude that Th should be a real number between
0 and 1. For Th value = 0.05 is suiting based on trial and errors. Now the width-half is calculated using the formula sqrt(-log(Th)*2*sigma^2) as this gives a 
value where the curve value drops below Th. The formula is from [Source: Wikipedia, https://en.wikipedia.org/wiki/Gaussian_function]. 
We use a 2*round(width-half)+1 to calculate the size of the mask and include both the positive and negative side of the curve this way. 

The radius is calculated as int(width/2) as that gives the exact half of the size for making the calculation easier.

After trial and error method and running the Gaussian_size function with various values of sigma, following observations were made:
	sigma values -->	0.5		1 		1.25 	1.5 	2
	
	width_half			2.0 	3.0 	4.0     5.0     6.0
	width 				5		7		9 		11 		13
	radius				2 		3		4 		5 		6

'''
#function to calculate the Gaussian function
def Gaussian_function(x, sigma):
	if sigma == 0:
		return 0
	else:
		g = math.exp(-x*x)/(2*sigma*sigma) 
		return g 									#this functions returns the value of the gaussian function as given by the formula based on x and sigma value

def Gaussian_size(sigma):
	#determining the size of the kernel
	width_half = round(math.sqrt(-math.log(0.05)*2.0*(sigma*sigma)))+1
	width 	   = 2*int(width_half)+1
	radius	   = int(width/2)
	return (width_half, width, radius)

def Gaussian_mask(sigma, radius, width):
	#creating arrays for gaussian kernel, gaussian derivative
	gaussian_kernel 	= np.ones((1, width), np.float32)
	gaussian_derivative = np.ones((1, width), np.float32)
				
	
	# if we use only the integer values of x, then the mask will not be robust, so instead I have used a mean value centered at i finally to append into gaussian kernel
	for i in range(-radius, radius+1):
		gaussian_1 = Gaussian_function(i,sigma)
		gaussian_2 = Gaussian_function(i-0.5,sigma)
		gaussian_3 = Gaussian_function(i+0.5,sigma)
		gauss      = (gaussian_1+gaussian_2+gaussian_3)/3

		#calculating the derivative gaussian with formula (-x/pow(sigma,3)*sqrt(2*pi))*Gaussian_function
		gaussian_derivative[0][i+radius] = ((-i)/(sigma*sigma*sigma*(math.sqrt(2*math.pi))))

		#calculating the final kernel with a constant of 1/sqrt(2*pi*sigma*sigma)
		gaussian_kernel[0][i+radius] = gauss*(1/(math.sqrt(2.0 * math.pi * sigma * sigma)))

	    
	return(gaussian_kernel, gaussian_derivative) 
'''
for sigma = 1; gaussian function values after the mean method were found to be:       gauss          = [0.000343108176284, 0.0209408862645, 0.208679908134, 0.42626692769, 0.208679908134, 0.0209408862645,  0.000343108176284]
for sigma = 1; gaussian_kernel values after including the constant were found to be: gaussian_kernel = [  1.36880364e-04,  8.35420471e-03,  8.32512379e-02, 1.70055896e-01, 8.32512379e-02,  8.35420471e-03,   1.36880364e-04]
for sigma = 1; gaussian derivative values after differentiating w.r.t x were : 	 gaussian_derivative = [ 1.19682682,       0.79788458,      0.39894229,      0.,            -0.39894229,      -0.79788458,        -1.19682682]
'''


#Part 4: Convolve the image I with G along the rows to give the x component image (Ix), and down the columns to give the y component image (Iy).

def gaussian_conv(I, G, S):

	#Creating an empty array Iyy needed for transposing the result
	Iyy = [] 	
	S = np.shape(I)	
	#Creating an empty array Ix: I[i,:]*G
	Ix = []	

	#Creating an empty array Iy: I[:,i]*G
	Iy = []			

	#convolving in the x direction
	for i in range(S[0]):
		I1 = (np.convolve(I[i,:], G, 'same'))
		Ix.append(np.array(I1))

	#convolving in the y direction
	for i in range(S[1]):
		I2 = (np.convolve(I[:,i], G, 'same'))
		Iyy.append(np.array(I2))
		Iy = np.transpose(Iyy)
	
	return(Ix, Iy)

#Part 5: Convolve Ix with Gx to give I_x, the x component of I convolved with the derivative of the Gaussian, andconvolve Iy with Gy to give I_y, y component of I convolved with the derivative of the Gaussian.

def gaussian_derv_conv(Ix,Iy,Gx,Gy, S):

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
	

#Part 6: Compute the magnitude of the edge response by combining the x and y components. The magnitude of theresult can be computed at each pixel (x; y) as: M(x; y) = sqrt(I_x^2+I_y^3)

def image_magnitude_orientation(I_x, I_y, S):
	#S = np.shape(I_x)
	M = np.ones((S[0], S[1]), np.float32)			#since the size of I_x = size of I_y
	O = np.ones((S[0], S[1]), np.float32)
	for i in range(0, S[0]):
		for j in range(0, S[1]):
			M[i][j] = math.sqrt((((I_x[i][j]**2) + ((I_y[i][j])**2))))
			O[i][j] = math.degrees(math.atan2(I_y[i][j], I_x[i][j]))+180
	return (M, O)
'''
for calculations of orientation, we use atan2 function because it returns the value of degrees in -pi to pi.
I am then converting it to degrees and adding 180 so as to get values between 0 - 360 degrees which can be plotted accordingly
'''


#Part 7: Implement non-maximum suppression algorithm

'''
Non-maximum supression is implemented on image M to get all edges one pixel thick which is an important feature of Canny Edge Detection
lets see how we can simplify this. We have O matrix with values from 0 to 360. So I simplify the values into four categories [0, 1, 2, 3]
each category has about 90 total degrees defined. 45 degrees towards right and 45 degrees towards left which can also be defined in terms if 
0 deg is east, 90 deg is north, 180 deg is west, 270 deg is south and again 360 deg is east. In these terms categories are defined as 
[0, 1, 2, 3] = [east + west, north-east + south-west, north + south, north-west + south-east]. Accordingly we check for neighbouring values
From the implementation of following function it would be 
much clearer. 
'''
def non_maximum_suppression(O, M):

	M1 = np.copy(M)
	Sm = np.shape(M)
	for i in range(0, Sm[0]):
		for j in range(0, Sm[1]):
			if ((i>0 and i<(Sm[0]-1)) and (j>0 and j<(Sm[1]-1))):
				if((0<O[i][j]<22.5) or (157.5<O[i][j]<202.5) or (337.5<O[i][j]<360)):		#category 0 : 0 bin
					if(M[i][j]<M[i+1][j] or M[i][j]<M[i-1][j]):
						M1[i][j] = 0
				if((22.5<O[i][j]<67.5) or (202.5<O[i][j]<247.5)):							#category 1
					if (M[i][j]<M[i+1][j-1] or M[i][j]<M[i-1][j+1]):
						M1[i][j] = 0
				if((67.5<O[i][j]<112.5) or (247.5<O[i][j]<292.5)):							#category 2
					if (M[i][j]<M[i][j-1] or M[i][j]<M[i][j+1]):
						M1[i][j] = 0
				if((112.5<O[i][j]<157.5) or (292.5<O[i][j]<337.5)):							#category 3
					if (M[i][j]<M[i][j+1] or M[i][j]<M[i][j-1]): 
						M1[i][j] = 0
	return M1
	



#Part 8: Apply Hysteresis thresholding to obtain final edge-map.
'''
Hysteresis Thresholding is a connected component method of finding the edges in an image.
'''

def Hysteresis_Threshold(M1, L, H):
	Sm = np.shape(M1)
	
	for i in range(Sm[0]):
		for j in range(Sm[1]):
			if (M1[i][j]<=L):
				M1[i][j] = 0
			elif (M1[i][j]>=H):
				M1[i][j] = 255
			elif (M1[i][j]>=L and(M1[i][j]<=H)):
				if ((i>0 and i<(Sm[0]-1)) and (j>0 and j<(Sm[1]-1))):
					if ((M1[i-1][j-1]>H) or (M1[i][j-1]>H) or (M1[i+1][j-1]>H) or (M1[i-1][j]>H) or (M1[i+1][j]>H) or (M1[i-1][j+1]>H) or (M1[i][j+1]>H) or (M1[i+1][j+1]>H)):
						M1[i][j] = 255
	return M1



def Plotting_Results(Ix, Iy, I_x, I_y, M, M1):
	plt.figure()   									# plotting all the images in subplot
	plt.subplot(2,3,1)
	plt.title('image X component of the convoln with a Gaussian')
	plt.imshow(Ix,cmap = cm.gray)
	plt.subplot(2,3,2)
	plt.title('image Y component of the convoln with a Gaussian')
	plt.imshow(Iy,cmap = cm.gray)
	plt.subplot(2,3,3)
	plt.title('Ix convoluted with Gx')
	plt.imshow(I_x,cmap = cm.gray)
	plt.subplot(2,3,4)
	plt.title('Iy convoluted with Gy')
	plt.imshow(I_y,cmap = cm.gray)
	plt.subplot(2,3,5)
	plt.title('Magnitude image ')
	plt.imshow(M,cmap = cm.gray)
	plt.subplot(2,3,6)
	plt.title('Canny Edge detection')
	plt.imshow(M1,cmap = cm.gray)
	plt.show()





def Canny_Edge(inp_I, sigma):
	I = np.array(inp_I)
	

	width_half, width, radius = Gaussian_size(sigma)

	#creating arrays for gaussian kernel, gaussian derivative
	gaussian_kernel 	= np.ones((1, width), np.float32)
	gaussian_derivative = np.ones((1, width), np.float32)

	gaussian_kernel, gaussian_derivative = Gaussian_mask(sigma, radius, width)
	G  = gaussian_kernel.flatten()
	Gx = gaussian_derivative.flatten()									#result for part 2 and part 3
	Gy = np.transpose(Gx)

	#to create the Ix and Iy arrays we need the shape of the image component, to convolve in the x and y direction
	S = np.shape(I)

	#Creating an empty array Ix: I[i,:]*G
	Ix = []	

	#Creating an empty array Iy: I[:,i]*G
	Iy = []	

	Ix, Iy = gaussian_conv(I,G, S)
	

	#Creating an empty array I_x: Ix*Gx
	I_x = []

	#Creating an empty array I_y: Iy*Gy
	I_y = []



	I_x, I_y = gaussian_derv_conv(Ix, Iy, Gx, Gy, S)
	

	#creating the magnitude image M and orientation image O
	M = np.ones((S[0], S[1]), np.float32)			#since the size of I_x = size of I_y
	O = np.ones((S[0], S[1]), np.float32)

	M, O = image_magnitude_orientation(I_x, I_y, S)

	M1 = non_maximum_suppression(O, M)
	
	M1 = Hysteresis_Threshold(M1, L, H)

	'''Plotting_Results(Ix, Iy, I_x, I_y, M, M1)'''
	plt.figure()
	plt.imshow(M1, cmap = cm.gray)
	plt.show()

#this is the function definition to be used in the second program.  
def Canny_Edge_(I_1, sigma, H, L):

	I = np.array(I_1)
	

	width_half, width, radius = Gaussian_size(sigma)

	#creating arrays for gaussian kernel, gaussian derivative
	gaussian_kernel 	= np.ones((1, width), np.float32)
	gaussian_derivative = np.ones((1, width), np.float32)

	gaussian_kernel, gaussian_derivative = Gaussian_mask(sigma, radius, width)
	G  = gaussian_kernel.flatten()
	Gx = gaussian_derivative.flatten()									#result for part 2 and part 3
	Gy = np.transpose(Gx)

	#to create the Ix and Iy arrays we need the shape of the image component, to convolve in the x and y direction
	Sh = np.shape(I)

	#Creating an empty array Ix: I[i,:]*G
	Ix = []	

	#Creating an empty array Iy: I[:,i]*G
	Iy = []	

	Ix, Iy = gaussian_conv(I,G, Sh)

	#Creating an empty array I_x: Ix*Gx
	I_x = []

	#Creating an empty array I_y: Iy*Gy
	I_y = []



	I_x, I_y = gaussian_derv_conv(Ix, Iy, Gx, Gy, Sh)

	#creating the magnitude image M and orientation image O
	M = np.ones((Sh[0], Sh[1]), np.float32)			#since the size of I_x = size of I_y
	O = np.ones((Sh[0], Sh[1]), np.float32)

	M, O = image_magnitude_orientation(I_x,I_y, Sh)

	M1 = non_maximum_suppression(O, M)
	
	M1 = Hysteresis_Threshold(M1, L, H)

	return M1

	Plotting_Results(Ix, Iy, I_x, I_y, M, M1)



if __name__ == "__main__":
	# Part 1: Read a gray scale image you can find from Berkeley Segmentation Dataset, Training images, store it as a matrix named I.
	sigma = 1
	print ("For sigma:", sigma)
	inp_I = Image.open("grayimage_1.jpg").convert('L')
	Canny_Edge(inp_I, sigma)

	inp_I1 = Image.open("grayimage_2.jpg").convert('L')
	Canny_Edge(inp_I1, sigma)
			
	inp_I2 = Image.open("grayimage_3.jpg").convert('L')
	Canny_Edge(inp_I2, sigma)
			
	sigma = 1.25
	print ("For sigma:", sigma)
	Canny_Edge(inp_I, sigma)
	Canny_Edge(inp_I1, sigma)
	Canny_Edge(inp_I2, sigma)

	sigma = 1.5
	print ("For sigma", sigma)
	Canny_Edge(inp_I, sigma)
	Canny_Edge(inp_I1, sigma)
	Canny_Edge(inp_I2, sigma)

'''else:
	Canny_Edge_(inp_I, 1)'''


