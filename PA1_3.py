#Entropy 
#__author__ = 'khushboo_agarwal'
#from __future__ import division
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import os,sys
import scipy

#def max_entropy(img):

def func_1(img):
	i1 = np.array(img)
	S = np.shape(i1)										#getting the shape of the input image

	'''
	to get the histogram of the image: since we know that the input image is of gray level it has 256 intensity levels and if we plot a histogram of 
	an image with 256 intensity level then its range could be the 0, ... , s[0]*s[1], which gives the no. of pixels in the image. So first we create an empty 
	array of size 256. Then we calculate the intensity/pixel value and then create a histogram.  Now we need to calculate the probability , ni/(s[0]*s[1]). 
	Since s[0]s[1] is 154401, I have directly divided. Also to check if the calculations are correct, use np.sum(M) = 1 as a checker for all your images; and Also
	to check if you have plotted the histogram correctly, we can use np.sum(H) = S[0]*S[1] always !!!. Now we do not know the threshold value to separate the foreground
	from the background, so we are just going to assume the threshold be be -1 for now and continue separating th 
	probabilties of foreground and the background. 
	'''
	
	H = np.zeros(256)
	intensity = 0
	#getting the H (histogram image)
	for i in range(S[0]):
		for j in range(S[1]):
			intensity = i1[i,j]
			H[intensity] = H[intensity]+1

	M = np.zeros(256)
	for intensity in range(256):
		M[intensity] = H[intensity]/154401

	threshold = -1
	maximum_entropy	  = 0
	for i in range(0, 255):
		sum_p = 0
		prob_bg = []				#background prob.
		prob_fg = []				#foreground prob.
		for j in range(0, i+1):
			sum_p = sum_p + M[j]

		T1 = 0						#calculating the background prob.
		for x in range(0, i+1):
			if(sum_p == 0):
				T1 = 0
			else:
				T1 = M[x]/sum_p
			prob_bg.append(T1)

		T2 = 0
		for x in range(i+1, 255):	#calculating the foreground prob.
			if(sum_p == 1):
				T2 = 0
			else:
				T2 = M[x]/(1-sum_p)
			prob_fg.append(T2)

		sumea = 0 		#entropy A
		sumeb = 0 		#entropy B

		for x in range(0, len(prob_bg)):
			if(prob_bg[x]== 0):
				sumea = sumea + 0
			else:
				sumea = sumea + prob_bg[x]*np.log(prob_bg[x])
		for y in range(0, len(prob_fg)):
			if(prob_fg[y] == 0):
				sumeb = sumeb + 0
			else:
				sumeb = sumeb + prob_fg[y]*np.log(prob_fg[y])

		if(maximum_entropy < -(sumea) - (sumeb)):
			maximum_entropy = -(sumea) -(sumeb)
			threshold = i
	
	print ("The threshold value is", threshold)

	im_new = np.copy(i1)
	for i in range(S[0]):
		for j in range(S[1]):
			if(i1[i,j] > threshold):
				im_new[i,j] = 1
			else:
				im_new[i,j] = 0
	plt.figure()
	plt.imshow(im_new, cmap = cm.gray)
	plt.show()

#load the images:
image_1 = Image.open("for3_1.jpg").convert('L')
func_1(image_1)
image_2 = Image.open("for3_2.jpg").convert('L')
func_1(image_2)
image_3 = Image.open("for3_3.jpg").convert('L')
func_1(image_3)