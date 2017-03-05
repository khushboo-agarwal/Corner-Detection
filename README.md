# Corner-Detection
[1 pts] Implement corner detection algorithm based on Hessian matrix (H) computation. 
Note that Hessian matrix is defined for a given image I at a pixel p as 􏰇Ixx(p) Ixy(p)􏰈 H1(p) = Ixy(p) Iyy(p) , (0.1) such that eigen-decomposition (spectral decomposition) of this matrix yields two eigenvalues as: λ1 and λ2. 
If both λ1,λ2 are large, we are at a corner. Provide the detected corners in the resulting output images in color. 
[1 pt] Implement Harris Corner Detection algorithm for the same input images you used in previous question. 
Rather than considering the Hessian of the original image I (i.e. second-order derivatives), we use the first-order derivatives of the 
smoothed version L(p, σ) for some Gaussian filter with standard deviation σ > 0. Note that you need to construct the following matrix for 
every pixel p, ￼􏰇 L2x(p, σ) Lx(p, σ)Ly(p, σ)􏰈 Lx(p, σ)Ly(p, σ) L2y(p, σ) , (0.2) H2(p, σ) = where L is obtained after smoothing I with 
Gaussian filter G. Now, instead of calculating those eigenvalues we computed in previous question, we will consider the cornerness measure 
as Cornerness(p, σ, α) = Det(H2) − α.Tr(H2), (0.3) where Det and Tr indicate the determinant and trace of the matrix H2, respectively. 
Please use non-negative α ≈ 1/25 as a starting value and try to optimize it by trying different values and comment about it. 
Provide the detected corners in the resulting output image in color. [1 pt] In the previous question, replace cornerness measure with the 
following: Cornerness(p, σ, α) = λ1λ2 − α(λ1 + λ2), (0.4) and determine the efficiency of this system and the system in the previous question by measuring and reporting the time. You are supposed to get the same results in accuracy but different results in efficiency.
