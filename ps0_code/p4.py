# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def part_a():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.
    # Hint: use io.imread to read in the image file

    img1 = None

    # BEGIN YOUR CODE HERE
    img1 = io.imread('image1.jpg', as_gray=True)
    u, s, v = np.linalg.svd(img1) # v - transpose
    # END YOUR CODE HERE
    return u, s, v

def part_b(u, s, v):
    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    rank1approx = None

    # BEGIN YOUR CODE HERE
    #recoverimg = np.dot(u, np.dot(np.diag(s), v))
    n = 1
    U = u[:, :n] # first left singular 
    S = s[0] # first singular
    V = v[:n, :] # first right singular
    rank1approx = np.dot(U * S, V)

    # END YOUR CODE HERE
    return rank1approx

def part_c(u, s, v):
    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    rank20approx = None

    # BEGIN YOUR CODE HERE
    n = 20
    S = s[0:n] # n-th left singular 
    U = u[:, :n] # n-th singular 
    V = v[:n, :] # n-th right singular 
    rank20approx = np.dot(U * S, V)
   
    # END YOUR CODE HERE
    return rank20approx

def show(img):
    io.imshow(img)
    io.show()
    return

if __name__ == '__main__':
    u, s, v = part_a()
    rank1approx = part_b(u, s, v)
    #io.imsave('p4b.jpg', rank1approx)
    rank20approx = part_c(u, s, v)
    #io.imsave('p4c.jpg', rank20approx)
    
