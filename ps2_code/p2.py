import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    h, w = points1.shape
    A = np.zeros((h, w*w))
    for i in range(h):
        p1x, p1y = points1[i][0], points1[i][1]
        p2x, p2y = points2[i][0], points2[i][1]
        A[i] = np.array([p1x*p2x, p2x*p1y, p2x, p1x*p2y, p1y*p2y, p2y, p1x, p1y, 1.])
    
    U, S, V = np.linalg.svd(A)
    f = V[-1]
    F_h = f.reshape(3, 3)

    U, S, V = np.linalg.svd(F_h)
    S_ = np.zeros((3, 3))
    S_[0, 0] = S[0]
    S_[1, 1] = S[1]
    F = np.dot(U, np.dot(S_, V))

    return F
    

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    h, _ = points1.shape
    mean1 = np.mean(points1, axis=0)
    mean2 = np.mean(points2, axis=0)
    N = h
    scale1 = np.sqrt((2*N) / np.sum((points1 - mean1)**2))
    scale2 = np.sqrt((2*N) / np.sum((points2 - mean2)**2))
    mean1_x, mean1_y  = mean1[0], mean1[1]
    mean2_x, mean2_y  = mean2[0], mean2[1]

    T = np.array([[scale1, 0., -mean1_x*scale1],
                  [0., scale1, -mean1_y*scale1],
                  [0., 0., 1.]])
    Tp = np.array([[scale2, 0., -mean2_x*scale2],
                  [0., scale2,  -mean2_y*scale2],
                  [0., 0., 1.]])

    q1 = T.dot(points1.transpose())
    q2 = Tp.dot(points2.transpose())

    Fq = lls_eight_point_alg(q1.T, q2.T)

    TpdotFq = Tp.transpose().dot(Fq)
    F = TpdotFq.dot(T)

    return F
    

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):

    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a,b,c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c-a*x)/b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x,y,_ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    # We change the figsize because matplotlib has weird behavior when 
    # plotting images of different sizes next to each other. This
    # fix should be changed to something more robust.
    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2 , 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines. Compute just the average distance
from points1 to their corresponding epipolar lines (which you get from points2).
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    N = points1.shape[0] + points2.shape[0]

    # distance = (a*x0 + b*y0 + c) / sqrt(a**2 + b**2)
    calc_dist = lambda line, pt: np.abs(pt.dot(line)) / np.sqrt(line[0]**2 + line[1]**2)
    
    pts = zip(points1, points2)
    result = 0.0
    for pt1, pt2 in pts:
        lp = F.dot(pt1)
        l = F.T.dot(pt2)
        result += (calc_dist(lp, pt2) + calc_dist(l, pt1))

    # average distance
    result = result / N

    return result

    

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS  8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized)
        print("Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
