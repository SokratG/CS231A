import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
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
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the first image
'''
def compute_epipole(points1, points2, F): 
    lines = F.T.dot(points2.transpose())
    #lines /= np.linalg.norm(lines[:2, :], axis=0)
    U, S, V = np.linalg.svd(lines.transpose())
    e = V[-1]
    e = e / e[-1]
    return e
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images. Do not divide the homographies by their 2,2 entry.
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    h, w = im2.shape
    T = np.array([[1, 0, -w/2],
                  [0, 1, -h/2],
                  [0, 0, 1]])
    e = T.dot(e2)
    e1p, e2p = e[0], e[1]
    alpha = 1 if e1p >= 0 else -1
    s1 = e1p / np.sqrt(e1p**2 + e2p**2)
    s2 = e2p / np.sqrt(e1p**2 + e2p**2)
    R = np.array([[alpha*s1, alpha*s2, 0],
                  [-alpha*s2, alpha*s1, 0],
                  [0, 0, 1]])
    e = R.dot(e)
    f = e[0]

    G = np.eye(3, 3)
    G[2, 0] = -1. / f
    T_inv = np.linalg.inv(T)
    H2 = T_inv.dot(G).dot(R).dot(T)
    e_x = np.array([[0, -e2[2], e2[1]],
                    [e2[2], 0, -e2[0]],
                    [-e2[1], e2[0], 0]])
    
    M = e_x.dot(F) + np.outer(e2, np.array([1., 1., 1.]))
    p_hat = H2.dot(M.dot(points1.transpose())).transpose()
    p_hat_p = H2.dot(points2.transpose()).transpose()
    p_hat_hom = p_hat[:, -1].reshape(p_hat.shape[0], 1)
    p_hat_p_hom = p_hat_p[:, -1].reshape(p_hat_p.shape[0], 1)

    W = p_hat / p_hat_hom
    b = (p_hat_p / p_hat_p_hom)[:, 0] # x
    
    
    a1, a2, a3 = np.linalg.lstsq(W, b)[0]
    Ha = np.array([[a1, a2, a3],
                   [0, 1, 0],
                   [0, 0, 1]])
    H1 = Ha.dot(H2).dot(M)
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'p1/p1_data/set1' # /p1_data/set1
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print('')
    print("H2:\n", H2)

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
