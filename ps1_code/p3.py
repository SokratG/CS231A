# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    # BEGIN YOUR CODE HERE
    line1 = (points[0], points[1]) # p1, p2
    line2 = (points[2], points[3]) # p3, p4
    # for simplicity
    x1, y1 = line1[0][0], line1[0][1]
    x2, y2 = line1[1][0], line1[1][1]
    x3, y3 = line2[0][0], line2[0][1]
    x4, y4 = line2[1][0], line2[1][1]
    
    D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4) 

    if D == 0.0:
        raise ZeroDivisionError() # or add np.finfo(np.float64).eps

    # intersection point coordinate
    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D
    vanishing_point = np.array([Px, Py])
    
    return vanishing_point
    # END YOUR CODE HERE

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    # BEGIN YOUR CODE HERE
    v1, v2, v3 = vanishing_points[0], vanishing_points[1], vanishing_points[2]
    # [HZ] section 8.8 - algorithm 8.2
    # vi * w * vj = 0
    # |vi.x|   |w1  0  w4|                                                                     |w1|
    # |vi.y| * |0  w1  w5| * |vj.x vj.y 1| = |vi.x*vj.x+vi.y*vj.y, vi.x+vj.x,  vi.y+vj.y, 1| * |w4|
    # | 1.0|   |w4 w5  w6|                                                                     |w5|
    #                                                                                          |w6|
    # A * w = 0  

    A = np.zeros((3, 4), dtype=np.float64)

    A[0] = np.array([v1[0]*v2[0]+v1[1]*v2[1], v1[0]+v2[0], v1[1]+v2[1], 1.])
    A[1] = np.array([v1[0]*v3[0]+v1[1]*v3[1], v1[0]+v3[0], v1[1]+v3[1], 1.])
    A[2] = np.array([v2[0]*v3[0]+v2[1]*v3[1], v2[0]+v3[0], v2[1]+v3[1], 1.]) 

    U, S, V = np.linalg.svd(A)
    W = V[-1] # the last column is the solution
    
    w = np.array([[W[0], 0., W[1]],
                  [0., W[0], W[2]],
                  [W[1], W[2], W[3]]]) 
    
    K = np.linalg.inv(np.linalg.cholesky(w).T)
    # the last value must equal 1.0 -> divide by last element of the matrix
    K = K / K[-1, -1]
    return K 
    # END YOUR CODE HERE

'''
COMPUTE_ANGLE_BETWEEN_PLANES
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    # BEGIN YOUR CODE HERE
    w = K.dot(K.T)
    
    vanishing_point1 = np.array([vanishing_pair1[0][0], vanishing_pair1[0][1], 1.], dtype=np.float64) 
    vanishing_point2 = np.array([vanishing_pair1[1][0], vanishing_pair1[1][1], 1.], dtype=np.float64) 
    vanishing_point3 = np.array([vanishing_pair2[0][0], vanishing_pair2[0][1], 1.], dtype=np.float64) 
    vanishing_point4 = np.array([vanishing_pair2[1][0], vanishing_pair2[1][1], 1.], dtype=np.float64)

    vanishing_line1 = np.cross(vanishing_point1, vanishing_point2)
    vanishing_line2 = np.cross(vanishing_point3, vanishing_point4)
    
    n = vanishing_line1.T.dot(w.dot(vanishing_line2))
    d = np.sqrt(vanishing_line1.T.dot(w.dot(vanishing_line1))) * np.sqrt(vanishing_line2.T.dot(w.dot(vanishing_line2)))
    
    angle = np.arccos(n / d)
    # convert to degrees
    angle = angle / np.pi * 180.0 
    return angle
    # END YOUR CODE HERE

'''
COMPUTE_ROTATION_MATRIX_BETWEEN_CAMERAS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    # BEGIN YOUR CODE HERE
    K_inv = np.linalg.inv(K)

    v1 = np.array([vanishing_points1[0][0], vanishing_points1[0][1], 1.])
    v2 = np.array([vanishing_points1[1][0], vanishing_points1[1][1], 1.])
    v3 = np.array([vanishing_points1[2][0], vanishing_points1[2][1], 1.])

    v1p = np.array([vanishing_points2[0][0], vanishing_points2[0][1], 1.])
    v2p = np.array([vanishing_points2[1][0], vanishing_points2[1][1], 1.])
    v3p = np.array([vanishing_points2[2][0], vanishing_points2[2][1], 1.])

    V = np.stack((v1, v2, v3)).transpose()
    Vp = np.stack((v1p, v2p, v3p)).transpose()

    d = K_inv.dot(V) / np.linalg.norm(K_inv.dot(V))
    dp = K_inv.dot(Vp) / np.linalg.norm(K_inv.dot(Vp))

    R = dp.dot(np.linalg.pinv(d))
    return R
    # END YOUR CODE HERE

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print()
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
