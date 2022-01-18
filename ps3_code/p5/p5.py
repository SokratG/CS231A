from shutil import ExecError
import numpy as np
import cv2


def draw_tracks(frame_num, frame, mask, points_prev, points_curr, color):
    """Draw the tracks and create an image.
    """
    for i, (p_prev, p_curr) in enumerate(zip(points_prev, points_curr)):
        a, b = p_curr.ravel()
        c, d = p_prev.ravel()
        mask = cv2.line(mask,  (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 3, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    cv2.imwrite('frame_%d.png' % frame_num, img)
    return img


def Q5_A():
    """Code for question 5a.

    Output:
      p0, p1, p2: (N,2) list of numpy arrays representing the pixel coordinates of the
      tracked features.  Include the visualization and your answer to the
      questions in the separate PDF.
    """
    # params for ShiTomasi corner detection
    feature_params = dict(
        maxCorners=200,
        qualityLevel=0.01,
        minDistance=7,
        blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(75, 75),
        maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01),
        flags=(cv2.OPTFLOW_LK_GET_MIN_EIGENVALS))

    # Read the frames.
    frame1 = cv2.imread('p5/p5_data/rgb1.png')
    frame2 = cv2.imread('p5/p5_data/rgb2.png')
    frame3 = cv2.imread('p5/p5_data/rgb3.png')
    frames = [frame1, frame2, frame3]

    # Convert to gray images.
    old_frame = frames[0]
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create some random colors for drawing
    color = np.random.randint(0, 255, (200, 3))

    # Create a mask image for drawing purposes
    mask = np.zeros_like(frame1)
    pts = [p0.reshape(-1, 2)]
    
    for i,frame in enumerate(frames[1:]):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Fill in this code
        # BEGIN YOUR CODE HERE
        pt, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        if pt is not None:
            points_prev = p0[st == 1]
            points_curr = pt[st == 1]
            pts.append(pt.reshape(-1, 2))
        
        old_gray = frame_gray.copy()
        p0 = points_curr.reshape(-1, 1, 2)
        frame_num = i
        #Once you compute the new feature points for this frame, comment this out
        #to save images for your PDF:
        draw_tracks(frame_num, frame, mask, points_prev, points_curr, color)
        # END YOUR CODE HERE
    p0 = np.array(pts[0])
    p1 = np.array(pts[1])
    p2 = np.array(pts[2])
    
    return p0, p1, p2


def Q5_B(p0, p1, p2, intrinsic):
    """Code for question 5b.

    Note that depth maps contain NaN values.
    Features that have NaN depth value in any of the frames should be excluded
    in the result.

    Input:
      p0, p1, p2: (N,2) numpy arrays, the results from Q2_A.
      intrinsic: (3,3) numpy array representing the camera intrinsic.

    Output:
      p0, p1, p2: (N,3) numpy arrays, the 3D positions of the tracked features
      in each frame.
    """
    depth0 = np.loadtxt('p5/p5_data/depth1.txt')
    depth1 = np.loadtxt('p5/p5_data/depth2.txt')
    depth2 = np.loadtxt('p5/p5_data/depth3.txt')

    # Fill in this code
    # BEGIN YOUR CODE HERE
    n, _ = p0.shape
    homo_coord = np.ones((n, 1))
    K_inv = np.linalg.inv(intrinsic)
    
    ph0 = np.hstack((p0, homo_coord))
    ph1 = np.hstack((p1, homo_coord))
    ph2 = np.hstack((p2, homo_coord))
    
    for i in range(n):
        # p0
        y, x = np.int32(ph0[i][:2])
        if (not np.isnan(depth0[x, y])):
            ph0[i] *= depth0[x, y]

        # p1
        y, x = np.int32(ph1[i][:2])
        if (not np.isnan(depth1[x, y])):
            ph1[i] *= depth1[x, y]
            
        # p2
        y, x = np.int32(ph2[i][:2])
        if (not np.isnan(depth2[x, y])):
            ph2[i] *= depth2[x, y]   
            
    p0 = K_inv.dot(ph0.transpose())
    p1 = K_inv.dot(ph1.transpose())
    p2 = K_inv.dot(ph2.transpose())
    # END YOUR CODE HERE
    

    return p0, p1, p2


if __name__ == "__main__":
    p0, p1, p2 = Q5_A()
    intrinsic = np.array([[486, 0, 318.5],
                          [0, 491, 237],
                          [0, 0, 1]])
    p0, p1, p2 = Q5_B(p0, p1, p2, intrinsic)
