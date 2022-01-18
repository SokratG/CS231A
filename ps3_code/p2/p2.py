import numpy as np
import scipy.io as sio
import argparse
from camera import Camera
from plotting import *


# A very simple, but useful method to take the difference between the
# first and second element (usually for 2D vectors)
def diff(x):
    return x[1] - x[0]


'''
FORM_INITIAL_VOXELS  create a basic grid of voxels ready for carving

Arguments:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

    num_voxels - The approximate number of voxels we desire in our grid

Returns:
    voxels - An ndarray of size (N, 3) where N is approximately equal the 
        num_voxels of voxel locations.

    voxel_size - The distance between the locations of adjacent voxels
        (a voxel is a cube)

Our initial voxels will create a rectangular prism defined by the x,y,z
limits. Each voxel will be a cube, so you'll have to compute the
approximate side-length (voxel_size) of these cubes, as well as how many
cubes you need to place in each dimension to get around the desired
number of voxel. This can be accomplished by first finding the total volume of
the voxel grid and dividing by the number of desired voxels. This will give an
approximate volume for each cubic voxel, which you can then use to find the 
side-length. The final "voxels" output should be a ndarray where every row is
the location of a voxel in 3D space.
'''
def form_initial_voxels(xlim, ylim, zlim, num_voxels):
    volume = diff(xlim) * diff(ylim) * diff(zlim)    
    voxel_size = (volume / num_voxels) ** (1/3)

    x_num = np.int32(diff(xlim) / voxel_size)
    y_num = np.int32(diff(ylim) / voxel_size)
    z_num = np.int32(diff(zlim) / voxel_size)
    
    # add some padding
    x_lbound, x_ubound = xlim[0] + (np.floor(voxel_size)/2.), xlim[1] - (np.floor(voxel_size)/2.) 
    y_lbound, y_ubound = ylim[0] + (np.floor(voxel_size)/2.), ylim[1] - (np.floor(voxel_size)/2.)
    z_lbound, z_ubound = zlim[0] + (np.floor(voxel_size)/2.), zlim[1] - (np.floor(voxel_size)/2.)

    voxel_x = np.linspace(x_lbound, x_ubound, x_num)
    voxel_y = np.linspace(y_lbound, y_ubound, y_num)
    voxel_z = np.linspace(z_lbound, z_ubound, z_num)

    VX, VY, VZ = np.meshgrid(voxel_x, voxel_y, voxel_z)
    
    xsize, ysize, zsize = VX.shape
    voxels = np.zeros((xsize*ysize*zsize, 3))
    voxels[:, 0] = VX.flatten()
    voxels[:, 1] = VY.flatten()
    voxels[:, 2] = VZ.flatten()

    return voxels, voxel_size
   
    


'''
GET_VOXEL_BOUNDS: Gives a nice bounding box in which the object will be carved
from. We feed these x/y/z limits into the construction of the inital voxel
cuboid. 

Arguments:
    cameras - The given data, which stores all the information
        associated with each camera (P, image, silhouettes, etc.)

    estimate_better_bounds - a flag that simply tells us whether to set tighter
        bounds. We can carve based on the silhouette we use.

    num_voxels - If estimating a better bound, the number of voxels needed for
        a quick carving.

Returns:
    xlim - The limits of the x dimension given as [xmin xmax]

    ylim - The limits of the y dimension given as [ymin ymax]

    zlim - The limits of the z dimension given as [zmin zmax]

The current method is to simply use the camera locations as the bounds. In the
section underneath the TODO, please implement a method to find tigther bounds
by doing a quick carving of the object on a grid with very few voxels. From this coarse carving,
we can determine tighter bounds. Of course, these bounds may be too strict, so we should have 
a buffer of one voxel_size around the carved object. 
'''
def get_voxel_bounds(cameras, estimate_better_bounds = False, num_voxels = 4000):
    camera_positions = np.vstack([c.T for c in cameras])
    xlim = [camera_positions[:,0].min(), camera_positions[:,0].max()]
    ylim = [camera_positions[:,1].min(), camera_positions[:,1].max()]
    zlim = [camera_positions[:,2].min(), camera_positions[:,2].max()]

    # For the zlim we need to see where each camera is looking. 
    camera_range = 0.6 * np.sqrt(diff( xlim )**2 + diff( ylim )**2)
    for c in cameras:
        viewpoint = c.T - camera_range * c.get_camera_direction()
        zlim[0] = min( zlim[0], viewpoint[2] )
        zlim[1] = max( zlim[1], viewpoint[2] )

    # Move the limits in a bit since the object must be inside the circle
    xlim = xlim + diff(xlim) / 4 * np.array([1, -1])
    ylim = ylim + diff(ylim) / 4 * np.array([1, -1])

    if estimate_better_bounds:
        init_voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)
        for camera in cameras:
            voxels = carve(init_voxels, camera)
        
        lower_voxel = np.min(voxels, axis=0) 
        upper_voxel = np.max(voxels, axis=0)
        
        xlim = np.array([lower_voxel[0], upper_voxel[0]])
        ylim = np.array([lower_voxel[1], upper_voxel[1]])
        zlim = np.array([lower_voxel[2], upper_voxel[2]])
        
    return xlim, ylim, zlim
    

'''
CARVE: carves away voxels that are not inside the silhouette contained in 
    the view of the camera. The resulting voxel array is returned.

Arguments:
    voxels - an Nx3 matrix where each row is the location of a cubic voxel

    camera - The camera we are using to carve the voxels with. Useful data
        stored in here are the "silhouette" matrix, "image", and the
        projection matrix "P". 

Returns:
    voxels - a subset of the argument passed that are inside the silhouette
'''
def carve(voxels, camera):
    n, _ = voxels.shape
    homogeneous_voxels = np.hstack((voxels, np.ones((n, 1))))
    project_voxels = camera.P.dot(homogeneous_voxels.T).transpose()
    project_voxels /= np.expand_dims(project_voxels[:, 2], axis=1)
    # take x,y coord
    projected_points = np.int32(project_voxels[:, 0:2])
    
    silhouette = camera.silhouette
    silhouette_idx = np.nonzero(silhouette)

    bound_box = (np.min(silhouette_idx[1]), np.max(silhouette_idx[1]), np.min(silhouette_idx[0]), np.max(silhouette_idx[0]))
    x_bound = (projected_points[:, 0] >= bound_box[0]) & (projected_points[:, 0] <= bound_box[1])
    y_bound = (projected_points[:, 1] >= bound_box[2]) & (projected_points[:, 1] <= bound_box[3])
    bound_shape_idx = np.argwhere((x_bound == True) & (y_bound == True)).reshape(-1)
    
    silhouette_idxs = projected_points[bound_shape_idx, :]
    mask_silhouette = silhouette[silhouette_idxs[:, 1], silhouette_idxs[:, 0]]

    carve_idxs = bound_shape_idx[mask_silhouette == True]
    
    return voxels[carve_idxs]


'''
ESTIMATE_SILHOUETTE: Uses a very naive and color-specific heuristic to generate
the silhouette of an object

Arguments:
    im - The image containing a known object. An ndarray of size (H, W, C).

Returns:
    silhouette - An ndarray of size (H, W), where each pixel location is 0 or 1.
        If the (i,j) value is 0, then that pixel location in the original image 
        does not correspond to the object. If the (i,j) value is 1, then that
        that pixel location in the original image does correspond to the object.
'''
def estimate_silhouette(im):
    return np.logical_and(im[:,:,0] > im[:,:,2], im[:,:,0] > im[:,:,1] )


if __name__ == '__main__':
    estimate_better_bounds = True #False
    use_true_silhouette = True
    frames = sio.loadmat('p2/frames.mat')['frames'][0]
    cameras = [Camera(x) for x in frames]

    # Generate the silhouettes based on a color heuristic
    if not use_true_silhouette:
        for i, c in enumerate(cameras):
            c.true_silhouette = c.silhouette
            c.silhouette = estimate_silhouette(c.image)
            if i == 0:
                plt.figure()
                plt.subplot(121)
                plt.imshow(c.true_silhouette, cmap = 'gray')
                plt.title('True Silhouette')
                plt.subplot(122)
                plt.imshow(c.silhouette, cmap = 'gray')
                plt.title('Estimated Silhouette')
                plt.show()

    # Generate the voxel grid
    # You can reduce the number of voxels for faster debugging, but
    # make sure you use the full amount for your final solution
    num_voxels = 6e6
    xlim, ylim, zlim = get_voxel_bounds(cameras, estimate_better_bounds)

    # This part is simply to test forming the initial voxel grid
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, 4000)
    plot_surface(voxels)
    voxels, voxel_size = form_initial_voxels(xlim, ylim, zlim, num_voxels)

    # Test the initial carving
    voxels = carve(voxels, cameras[0])
    if use_true_silhouette:
        plot_surface(voxels)

    # Result after all carvings
    for c in cameras:
        voxels = carve(voxels, c)  
    plot_surface(voxels, voxel_size)
