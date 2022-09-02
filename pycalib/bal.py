import sys
import cv2
import numpy as np
import pycalib

def bal_recalib(camera_params, camera_indices, point_indices, points_2d):
    num_cameras = len(camera_params)
    assert camera_params.ndim == 2
    pycalib.util.check_observations(camera_indices, point_indices, points_2d)


    if camera_params.shape[1] ==9:
        # BAL -> full camera parameters
        c = []
        for i in camera_params:
            c.append( bal_cam9_to_cam17(i) )
        camera_params = np.array(c)
    assert camera_params.shape[1] == 17


    K = []
    D = []
    for c in camera_params:
        _, _, k, d= pycalib.ba.decode_camera_param(c)
        K.append(k)
        D.append(d)
    K = np.array(K)
    D = np.array(D)


    R, t, X, _ = pycalib.calib.excalibN(K, D, camera_indices, point_indices, points_2d)
    assert len(R) == num_cameras
    assert len(t) == num_cameras

    new_cameras = camera_params.copy()
    for i in range(num_cameras):
        new_cameras[i, 0:3] = cv2.Rodrigues(R[i])[0].reshape(-1)
        new_cameras[i, 3:6] = t[i].reshape(-1)

    return new_cameras, X


# https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
def bal_read(file, *, verify_correspondences=True, verify_indices=True, n_camparams=9):
    """
    read BAL data as-is without applying any conversions
    """

    n_cameras, n_points, n_observations = map(
        int, file.readline().split())

    camera_indices = np.empty(n_observations, dtype=int)
    point_indices = np.empty(n_observations, dtype=int)
    points_2d = np.empty((n_observations, 2))

    for i in range(n_observations):
        camera_index, point_index, x, y = file.readline().split()
        camera_indices[i] = int(float(camera_index))
        point_indices[i] = int(float(point_index))
        points_2d[i] = [float(x), float(y)]

    camera_params = np.empty(n_cameras * n_camparams)
    for i in range(n_cameras * n_camparams):
        camera_params[i] = float(file.readline())
    camera_params = camera_params.reshape((n_cameras, -1))

    points_3d = np.empty(n_points * 3)
    for i in range(n_points * 3):
        points_3d[i] = float(file.readline())
    points_3d = points_3d.reshape((n_points, -1))

    if verify_indices:
        assert np.all(camera_indices >= 0)
        assert np.all(camera_indices < n_cameras)
        assert np.all(point_indices >= 0)
        assert np.all(point_indices < n_points)

    if verify_correspondences:
        pid, count = np.unique(point_indices, return_counts=True)
        assert count.min() >= 2

    camera_indices = camera_indices.astype(np.int32)
    point_indices = point_indices.astype(np.int32)

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def bal_flip_uv(points_2d):
    return - points_2d

# def bal_load_numpy(fp, *, use_initial_pose=True, need_uv_flip=True):
#     """
#     read BAL data with applying conversions
#     """
# 
#     # http://grail.cs.washington.edu/projects/bal/
# 
#     camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(fp)
# 
#     # Do we need to flip u,v directions?  (original BAL requires this since its projection model is p = -P / P.z)
#     if need_uv_flip:
#         #print('flipping UV', file=sys.stderr)
#         points_2d = - points_2d
# 
#     # Do we guess the initial pose by ourselves?
#     if use_initial_pose is False:
#         camera_params, points_3d = bal_recalib(camera_params, camera_indices, point_indices, points_2d)
# 
#     return camera_params, points_3d, camera_indices, point_indices, points_2d

def bal_cam9_to_cam17(camera_params):
    """ converts cameras with 9 params (r, t, f, k1, k2) to 17 params (r, t, f, cx, cv, k1, k2, p1, p2, k3, k4, k5, k6) """
    n = camera_params.shape[0]
    c, m = bal_cam9_to_cam14(camera_params)
    c = np.hstack( [c, np.zeros((n, 3))] )
    m = np.concatenate( [m, [False, False, False]] )
    return c, m

def bal_cam9_to_cam14(camera_params):
    """ converts cameras with 9 params (r, t, f, k1, k2) to 14 params (r, t, f, cx, cv, k1, k2, p1, p2, k3) """
    n = camera_params.shape[0]
    assert camera_params.shape[1] == 9

    c = np.zeros((n, 14))
    m = np.ones(14, dtype=bool)

    c[:, :6] = camera_params[:, :6] # r, t
    c[:, 6] = camera_params[:, 6] # f
    m[7:9] = False # cx, cy
    c[:, 9:11] = camera_params[:, 7:9] # k1, k2
    m[11:14] = False # p1, p2, k3

    return c, m

