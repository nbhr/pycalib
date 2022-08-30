import cv2
import numpy as np
import pycalib

def bal_recalib(camera_params, points_2d, camera_indices, point_indices):
    num_cameras = len(camera_params)
    print(camera_params.shape)

    K = []
    D = []
    for i in range(num_cameras):
        k = np.eye(3)
        k[0, 0] = k[1, 1] = camera_params[i, 6]
        K.append(k)

        d = np.zeros(5)
        d[:2] = camera_params[i, 7:9]
        D.append(d)
    K = np.array(K)
    D = np.array(D)

    print(K)

    R, t, X, _ = pycalib.calib.excalibN(K, D, np.hstack([camera_indices[:,None], point_indices[:,None], points_2d]))
    assert len(R) == num_cameras
    assert len(t) == num_cameras

    new_cameras = camera_params.copy()
    for i in range(num_cameras):
        new_cameras[i, 0:3] = cv2.Rodrigues(R[i])[0].reshape(-1)
        new_cameras[i, 3:6] = t[i].reshape(-1)

    return new_cameras, X

# https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
def read_bal_data(file, n_camparams=9):
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

    return camera_params, points_3d, camera_indices, point_indices, points_2d

def bal_load_numpy(fp, *, use_initial_pose=True, need_uv_flip=True):
    # http://grail.cs.washington.edu/projects/bal/

    camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(fp)

    # Do we need to flip u,v directions?  (original BAL requires this since its projection model is p = -P / P.z)
    if need_uv_flip:
        points_2d = - points_2d

    # Do we guess the initial pose by ourselves?
    if use_initial_pose is False:
        camera_params, points_3d = bal_recalib(camera_params, points_2d, camera_indices, point_indices)

    return camera_params, points_3d, camera_indices, point_indices, points_2d

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

def bal_cam9_to_cam15(camera_params):
    """ converts cameras with 9 params (r, t, f, k1, k2) to 15 params (r, t, fx, fy, cx, cv, k1, k2, p1, p2, k3) """
    n = camera_params.shape[0]
    assert camera_params.shape[1] == 9

    c = np.zeros((n, 15))
    m = np.ones(15, dtype=bool)

    c[:, :6] = camera_params[:, :6] # r, t
    c[:, 6] = camera_params[:, 6] # fx
    c[:, 7] = camera_params[:, 6] # fy
    m[8:10] = False # cx, cy
    c[:, 10:12] = camera_params[:, 7:9] # k1, k2
    m[12:15] = False # p1, p2, k3

    return c, m

