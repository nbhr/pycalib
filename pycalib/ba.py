import cv2
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pycalib.plot
import pycalib.calib

# https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.

    Rodrigues' rotation formula is used.
    see also https://github.com/strasdat/Sophus/blob/master/sophus/so3.hpp#L257
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project(points, camera_params):
    """Convert 3-D points to 2-D by projecting onto images."""

    assert camera_params.shape[1] == 17

    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    fx = camera_params[:, 6]
    fy = camera_params[:, 6]
    cx = camera_params[:, 7]
    cy = camera_params[:, 8]
    k1 = camera_params[:, 9]
    k2 = camera_params[:, 10]
    p1 = camera_params[:, 11]
    p2 = camera_params[:, 12]
    k3 = camera_params[:, 13]
    k4 = camera_params[:, 14]
    k5 = camera_params[:, 15]
    k6 = camera_params[:, 16]

    n = np.sum(points_proj**2, axis=1)
    n2 = n**2
    n3 = n2 * n
    kdist = (1 + k1 * n + k2 * n2 + k3 * n3) / (1 + k4 * n + k5 * n2 + k6 * n3)
    pdist = 2 * points_proj[:,0] * points_proj[:,1]
    u = points_proj[:,0]*kdist + p1*pdist + p2*(n2+2*points_proj[:,0]**2)
    v = points_proj[:,1]*kdist + p2*pdist + p1*(n2+2*points_proj[:,0]**2)

    points_proj[:, 0] = fx*u + cx
    points_proj[:, 1] = fy*v + cy

    return points_proj

def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, mask, cam0):
    """Compute residuals.

    `params` contains camera parameters and 3-D coordinates.
    """
    n_params = np.sum(mask)
    camera_params = cam0.reshape((n_cameras, 17))
    camera_params[:, mask] = params[:n_cameras * n_params].reshape((n_cameras, n_params))
    # camera_params = params[:n_cameras * 17].reshape((n_cameras, 17))
    points_3d = params[n_cameras * n_params:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, cam_param_len=17):
    m = camera_indices.size * 2
    n = n_cameras * cam_param_len + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(cam_param_len):
        A[2 * i, camera_indices * cam_param_len + s] = 1
        A[2 * i + 1, camera_indices * cam_param_len + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * cam_param_len + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * cam_param_len + point_indices * 3 + s] = 1

    return A

def bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d, *, verbose=2, mask=None):
    """
    Optimize camera poses and intrinsics non-linearly

    Parameters
    ----------
    camera_params : ndarray
        C x 17 array of camera parameters (rvec, tvec, f, u0, v0, k1, k2, p1, p2, k3, k4, k5, k6)
    points_3d : ndarray
        M x 3 array of 3D points
    camera_indices : ndarray
        N array of camera indices. points_3d[point_indices[i]] is observed by camera_indices[i] at points_2d[i]
    point_indices : ndarray
        N array of point indices. points_3d[point_indices[i]] is observed by camera_indices[i] at points_2d[i]
    points_2d: ndarray
        N x 2 array of 2D points.  points_3d[point_indices[i]] is observed by camera_indices[i] at points_2d[i]


    Returns
    -------
    camera_params : ndarray
        C x 17 array of camera parameters (rvec, tvec, f, u0, v0, k1, k2, p1, p2, k3, k4, k5, k6)
    points_3d : ndarray
        M x 3 array of 3D points
    reproj : float
        reprojection error (RMSE)
    res : OptimizeResult
        output of `scipy.optimize.least_squares()`

    """

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n_observations = points_2d.shape[0]

    assert camera_params.shape[1] == 17, camera_params.shape[1]
    pycalib.util.check_observations(camera_indices, point_indices, points_2d)

    """
    print(camera_params.shape)
    print(points_3d.shape)
    print(camera_indices.shape)
    print(point_indices.shape)
    print(points_2d.shape)
    """

    if mask is None:
        mask = np.ones(camera_params.shape[1], dtype=bool)
    assert mask.dtype == bool
    assert len(mask) == camera_params.shape[1]

    camera_params0 = camera_params.copy()

    cam_param_len = np.sum(mask)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, cam_param_len)

    x0 = np.hstack((camera_params[:, mask].ravel(), points_3d.ravel()))

    res = least_squares(reprojection_error, x0, jac_sparsity=A, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf', args=(n_cameras, n_points, camera_indices, point_indices, points_2d, mask, camera_params0))

    n = camera_params[:, mask].size
    x = res.x[:n]
    camera_params[:, mask] = x.reshape((n_cameras, -1))
    points_3d = res.x[n:].reshape((-1, 3))

    e = reprojection_error(res.x, n_cameras, n_points, camera_indices, point_indices, points_2d, mask, camera_params0).reshape((-1, 2))
    e = np.linalg.norm(e, axis=1)
    assert( len(e) == n_observations )
    e = np.sqrt(np.mean(e))

    return camera_params, points_3d, e , res

def encode_camera_params(R_Nx3x3, T_Nx3, K_Nx3x3, D_Nx8):
    N = len(R_Nx3x3)
    assert R_Nx3x3.shape == (N, 3, 3), R_Nx3x3.shape
    assert T_Nx3.shape == (N, 3, 1), T_Nx3.shape
    assert K_Nx3x3.shape == (N, 3, 3), K_Nx3x3.shape
    assert D_Nx8.shape == (N, 8), D_Nx8.shape

    p = []
    for r, t, k, d in zip(R_Nx3x3, T_Nx3, K_Nx3x3, D_Nx8):
        p.append( encode_camera_param(r, t, k, d) )
    p = np.array(p)

    assert p.shape == (N, 17)

    return p


def decode_camera_params(params_Nx17):
    N = len(params_Nx17)
    assert params_Nx17.shape == (N, 17)

    R = []
    T = []
    K = []
    D = []

    for p in params_Nx17:
        r, t, k, d = decode_camera_param(p)
        R.append(r)
        T.append(t)
        K.append(k)
        D.append(d)

    R = np.array(R)
    T = np.array(T)
    K = np.array(K)
    D = np.array(D)

    assert R.shape == (N, 3, 3), R.shape
    assert T.shape == (N, 3, 1), T.shape
    assert K.shape == (N, 3, 3), K.shape
    assert D.shape == (N, 8), D.shape

    return R, T, K, D


def encode_camera_param(rmat, tvec, K, distCoeffs):
    """
    Convert camera parameters for optimization (mat -> 17 params)
    """

    x = np.zeros(17)
    x[:3] = cv2.Rodrigues(rmat)[0].reshape(-1)
    x[3:6] = tvec.reshape(-1)
    x[6] = K[0,0]
    x[7] = K[0,2]
    x[8] = K[1,2]
    x[9:9+len(distCoeffs)] = distCoeffs
    return x
    
def decode_camera_param(x):
    """
    Convert optimized camera parameters (17 params -> mat)
    """

    rmat = cv2.Rodrigues(x[:3])[0]
    tvec = x[3:6].reshape((3, 1))
    K = np.eye(3)
    K[0,0] = x[6]
    K[1,1] = x[6]
    K[0,2] = x[7]
    K[1,2] = x[8]
    distCoeffs = x[9:17]
    return rmat, tvec, K, distCoeffs
    
def make_mask(r, t, *, f=False, u0=False, v0=False, k1=False, k2=False, p1=False, p2=False, k3=False, k4=False, k5=False, k6=False):
    """
    Generate a mask to indicate the parameters to be optimized

    Parameters
    ----------
    r : bool
        Optimize the rotation matrix if True
    t : bool
        Optimize the translation vector if True
    f : bool
        Optimize the focal length if True
    u0 : bool
        Optimize u0 if True
    v0 : bool
        Optimize v0 if True
    k1 : bool
        Optimize k1 if True
    k2 : bool
        Optimize k2 if True
    p1 : bool
        Optimize p1 if True
    p2 : bool
        Optimize p2 if True
    k3 : bool
        Optimize k3 if True
    k4 : bool
        Optimize k4 if True
    k5 : bool
        Optimize k5 if True
    k6 : bool
        Optimize k6 if True
    """

    r = np.tile(r, 3)
    t = np.tile(t, 3)
    return np.concatenate([r, t, [f, u0, v0, k1, k2, p1, p2, k3, k4, k5, k6]]).astype(bool)

"""
def excalib2_ba(p1, p2, A1, d1, A2, d2, *, verbose=0):
    R, t, E, status, X = pycalib.calib.excalib2(p1, p2, A1, d1, A2, d2)
    assert np.all(status == 1)

    # BA
    Nc = 2
    Np = X.shape[1]

    ## Camera parameters
    camera_params = np.array([
        encode_camera_param( np.eye(3), np.zeros((3,1)), A1, np.zeros(5) if d1 is None else d1 ),
        encode_camera_param( R,         t,               A2, np.zeros(5) if d2 is None else d2 ),
    ])

    ## camera_indices[i] == the camera observes point_2d[i,:]
    camera_indices = np.repeat(np.arange(Nc), Np)

    ## point_indices[i] == the 3D point behind point_2d[i,:]
    point_indices = np.tile(np.arange(Np), Nc)

    ## Optimization target
    ## R, t, f, u0, v0, k1, k2, p1, p2, k3, k4, k5, k6
    mask = make_mask(True, True, False, False, False, False, False, False, False, False)

    ## 3D est (Np, 3)
    X_est = X.T

    ## 2D est (Nc, Np, 2)
    p1 = pycalib.calib.transpose_to_col(p1, 2).astype(np.float)
    p2 = pycalib.calib.transpose_to_col(p2, 2).astype(np.float)
    x_est = np.array( [ p1, p2 ] )

    ## optim
    cam_opt, X_opt, ret = bundle_adjustment(camera_params, X_est, camera_indices, point_indices, x_est.reshape((-1, 2)), mask=mask, verbose=verbose)

    rmat1, tvec1, k1, d1 = decode_camera_param(cam_opt[0])
    rmat2, tvec2, k2, d2 = decode_camera_param(cam_opt[1])

    assert np.allclose(k1, A1)
    assert np.allclose(k2, A2)
    assert np.count_nonzero(d1) == 0
    assert np.count_nonzero(d2) == 0

    #c1 = R1 x + t1
    #c2 = R2 x + t2
    #x = R1.T @ (c1 - t1)
    #c2 = R2 R1.T (c1 - t1) + t2

    R = rmat2 @ rmat1.T
    t = tvec2 - R @ tvec1
    X = rmat1 @ X_opt.T  + tvec1
    E = pycalib.calib.skew(t) @ R

    return R, t, E, status, X
"""

