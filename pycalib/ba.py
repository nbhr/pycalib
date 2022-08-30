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
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]
    n_observations = points_2d.shape[0]

    assert camera_params.shape[1] == 17, camera_params.shape[1]
    assert points_3d.shape[1] == 3
    assert points_2d.shape[1] == 2
    assert camera_indices.shape[0] == n_observations
    assert point_indices.shape[0] == n_observations

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
    reproj = res.cost / n_observations

    return camera_params, points_3d, reproj, res

def encode_camera_param(rmat, tvec, K, distCoeffs):
    x = np.zeros(17)
    x[:3] = cv2.Rodrigues(rmat)[0].reshape(-1)
    x[3:6] = tvec.reshape(-1)
    x[6] = K[0,0]
    x[7] = K[0,2]
    x[8] = K[1,2]
    x[9:9+len(distCoeffs)] = distCoeffs
    return x
    
def decode_camera_param(x):
    rmat = cv2.Rodrigues(x[:3])[0]
    tvec = x[3:6].reshape((3, 1))
    K = np.eye(3)
    K[0,0] = x[6]
    K[1,1] = x[6]
    K[0,2] = x[7]
    K[1,2] = x[8]
    distCoeffs = x[9:17]
    return rmat, tvec, K, distCoeffs
    
def make_mask(refine_r, refine_t, refine_f=False, refine_u0=False, refine_v0=False, refine_k1=False, refine_k2=False, refine_p1=False, refine_p2=False, refine_k3=False, refine_k4=False, refine_k5=False, refine_k6=False):
    r = np.tile(refine_r, 3)
    t = np.tile(refine_t, 3)
    return np.concatenate([r, t, [refine_f, refine_u0, refine_v0, refine_k1, refine_k2, refine_p1, refine_p2, refine_k3, refine_k4, refine_k5, refine_k6]]).astype(bool)

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

