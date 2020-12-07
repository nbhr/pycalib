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

    assert camera_params.shape[1] == 15

    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    fx = camera_params[:, 6]
    fy = camera_params[:, 7]
    cx = camera_params[:, 8]
    cy = camera_params[:, 9]
    k1 = camera_params[:, 10]
    k2 = camera_params[:, 11]
    p1 = camera_params[:, 12]
    p2 = camera_params[:, 13]
    k3 = camera_params[:, 14]

    n = np.sum(points_proj**2, axis=1)
    n2 = n**2
    kdist = 1 + k1 * n + k2 * n2 + k3 * n**3
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
    camera_params = cam0.reshape((n_cameras, 15))
    camera_params[:, mask] = params[:n_cameras * n_params].reshape((n_cameras, n_params))
    # camera_params = params[:n_cameras * 15].reshape((n_cameras, 15))
    points_3d = params[n_cameras * n_params:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, cam_param_len=15):
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
    assert camera_params.shape[1] == 15

    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    if mask is None:
        mask = np.ones(camera_params.shape[1], dtype=bool)
    assert mask.dtype == bool
    assert len(mask) == camera_params.shape[1]

    camera_params0 = camera_params.copy()

    cam_param_len = np.sum(mask)
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices, cam_param_len)

    x0 = np.hstack((camera_params[:, mask].ravel(), points_3d.ravel()))

    res = least_squares(pycalib.ba.reprojection_error, x0, jac_sparsity=A, verbose=verbose, x_scale='jac', ftol=1e-4, method='trf', args=(n_cameras, n_points, camera_indices, point_indices, points_2d, mask, camera_params0))

    return res


