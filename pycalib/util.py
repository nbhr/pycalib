import numpy as np
import os
import gzip
import bz2
import json

def check_observations(camera_indices, point_indices, points_2d):
    n_observations = points_2d.shape[0]
    n_cameras = camera_indices.max() + 1
    n_points = point_indices.max() + 1

    assert camera_indices.ndim == 1
    assert camera_indices.shape[0] == n_observations
    assert np.all(camera_indices >= 0)
    assert np.all(camera_indices < n_cameras)
    assert len(np.unique(camera_indices)) == n_cameras, "camera_indices must be [0:n_cameras-1]"

    assert point_indices.ndim == 1
    assert point_indices.shape[0] == n_observations
    assert np.all(point_indices >= 0)
    assert np.all(point_indices < n_points)
    assert len(np.unique(point_indices)) == n_points, "point_indices must be [0:n_points-1]"

    assert points_2d.ndim == 2
    assert points_2d.shape[1] == 2

    _, count = np.unique(point_indices, return_counts=True)
    assert np.all(count >= 2)

def check_calib(K, D, R, T):
    if K is not None:
        assert K.ndim == 3,             f'camera_matrix.shape = {camera_matrix.shape} should be (-1, 3, 3)'
        assert K.shape[1:] == (3, 3),   f'camera_matrix.shape = {camera_matrix.shape} should be (-1, 3, 3)'
    if D is not None:
        assert D.ndim == 2,             f'dist_coeffs.shape = {dist_coeffs.shape} should be (-1, 8)'
        assert D.shape[1] == 8,         f'dist_coeffs.shape = {dist_coeffs.shape} should be (-1, 8)'
    if R is not None:
        assert R.ndim == 3,             f'rmat.shape = {rmat.shape} should be (-1, 3, 3)'
        assert R.shape[1:] == (3, 3),   f'rmat.shape = {rmat.shape} should be (-1, 3, 3)'
    if T is not None:
        assert T.ndim == 3,             f'tvec.shape = {tvec.shape} should be (-1, 3, 1)'
        assert T.shape[1:] == (3, 1),   f'tvec.shape = {tvec.shape} should be (-1, 3, 1)'

def load_calib(filename):
    with open_z(filename, 'r') as fp:
        J = json.load(fp)

    reproj = J['reproj'] if 'reproj' in J.keys() else -1
    camera_matrix = np.array(J['K']) if 'K' in J.keys() else None
    dist_coeffs = np.array(J['d']) if 'd' in J.keys() else None
    rmat = np.array(J['R']) if 'R' in J.keys() else None
    tvec = np.array(J['t']) if 't' in J.keys() else None

    check_calib(camera_matrix, dist_coeffs, rmat, tvec)

    return camera_matrix, dist_coeffs, rmat, tvec, reproj

def save_calib(filename, *, camera_matrix=None, dist_coeffs=None, rmat=None, tvec=None, reproj=None):
    check_calib(camera_matrix, dist_coeffs, rmat, tvec)

    J = {}
    if camera_matrix is not None:
        J['K'] = camera_matrix.tolist()
    if dist_coeffs is not None:
        J['d'] = dist_coeffs.tolist()
    if rmat is not None:
        J['R'] = rmat.tolist()
    if tvec is not None:
        J['t'] = tvec.tolist()
    if reproj is not None:
        J['reproj'] = reproj

    with open_z(filename, 'w') as fp:
        json.dump(J, fp, indent=2)

def open_z(filename, mode):
    f, e = os.path.splitext(filename)
    if e.lower() == '.gz':
        return gzip.open(filename, mode)
    elif e.lower() == '.bz2':
        return bz2.open(filename, mode)
    else:
        return open(filename, mode)

def is_shape(a, sz):
    sz0 = a.shape
    if len(sz0) != sz:
        return False
    for i in len(sz0):
        if sz[i] != -1 and sz[i] != sz0[i]:
            return False
    return True

def transpose_to_col(a, m):
    if a.ndim == 1:
        return a.reshape((m, 1))
    
    assert a.ndim == 2, "a is not 2-dim"
    if a.shape[1] == m:
        return a
    else:
        assert a.shape[0] == m, "a is not 2x? or ?x2"
        return a.T
