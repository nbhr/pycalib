import cv2
import numpy as np
import torch

import pycalib

def bal_recalib(cameras, observations, mask):
    num_cameras = len(cameras)

    K = []
    D = []
    for i in range(num_cameras):
        k = np.eye(3)
        k[0, 0] = k[1, 1] = cameras[i, 6]
        K.append(k)

        d = np.zeros(5)
        d[:2] = cameras[i, 7:9]
        D.append(d)
    K = np.array(K)
    D = np.array(D)

    R, t, X = pycalib.calib.excalibN(K, D, observations)
    assert len(R) == num_cameras
    assert len(t) == num_cameras

    new_cameras = cameras.copy()
    for i in range(num_cameras):
        new_cameras[i, 0:3] = cv2.Rodrigues(R[i])[0].reshape(-1)
        new_cameras[i, 3:6] = t[i].reshape(-1)

    return new_cameras, X

def bal_load_numpy(fp, *, use_initial_pose=True, need_uv_flip=True):
    # http://grail.cs.washington.edu/projects/bal/

    # load all lines
    lines = fp.readlines()

    # num of cameras / points / observations from the 1st line
    num_cameras, num_points, num_observations = [int(x) for x in lines[0].strip().split()]
    curr = 1

    # 2D observations
    observations = np.array([np.loadtxt(lines[i:i+1]) for i in np.arange(curr, curr+num_observations)])
    curr += num_observations
    assert observations.shape == (num_observations, 4)
    assert np.max(observations[:, 1]) == num_points - 1
    assert np.min(observations[:, 1]) == 0
    assert len(np.unique(observations[:, 1].astype(np.int32))) == num_points

    # Cameras
    cameras = np.array([np.loadtxt(lines[i:i+9]) for i in np.arange(curr, curr+num_cameras*9, 9)])
    curr += num_cameras*9
    assert cameras.shape == (num_cameras, 9)

    # 3D points
    points = np.array([np.loadtxt(lines[i:i+3]) for i in np.arange(curr, curr+num_points*3, 3)])
    assert points.shape == (num_points, 3)

    # Sort observations by cameras and then points
    observations = observations[np.lexsort((observations[:, 1], observations[:, 0]))]

    # Mask
    mask = np.zeros((num_cameras, num_points), dtype=np.int)
    mask[observations[:, 0].astype(np.int), observations[:, 1].astype(np.int)] = 1
    assert np.count_nonzero(mask) == num_observations

    # Do we need to flip u,v directions?  (original BAL requires this since its projection model is p = -P / P.z)
    if need_uv_flip:
        observations[:, 2:] = - observations[:, 2:]

    # Do we guess the initial pose by ourselves?
    if use_initial_pose is False:
        cameras, points = bal_recalib(cameras, observations, mask)
        assert points.shape == (num_points, 3)

    return cameras, observations, points, mask

def bal_load(fp, *, use_initial_pose=True, need_uv_flip=True):
    cameras, observations, points, masks = bal_load_numpy(fp, use_initial_pose=use_initial_pose, need_uv_flip=need_uv_flip)
    num_cameras = len(cameras)
    num_points = len(points)

    # build the model
    cams = []
    for i in range(num_cameras):
        cam = pycalib.ba.Camera(cameras[i, 0:3], cameras[i, 3:6], cameras[i, 6], None, 0, 0, cameras[i, 7:9])
        cams.append(cam)

    masks = torch.from_numpy(masks)
    pt2ds = torch.from_numpy(observations[:, 2:])
    assert masks.shape == (num_cameras, num_points)

    model = pycalib.ba.Projection(cams, points.T)

    return model, masks, pt2ds
