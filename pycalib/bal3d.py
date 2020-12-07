"""
import cv2
import numpy as np
import torch

import pycalib

def bal3d_load_numpy(fp, *, need_uv_flip=True):
    # load all lines
    lines = fp.readlines()

    # num of cameras / points / observations from the 1st line
    num_cameras, num_model_points, num_poses, num_observations = [int(x) for x in lines[0].strip().split()]
    curr = 1

    # 2D observations (cam, pose, point, x, y)
    observations = np.array([np.loadtxt(lines[i:i+1]) for i in np.arange(curr, curr+num_observations)])
    curr += num_observations
    assert observations.shape == (num_observations, 5)
    #assert np.min(observations[:, 0]) == 0
    #assert np.max(observations[:, 0]) == num_cameras -1
    assert np.min(observations[:, 1]) == 0
    assert np.max(observations[:, 1]) == num_poses -1
    assert np.min(observations[:, 2]) == 0
    assert len(np.unique(observations[:, 1].astype(np.int32))) == num_poses

    # Model points
    model_points = np.array([np.loadtxt(lines[i:i+3]) for i in np.arange(curr, curr+num_model_points*3, 3)])
    assert model_points.shape == (num_model_points, 3)
    curr += num_model_points * 3

    # Model poses
    model_poses = np.array([np.loadtxt(lines[i:i+6]) for i in np.arange(curr, curr+num_poses*6, 6)])
    curr += num_poses * 6
    assert model_poses.shape == (num_poses, 6)

    # Mask
    mask = np.zeros((num_cameras, num_poses, num_model_points), dtype=np.int)
    mask[observations[:, 0].astype(np.int), observations[:, 1].astype(np.int), observations[:, 2].astype(np.int)] = 1
    assert np.count_nonzero(mask) == num_observations

    # Do we need to flip u,v directions?  (original BAL requires this since its projection model is p = -P / P.z)
    if need_uv_flip:
        observations[:, 3:] = - observations[:, 3:]

    return model_points, model_poses, observations, mask
"""