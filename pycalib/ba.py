import torch
from torch.nn import Module, ModuleList, ParameterList
from torch.nn.parameter import Parameter
import torch.optim as optim

import cv2
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pycalib.plot
import pycalib.calib

torch.set_default_tensor_type(torch.DoubleTensor)

def skew(x):
    """
    Return skew-symmetric matrix
    """
    return torch.tensor([[
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ]], device=x.device)

def rvec2mat(rvec):
    """
    Convert Rodrigues vector to rotation matrix.
    This is identical to vector-to-matrix convertion by cv2.Rodrigues().

    Parameters
    ----------
    rvec : 1x3 or 3x1 torch.tensor
        Rodrigues vector

    Returns
    -------
    rmat : 3x3 torch.tensor
        Rotation matrix
    """

    # see also https://github.com/strasdat/Sophus/blob/master/sophus/so3.hpp#L257

    # https://discuss.pytorch.org/t/how-to-normalize-embedding-vectors/1209/2
    device = rvec.device
    rvec = rvec.reshape(3)
    theta = torch.norm(rvec, 2)
    if theta != 0:
        rvec = rvec / theta
    #else:
    #   No need to return here, since the following equation automatically yields
    #   eye(3) for rvec == 0. If returns, it disconnects the autograd graph.
    #   return torch.eye(3)

    c_th = torch.cos(theta)
    s_th = torch.sin(theta)
    return c_th * torch.eye(3).to(device) + (1-c_th) * torch.ger(rvec, rvec) + s_th * skew(rvec)


def distort(pt3d, d):
    """
    Distort points in normalized camera. Identical to cv2.projectPoints(pt3d, np.zeros(3),
    np.zeros(3), np.eye(3), distCoeffs).

    Parameters
    ----------
    pt3d : 3xN torch.tensor
        Ideal positions (not necessarily in normalized camera, i.e., pt3d[3,:] is
        not required to be 1 since this function first does x/=z and y/=z anyway)
    d : list of torch.tensor
        Distortion coeffs (k1, k2, p1, p2, k3)

    Returns
    -------
    n : 2xN torch.tensor
        Distorted points in normalized camera
    """
    pt3d = pt3d.reshape((3, -1))
    x1 = pt3d[0, :] / pt3d[2, :]
    y1 = pt3d[1, :] / pt3d[2, :]
    r2 = (x1*x1 + y1*y1)
    r4 = r2*r2
    r6 = r4*r2
    kdist = 1 + d[0]*r2 + d[1]*r4 + d[4]*r6
    pdist = 2*x1*y1
    x2 = x1*kdist + d[2]*pdist + d[3]*(r2+2*x1*x1)
    y2 = y1*kdist + d[3]*pdist + d[2]*(r2+2*y1*y1)
    return x2, y2


def projectPoints(pt3d, rvec, tvec, fx, fy, cx, cy, distCoeffs):
    """
    Project 3D points in WCS to image. Identical to cv2.projectPoints(pt3d, rvec, tvec,
    cameraMatrix, distCoeffs), where cameraMatrix = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].

    Parameters
    ----------
    pt3d : 3xN torch.tensor
        3D points in world coordinate system
    rvec : 1x3 or 3x1 torch.tensor
        Rodrigues vector representing world-to-camera conversion
    tvec : 1x3 or 3x1 torch.tensor
        Translation vector representing world-to-camera conversion
    fx : torch.tensor
        Focal length (aka alpha)
    fy : torch.tensor
        Focal length (aka beta)
    cx : torch.tensor
        principal point (aka u0)
    cy : torch.tensor
        principal point (aka v0)
    distCoeffs : 1x5 or 5x1 torch.tensor
        Distortion coeffs (k1, k2, p1, p2, k3)

    Returns
    -------
    pt2d : 2xN torch.tensor
        Image points
    """
    R = rvec2mat(rvec)
    u, v = distort(torch.matmul(R, pt3d) + tvec.unsqueeze(1), distCoeffs)
    return torch.stack([fx*u + cx, fy*v + cy])


class Camera(Module):
    """
    Camera

    Attributes
    ----------
    rvec : 1x3 or 3x1 torch.tensor
        Rodrigues vector representing world-to-camera conversion
    tvec : 1x3 or 3x1 torch.tensor
        Translation vector representing world-to-camera conversion
    fx : torch.tensor
        Focal length (aka alpha)
    fy : torch.tensor
        Focal length (aka beta)
    cx : torch.tensor
        principal point (aka u0)
    cy : torch.tensor
        principal point (aka v0)
    distCoeffs : 1x5 or 5x1 torch.tensor
        Distortion coeffs (k1, k2, p1, p2, k3)
    """

    def __init__(self, rvec, tvec, fx, fy, cx, cy, distCoeffs):
        """
        Parameters
        ----------
        rvec : 1x3 or 3x1 numpy.ndarray
            Rodrigues vector representing world-to-camera conversion
        tvec : 1x3 or 3x1 numpy.ndarray
            Translation vector representing world-to-camera conversion
        fx : double
            Focal length (aka alpha)
        fy : double or None
            Focal length (aka beta)
        cx : double
            principal point (aka u0)
        cy : double
            principal point (aka v0)
        distCoeffs : numpy.ndarray of size 1, 2, ..., or 5
            Distortion coeffs (k1, k2, p1, p2, k3). If the size is less than 5, the higher coeffs are fixed to zero.
        """
        super(Camera, self).__init__()

        def get_d(distCoeffs, i):
            if distCoeffs is not None and distCoeffs.size > i:
                return Parameter(torch.tensor(distCoeffs[i]))
            else:
                return Parameter(torch.tensor(0.0), requires_grad=False)

        self.rvec = Parameter(torch.from_numpy(rvec))
        self.tvec = Parameter(torch.from_numpy(tvec.reshape(-1)))
        self.fx = Parameter(torch.tensor(np.double(fx)))
        self.fy = Parameter(torch.tensor(np.double(fy))) if fy is not None else self.fx
        self.cx = Parameter(torch.tensor(np.double(cx))) if cx is not None else Parameter(torch.tensor(0.0, requires_grad=False), requires_grad=False)
        self.cy = Parameter(torch.tensor(np.double(cy))) if cy is not None else Parameter(torch.tensor(0.0, requires_grad=False), requires_grad=False)
        self.k1 = get_d(distCoeffs, 0)
        self.k2 = get_d(distCoeffs, 1)
        self.p1 = get_d(distCoeffs, 2)
        self.p2 = get_d(distCoeffs, 3)
        self.k3 = get_d(distCoeffs, 4)

        self.distCoeffs = [self.k1, self.k2, self.p1, self.p2, self.k3]

    def forward(self, pt3d):
        """
        Project 3D points in WCS to the image plane

        Parameters
        ----------
        pt3d : 3xN torch.tensor
            3D points in world coordinate system

        Returns
        -------
        pt2d : 2xN torch.tensor
            Image points
        """
        return projectPoints(pt3d, self.rvec, self.tvec, self.fx, self.fy, self.cx, self.cy, self.distCoeffs)

    def extra_repr(self):
        return 'rvec={},\ntvec={},\nfx={}, fy={}, cx={}, cy={},\ndist=[{}, {}, {}, {}, {}]'.format(
            self.rvec.data.numpy(), self.tvec.data.numpy(), self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2, self.k3
        )

    def w2c(self):
        return cv2.Rodrigues(self.rvec.data.numpy())[0], self.tvec.data.numpy().reshape((3, 1))

    def c2w(self):
        R, t = self.w2c()
        return R.T, -R.T @ t

    def plot(self, ax, *, label=None, color='g', scale=1):
        R, t = self.c2w()
        pycalib.plot.plotCamera(ax, R, t, color, scale)
        if label is not None:
            ax.text(t[0], t[1], t[2], label, fontsize=32)

class Projection(Module):
    """
    Projection of 3D points to multiple cameras with visibility mask
    """
    def __init__(self, cameras, pt3d):
        """
        Parameters
        ----------
        cameras : list of Camera objects
            Cameras observing the 3D points in WCS
        pt3d : 3xN torch.tensor
            3D points in WCS
        """
        super(Projection, self).__init__()
        self.Nc = len(cameras)
        self.Np = pt3d.shape[1]
        self.cameras = ModuleList(cameras)
        self.pt3d = Parameter(torch.from_numpy(pt3d))
        assert pt3d.ndim == 2
        assert pt3d.shape[0] == 3

    def forward(self, mask):
        """
        Project 3D points specified by visibility mask

        Parameters
        ----------
        mask : Nc x Np torch.tensor or numpy.ndarray
            Binary visibility mask. If (i, j) is non-zero, the j-th 3D point is projected to
            the i-th camera.

        Returns
        -------
        pt2d : Mx2 torch.tensor
            (u,v) positions of M projected points, where M = np.count_nonzero(mask).
            The size is Mx2, i.e., [[u0, v0], [u1, v1], [u2, v2], ...]
        """
        # mask is a binary Nc x Np matrix
        assert mask.dim() == 2
        assert mask.shape == (self.Nc, self.Np)
        rep = []
        for c in range(self.Nc):
            idx = mask[c, :].nonzero()
            X = self.pt3d[:, idx].squeeze()
            x = self.cameras[c](X)
            rep.append(x.transpose(0, 1))
            #rep.append(x)
        return torch.cat(rep)#.view(-1)

    def get_maxvar(self):
        t = [c.c2w()[1] for c in self.cameras]
        t = np.hstack(t)
        t_min = t.min(axis=1)
        t_max = t.max(axis=1)
        return (t_max - t_min).max()

    def plot(self, ax, *, scale=-1):
        if scale < 0:
            s = self.get_maxvar()
            if s > 0:
                scale = 0.05 / s
            else:
                s = 1

        cmap = plt.get_cmap("tab10")
        for i, c in enumerate(self.cameras):
            c.plot(ax, label=f'cam{i}', color=cmap(i), scale=scale)
        p = self.pt3d.data.numpy()
        ax.plot(p[0,:], p[1,:], p[2,:], ".", color='black')

    def plot_fig(self, *, scale=-1):
        fig = plt.figure()
        ax = Axes3D(fig)
        self.plot(ax, scale=scale)
        return fig, ax

def load_bal(fp, *, use_initial_pose=True, need_uv_flip=True):
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
        def camD(i):
            D = np.zeros(5)
            D[:2] = cameras[i, 7:9]
            return D

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
        print(X.shape)
        print(num_points)
        assert len(R) == num_cameras
        assert len(t) == num_cameras
        assert len(X) == num_points
        assert X.shape == points.shape

        for i in range(num_cameras):
            cameras[i, 0:3] = cv2.Rodrigues(R[i])[0].reshape(-1)
            cameras[i, 3:6] = t[i].reshape(-1)
        points = X




    # build the model
    cams = []
    for i in range(num_cameras):
        cam = Camera(cameras[i, 0:3], cameras[i, 3:6], cameras[i, 6], None, 0, 0, cameras[i, 7:9])
        cams.append(cam)

    masks = torch.from_numpy(mask)
    pt2ds = torch.from_numpy(observations[:, 2:])
    assert masks.shape == (num_cameras, num_points)

    model = Projection(cams, points.T)

    return model, masks, pt2ds


