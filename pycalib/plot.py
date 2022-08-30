import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

def axisEqual3D(ax):
    """
    https://stackoverflow.com/a/19248731
    """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

def plotCamera(ax, R, t, c, scale):
    """Plot a camera in 3D

    Args:
        ax (Axis3D): target axis
        R: 3x3 c2w rotation matrix
        t: 3x1 c2w translation vector
        c: color string
        scale: scaling factor
    """
    if t.shape[0] != 1:
        t = t.T

    ps_c = np.array(([0,0,0], [1,1,3], [1,-1,3], [-1,-1,3], [-1,1,3]))
    ps_w = (scale * R @ ps_c.T + t.T).T

    L01 = np.array([ps_w[0], ps_w[1]])
    L02 = np.array([ps_w[0], ps_w[2]])
    L03 = np.array([ps_w[0], ps_w[3]])
    L04 = np.array([ps_w[0], ps_w[4]])
    L1234 = np.array([ps_w[1], ps_w[2], ps_w[3], ps_w[4], ps_w[1]])
    ax.plot(L01[:,0], L01[:,1], L01[:,2], "-", color=c)
    ax.plot(L02[:,0], L02[:,1], L02[:,2], "-", color=c)
    ax.plot(L03[:,0], L03[:,1], L03[:,2], "-", color=c)
    ax.plot(L04[:,0], L04[:,1], L04[:,2], "-", color=c)
    ax.plot(L1234[:,0], L1234[:,1], L1234[:,2], "-", color=c)

    #axisEqual3D(ax)

def plotCameras(camera_params, points_3d):
    """Plot cameras and points in 3D

    Args:
        camera_params: Nx17 camera parameters (rvec, tvec, f, ...)
        points_3d: Mx3 3D points
    """
    assert points_3d.shape[1] == 3
    Nc = len(camera_params)

    R = []
    t = []
    for c in camera_params:
        R.append(cv2.Rodrigues(c[:3])[0])
        t.append(c[3:6])

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)
    ax.plot(points_3d[:,0], points_3d[:,1], points_3d[:,2], "o")
    cmap = plt.get_cmap("tab10")
    for i in range(Nc):
        plotCamera(ax, R[i].T, - R[i].T @ t[i][:,None], cmap(i), 0.05)
    #plt.savefig('a.png')
    fig.show()
    return fig

