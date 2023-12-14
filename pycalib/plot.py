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
        getattr(ax, 'set_{}label'.format(dim))(dim)

def plotCamera(ax, R, t, *, color=None, scale=1, width=2, height=1.5, focal_length=3):
    """Plot a camera in 3D

    Args:
        ax (Axis3D): target axis
        R: 3x3 c2w rotation matrix
        t: 3x1 c2w translation vector
        c: color string
        scale: scaling factor
        width: horizontal size of the camera screen
        height: vertical size of the camera screen
        focal_length: height of the camera cone
    """
    t = t.reshape((3, 1))

    w = width
    h = height
    f = focal_length

    # focus, br, tr, tl, bl, triangle
    ps_c = np.array(([0,0,0], [w/2,h/2,f], [w/2,-h/2,f], [-w/2,-h/2,f], [-w/2,h/2,f], [0,-h/2-h/4,f]))
    ps_w = (scale * R @ ps_c.T + t).T

    L01 = np.array([ps_w[0], ps_w[1]])
    L02 = np.array([ps_w[0], ps_w[2]])
    L03 = np.array([ps_w[0], ps_w[3]])
    L04 = np.array([ps_w[0], ps_w[4]])
    L1234 = np.array([ps_w[1], ps_w[2], ps_w[3], ps_w[4], ps_w[1]])
    L253 = np.array([ps_w[2], ps_w[5], ps_w[3]])

    p = ax.plot(L01[:,0], L01[:,1], L01[:,2], "-", color=color)
    if color is None:
        color = p[-1].get_color()
    ax.plot(L02[:,0], L02[:,1], L02[:,2], "-", color=color)
    ax.plot(L03[:,0], L03[:,1], L03[:,2], "-", color=color)
    ax.plot(L04[:,0], L04[:,1], L04[:,2], "-", color=color)
    ax.plot(L1234[:,0], L1234[:,1], L1234[:,2], "-", color=color)
    ax.plot(L253[:,0], L253[:,1], L253[:,2], "-", color=color)


def plotCameras(camera_params, points_3d, *, scale=-1, title=None, draw_zplane=False):
    """Plot cameras and points in 3D

    Args:
        camera_params: Nx17 camera parameters (rvec, tvec, f, ...)
        points_3d: Mx3 3D points
    """
    assert points_3d.shape[1] == 3
    Nc = len(camera_params)

    R = []
    t = []
    p = []
    for c in camera_params:
        R.append(cv2.Rodrigues(c[:3])[0])
        t.append(c[3:6])
        p.append(- R[-1].T @ t[-1][:,None])
    p = np.array(p)

    if scale <= 0:
        scale = 0.05
        if Nc != 1:
            l = np.linalg.norm(t[0] - t[1])
            scale *= l

    fig = plt.figure()
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.set_xlim(p[:,0].min(), p[:,0].max())
    ax.set_ylim(p[:,1].min(), p[:,1].max())
    ax.set_zlim(p[:,2].min(), p[:,2].max())
    #ax.set_ylim(-lim, lim)
    #ax.set_zlim(0, lim)
    ax.plot(points_3d[:,0], points_3d[:,1], points_3d[:,2], "o")
    cmap = plt.get_cmap("tab10")
    for i in range(Nc):
        plotCamera(ax, R[i].T, p[i], color=cmap(i), scale=scale)
        #plotCamera(ax, R[i].T, - R[i].T @ t[i][:,None], cmap(i), scale)
    #plt.savefig('a.png')

    if title is not None:
        ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
    if draw_zplane:
        s = np.linalg.norm(t[0] - t[1])
        ax.plot_surface(*np.meshgrid([-s, s], [-s, s]), np.zeros((2,2)), alpha=0.2)
        ax.plot([0, s], [0, 0], [0, 0])
        ax.plot([0, 0], [0, s], [0, 0])
        ax.text(0, 0, 0, 'O')
        ax.text(s, 0, 0, 'X')
        ax.text(0, s, 0, 'Y')

    fig.show()
    return fig

