import numpy as np
import matplotlib.pyplot as plt
import cv2

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

def plotPoly(ax, corners_Nx3, **kwargs):
    """Plot a polygon in 3D

    Args:
        ax (Axis3D): target axis
        corners_Nx3: Nx3 corner points
    """
    corners = np.vstack((corners_Nx3, corners_Nx3[0]))
    ax.plot(corners[:,0], corners[:,1], corners[:,2], **kwargs)

def plotCamera(ax, R, t, *, color=None, scale=1, width=2, height=1.5, focal_length=3, label=None, is_w2c=False, legend=None):
    """Plot a camera in 3D

    Args:
        ax (Axis3D): target axis
        R: 3x3 c2w rotation matrix (interpreted as w2c if is_w2c is set)
        t: 3x1 c2w translation vector (interpreted as w2c if is_w2c is set)
        c: color string
        scale: scaling factor
        width: horizontal size of the camera screen
        height: vertical size of the camera screen
        focal_length: height of the camera cone
        label: text at the camera position
        legend: text shown in the legend
        is_w2c: set True if R, t are w2c
    """

    t = t.reshape((3, 1))

    if is_w2c:
        R = R.T
        t = -R@t

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

    p = ax.plot(L01[:,0], L01[:,1], L01[:,2], "-", color=color, label=legend)
    if color is None:
        color = p[-1].get_color()
    ax.plot(L02[:,0], L02[:,1], L02[:,2], "-", color=color)
    ax.plot(L03[:,0], L03[:,1], L03[:,2], "-", color=color)
    ax.plot(L04[:,0], L04[:,1], L04[:,2], "-", color=color)
    ax.plot(L1234[:,0], L1234[:,1], L1234[:,2], "-", color=color)
    ax.plot(L253[:,0], L253[:,1], L253[:,2], "-", color=color)

    if label is not None:
        ax.text(ps_w[0][0], ps_w[0][1], ps_w[0][2], label)


def plotCameras(camera_params, points_3d, *, scale=-1, title=None, draw_zplane=False, label=None):
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
    ax = plt.axes(projection='3d')
    fig.add_axes(ax)
    ax.set_xlim(p[:,0].min(), p[:,0].max())
    ax.set_ylim(p[:,1].min(), p[:,1].max())
    ax.set_zlim(p[:,2].min(), p[:,2].max())
    #ax.set_ylim(-lim, lim)
    #ax.set_zlim(0, lim)
    ax.plot(points_3d[:,0], points_3d[:,1], points_3d[:,2], "o")
    cmap = plt.get_cmap("tab20")
    if label is None:
        label = [ f'CAM{i}' for i in range(Nc) ]
    for i in range(Nc):
        plotCamera(ax, R[i].T, p[i], color=cmap(i), scale=scale, label=label[i])
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

    axisEqual3D(ax)
    #fig.show()
    return fig

def plotMirror(ax, objpts, n, d, label):
    assert objpts.shape[1] == 3
    t = - objpts @ n.reshape((3,1)) - d
    p = objpts - t.reshape((-1,1)) * n.reshape((1,3))
    ax.plot(p[:,0], p[:,1], p[:,2], label=label)
