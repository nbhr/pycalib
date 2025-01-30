import numpy as np
import itertools
import cv2
import pycalib

def solve_circle_3d(K, ellipse, *, radius=1):
    """
    Returns the 2 solutions of the 3D circle center and normal vector in the camera coordinate system from its projection, i.e., the ellipse, by Chen et al. 2004.

    Parameters
    ----------
    K : ndarray
        Camera matrix
    ellipse : tuple
        (center, axes, angle) of the ellipse, given by cv2.fitEllipse()

    Returns
    -------
    c : ndarray, 2x3
        Center of the circle in the camera coordinate system
    n : ndarray, 2x3
        Normal vector of the plane of the circle in the camera coordinate system
    """

    assert K.shape == (3, 3), 'K must be a 3x3 camera matrix'
    assert len(ellipse) == 3, 'ellipse must be a tuple (center, axes, angle), given by cv2.fitEllipse()'
    assert radius > 0, 'radius must be positive'

    A, B, C, D, E, F = ellipse_to_quadric(ellipse, K[0, 2], K[1, 2])
    c, n = solve_circle_from_quadric(A, B, C, D, E, F, K[0, 0], radius=radius)

    return c, n

def solve_circle_from_quadric(A, B, C, D, E, F, focal_length, *, radius=1):
    """
    Returns the 2 solutions of the 3D circle center and normal vector in the camera coordinate system from the quadric equation of the ellipse, by Chen et al. 2004.

    Parameters
    ----------
    A : float
        Coefficient of x^2
    B : float
        Coefficient of xy
    C : float
        Coefficient of y^2
    D : float
        Coefficient of x
    E : float
        Coefficient of y
    F : float
        Constant term
    focal_length : float
        Focal length of the camera
    radius : float
        Radius of the circle

    Returns
    -------
    c : ndarray, 2x3
        Center of the circle in the camera coordinate system
    n : ndarray, 2x3
        Normal vector of the plane of the circle in the camera coordinate system
    """
    # Eq (5)
    Q = np.array([[A, B, D/focal_length], [B, C, E/focal_length], [D/focal_length, E/focal_length, F/focal_length/focal_length]])
    assert np.allclose(Q, Q.T)

    # Eq (4), for debugging
    #p = np.zeros((p_gt.shape[0], 3))
    #p[:,:2] = p_gt - K_gt[:2, 2]
    #p[:, 2] = focal_length
    #e = np.einsum('ij,ji->i', p, Q @ p.T)
    #assert np.allclose(e, 0, atol=1e-4)

    # Eq (11)
    eigvals, eigvecs = np.linalg.eigh(Q)
    assert np.allclose(Q, eigvecs @ np.diag(eigvals) @ eigvecs.T)
    #print(f'Eigenvalues: {eigvals}')
    #print(f'Eigenvectors:\n{eigvecs}')

    # Eq (16)
    for i0, i1, i2 in itertools.permutations([0, 1, 2]):
        l1, l2, l3 = eigvals[i0], eigvals[i1], eigvals[i2]
        if l1 * l2 > 0 and l1 * l3 < 0 and np.abs(l1) >= np.abs(l2):
            eigvals = np.array([l1, l2, l3])
            eigvecs = eigvecs[:, [i0, i1, i2]]
            break
    else:
        assert False
    assert np.allclose(Q, eigvecs @ np.diag(eigvals) @ eigvecs.T)

    # Eq (18)
    l1, l2, l3 = eigvals
    g = np.sqrt((l2-l3)/(l1-l3))
    h = np.sqrt((l1-l2)/(l1-l3))

    # Eqs (19), (20)
    c_candidates = []
    n_candidates = []

    for S1 in [-1, 1]:
        for S2 in [-1, 1]:
            for S3 in [-1, 1]:
                z0 = S3 * l2 * radius / np.sqrt(-l1*l3)
                x0 = - S2 * np.sqrt((l1-l2)*(l2-l3)) / l2 * z0
                y0 = 0
                c = z0 * eigvecs @ np.array([S2*l3/l2*h, 0, -S1*l1/l2*g]).reshape((3,1))
                n = eigvecs @ np.array([S2*h, 0, -S1*g]).reshape((3,1))
                R = np.array([[g, 0, S2*h], [0, -S1, 0], [S1*S2*h, 0, -S1*g]])
                if n[2] < 0 and c[2] > 0:
                    c_candidates.append(c)
                    n_candidates.append(n)

    assert len(c_candidates) == 2
    assert len(n_candidates) == 2

    return c_candidates, n_candidates

def draw_ellipse_quadric(img, u0, v0, A, B, C, D, E, F, color=(0, 255, 0), thickness=-1):
    """
    Draws an ellipse on the image using the quadric equation

    Parameters
    ----------
    img : ndarray
        Image
    u0 : float
        x-coordinate of the image center (principal point)
    v0 : float
        y-coordinate of the image center (principal point)
    A : float
        Coefficient of x^2
    B : float
        Coefficient of xy
    C : float
        Coefficient of y^2
    D : float
        Coefficient of x
    E : float
        Coefficient of y
    F : float
        Constant term
    color : tuple
        Color of the ellipse
    thickness : int
        Thickness of the ellipse (-1: filled)
    """

    w = img.shape[1]
    h = img.shape[0]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    x = xx.flatten() - u0
    y = yy.flatten() - v0
    e = A*x**2 + 2*B*x*y + C*y**2 + 2*D*x + 2*E*y + F
    e = e.reshape((h, w))
    if thickness<0:
        img[e < 0] = color
    else:
        img[np.logical_and(e<=thickness, e>=thickness)] = color
    return img

def ellipse_to_quadric(ellipse, u0, v0):
    """
    Returns A, B, C, D, E, F of the quadric equation of the ellipse

    Parameters
    ----------
    ellipse : tuple
        (center, axes, angle) of the ellipse, given by cv2.fitEllipse()
    u0 : float
        x-coordinate of the image center (principal point)
    v0 : float
        y-coordinate of the image center (principal point)

    Returns
    -------
    A : float
        Coefficient of x^2
    B : float
        Coefficient of xy
    C : float
        Coefficient of y^2
    D : float
        Coefficient of x
    E : float
        Coefficient of y
    F : float
        Constant term
    """

    u = ellipse[0][0] - u0
    v = ellipse[0][1] - v0
    a2 = (ellipse[1][1]/2)**2
    b2 = (ellipse[1][0]/2)**2
    theta = ellipse[2] / 180 * np.pi + np.pi/2
    ct = np.cos(theta)
    st = np.sin(theta)
    ct2 = ct*ct
    st2 = st*st

    A = ct2/a2 + st2/b2
    B = ct * st * (1/a2 - 1/b2)
    C = st2/a2 + ct2/b2
    D = - A*u - B*v
    E = - B*u - C*v
    F = A*u*u + 2*B*u*v + C*v*v - 1

    return A, B, C, D, E, F
