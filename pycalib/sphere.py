import numpy as np
import scipy.optimize
import cv2

# memo: cv2.fitEllipse returns ( (cx, cy), (2*a, 2*b), (theta_in_deg) )

def resample_ellipse(ellipse, N):
    t = np.linspace(0, 2*np.pi, N)
    p = np.array([ellipse[1][0] * np.cos(t), ellipse[1][1] * np.sin(t)]) / 2
    theta = ellipse[2] / 180 * np.pi
    rot = np.array([ [np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)] ])
    p = rot @ p + np.array(ellipse[0]).reshape((2, 1))
    return p.T.reshape((-1, 1, 2))


def render_sphere(center_3d, radius, K, img_w, img_h):
    # project a sphere to image (for generating dummy data)

    # [u, v, 1].T
    u, v = np.meshgrid(np.arange(img_w), np.arange(img_h))
    uv1 = np.dstack([u, v, np.ones(u.shape)])

    # (nx, ny, 1) = K^{-1} @ [u, v, 1].T
    nxyz = np.einsum('xy,ijy->ijx', np.linalg.inv(K), uv1)

    # nearest point on the viewing ray from the sphere center
    t = np.einsum('x,ijx->ij', center_3d, nxyz)
    t = t / np.einsum('ijx,ijx->ij', nxyz, nxyz)
    p = nxyz * t[:,:,None]

    # distances from the sphere center to each viewing ray 
    d = np.linalg.norm(center_3d - p, axis=2)
    #print(np.max(d), np.min(d))

    # if the distance is smaller than the radius, it is in the sphere
    img = ((d <= radius) * 255).astype(np.uint8)

    return img


def calc_sphere_center_from_ellipse(cnt, K, r=None):
    # Maalek and Lichti, "Correcting the Eccentricity Error of Projected Spherical Objectsin Perspective Cameras," 2021
    ellipse = cv2.fitEllipse(cnt)
    cx, cy = ellipse[0]
    a, b = np.array(ellipse[1]) / 2
    theta = ellipse[2] / 180 * np.pi

    if a < b:
        a, b = b, a
        theta += np.pi / 2

    if cx == K[0,2] and cy == K[1,2]:
        return np.array([cx, cy])

    dir = np.array([cx-K[0,2], cy-K[1,2]])
    dir = dir / np.linalg.norm(dir)

    fe = np.sqrt(a**2 - b**2)
    ee = fe / np.sqrt(1 + (K[0,0]/b)**2)

    return np.array([cx, cy]) - ee * dir


def fit_sphere_center_3d_to_ellipse(cnt, K, *, r=1, resample=False):
    Kinv = np.linalg.inv(K)

    # x0 is an initial guess of the sphere center
    ellipse = cv2.fitEllipse(cnt)
    x0 = Kinv @ np.array([ellipse[0][0], ellipse[0][1], 1]).reshape((3,1))
    x0 *= K[0,0] / ((ellipse[1][0] + ellipse[1][1]) / 2) * r

    # resample?
    if resample:
        cnt = resample_ellipse(ellipse, 100)

    uv1 = np.dstack([ cnt[:,0,0], cnt[:,0,1], np.ones(cnt.shape[0])]).reshape((-1, 3))
    nxyz = np.einsum('xy,iy->ix', Kinv, uv1)
    nxyz = nxyz / np.linalg.norm(nxyz, axis=1)[:,None]
    r2 = r*r

    def f(x):
        # p is the closest point on the ray along nxyz to the sphere center x
        t = np.einsum('x,ix->i', x, nxyz)
        p = nxyz * t[:,None] - x
        # distance from x to p should be equal to the radius
        d = np.einsum('ij,ij->i', p, p) - r2
        return d

    def jac(x):
        t = np.einsum('x,ix->i', x, nxyz)
        p = nxyz * t[:,None]
        d = p - x

        a = np.einsum('ij,ik->ijk', nxyz, nxyz)
        a -= np.broadcast_to(np.eye(3), a.shape)
        a *= 2

        return (a @ d[:,:,None]).reshape(d.shape)

    ret = scipy.optimize.least_squares(f, x0.flatten(), jac, bounds=([-np.inf, -np.inf, 0], np.inf))
    return ret['x']


