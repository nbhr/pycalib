# This is based on the Python version of the original implementation
#
#   https://github.com/computer-vision/takahashi2012cvpr
#

import numpy as np
import cv2
import scipy as sp
import itertools

def w2c(Xp: np.ndarray, rvec: np.ndarray, tvec:np.ndarray) -> np.ndarray:
    R = cv2.Rodrigues(rvec)[0]
    x = R @ Xp.T + tvec.reshape((3, 1))
    return x.T

def sub_tmn_orth(Cp_candidates: np.ndarray) -> np.ndarray:
    num_mirrors = len(Cp_candidates)
    assert Cp_candidates.shape[2:] == (3, 3), Cp_candidates.shape
    best_rho = np.inf
    best_cp = None
    for cp in itertools.product(*Cp_candidates):
        rho = 0
        for cp1, cp2 in itertools.combinations(cp, 2):
            Q = cp1 - cp2
            M = Q.T @ Q
            D, _ = np.linalg.eig(M)
            rho += D.min() / np.sum(D)
        if rho < best_rho:
            best_rho = rho
            best_cp = cp
    return best_cp

def make_A(Xp, n):
    num_mirrors = len(n)
    num_points = len(Xp)
    assert n.shape == (num_mirrors, 3), n.shape
    assert Xp.shape == (num_points, 3), Xp.shape

    A = []
    for i in range(num_mirrors):
        A_normal = np.zeros((3,num_mirrors))
        A_normal[:,i] = 2 * n[i]
        for j in range(num_points):
            A_sub = np.hstack([np.eye(3), A_normal, Xp[j, 0]*np.eye(3), Xp[j,1]*np.eye(3)])
            A.append(A_sub)
    A = np.vstack(A)
    return A

def make_B(Cp, n):
    num_mirrors = len(n)
    num_points = Cp.shape[1]
    assert n.shape == (num_mirrors, 3), n.shape
    assert Cp.shape == (num_mirrors, num_points, 3), Cp.shape

    B = []
    for i in range(num_mirrors):
        for j in range(num_points):

            ni = n[i].reshape((3,1))
            cij = Cp[i,j].reshape((3,1))

            b = -2 * (ni.T @ cij) * ni + cij
            B.append(b.flatten())
    B = np.concatenate(B)
    return B

def sub_tnm_rt(Xp: np.ndarray, Cp: np.ndarray):
    num_mirrors = len(Cp)
    # compute the axis vector m for each pair of mirror pose.
    m_all = [[] for _ in range(num_mirrors)]
    for i1 in range(num_mirrors-1):
        for i2 in range(i1+1, num_mirrors):
            Cp_1 = Cp[i1]
            Cp_2 = Cp[i2]
            # compute the Q matrix
            Q = Cp_2 - Cp_1
            # compute the M matrix
            M = Q.T @ Q
            # compute the eigen vector of M
            D, V = np.linalg.eig(M)
            # compute the m vector as a eigen vector for smallest eigen value.
            m = V[:, D.argmin()]

            m_all[i1].append(m)
            m_all[i2].append(m)

    n_all = []
    for m in m_all:
        # compute the S matrix
        S = np.array(m) # M-1 x 3
        # compute the eigen value of S
        D, V = np.linalg.eig(S.T @ S)
        # compute the normal vector as a eigen vector for smallest eigen value.
        n = V[:, D.argmin()]
        # solve the sign ambiguity
        if n[2] > 0:
            n = - n
        n_all.append(n)
    n_all = np.array(n_all)

    A = make_A(Xp, n_all)
    B = make_B(Cp, n_all)
    X = np.linalg.solve(A.T@A, A.T@B)

    T = X[:3]
    d = X[3:3+num_mirrors]
    r1 = X[3+num_mirrors:6+num_mirrors]
    r2 = X[6+num_mirrors:]

    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    r3 = r3 / np.linalg.norm(r3)
    Q = np.array([r1, r2, r3]).T
    U, S, Vh = np.linalg.svd(Q)
    R = U @ Vh

    #print('R', R)
    #print('T', T)
    #print('d', d)
    return R, T, n_all, d

def householder(n, d):
    #  H = [ eye(3) - 2 * n * n', -2 * d * n; zeros(1,3), 1];
    n = n.reshape((3, 1))
    H = np.eye(4)
    H[:3, :3] = np.eye(3) - 2 * (n @ n.T)
    H[:3,  3] = (-2 * d * n).flatten()
    return H

def sub_reproj_single(Xp, q, R, T, n, d, K):
    N = len(Xp)
    assert Xp.shape == (N, 3), Xp.shape
    assert q.shape == (N, 2), q.shape
    assert R.shape == (3, 3), R.shape
    assert K.shape == (3, 3), K.shape

    # 4x4 householder transformation matrix
    H = householder(n, d)

    # reference points computed with estimated parameters.
    temp_Cp_h = np.ones((4, len(Xp)))
    temp_Cp_h[:3, :] = R @ Xp.T + T.reshape((3, 1))
    Cp_h = H @ temp_Cp_h

    # project the reference points to the image plane
    temp_q = K @ Cp_h[:3,:]
    temp_q = temp_q[:2,:] / temp_q[2,:]

    # mask errors by NaN at negative 2D observations (we should not use 0 here, as we cannot distingish if the error is actually zero or not, and also fakes the average error to be smaller as the mask grows.)
    diff = q - temp_q[:2].T
    diff[np.isnan(q[:,0]),:] = np.nan
    return diff

def sub_reproj_vec(Xp, q, R, T, n, d, K):
    num_mirrors = len(n)
    num_points = len(Xp)

    assert R.shape == (3, 3), R.shape
    assert T.shape == (3,) or T.shape == (3, 1), T.shape
    assert n.shape == (num_mirrors, 3) or n.shape == (num_mirrors, 3, 1), n.shape
    assert len(d) == num_mirrors
    assert q.shape == (num_mirrors, num_points, 2), q.shape
    assert K.shape == (3, 3)
    assert Xp.shape == (num_points, 3), Xp.shape

    err = []
    for i in range(num_mirrors):
        rep = sub_reproj_single(Xp, q[i], R, T, n[i], d[i], K)
        err.extend(rep)

    return np.array(err)

def sub_reproj(Xp, q, R, T, n, d, K):
    err = sub_reproj_vec(Xp, q, R, T, n, d, K)
    assert err.shape[1] == 2
    err = np.linalg.norm(err, axis=1)
    assert len(err) == len(Xp) * len(q)
    return np.nanmean(err)   # err may have NaN for masked observations

def tnm(Xp: np.ndarray, q: np.ndarray, K: np.ndarray, *, flags=None) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Finds the camera and the mirror poses from N>=3 input images

    Parameters
    ----------
    Xp : numpy.ndarray
        Nx3 matrix representing 3D coordinates of N reference points in the reference coordinate system.
    q : numpy.ndarray
        MxNx2 matrix representing 2D projections of N reference points obeserved via M mirrors.
        NaN values indicate "masked / invalid" points.
    K : numpy.ndarray
        3x3 intrinsic camera parameter.
    flags : int
        flags for cv2.solvePnP

    Returns
    -------
    R : numpy.ndarray
        3x3 rotation matrix of the camera.
    T : numpy.ndarray
        3x1 translation vector of the camera.
    n : numpy.ndarray
        Mx3 normal vectors of the mirror planes.
    d : numpy.ndarray
        Mx1 distances between the camera and the mirrors.
    rep : float
        The average of the reprojection errors.
    """

    # Num of mirrors
    M = q.shape[0]
    assert M >= 3, M
    
    # Num of reference points
    N = Xp.shape[0]
    assert N >= 3, N
    
    assert Xp.shape == (N, 3), Xp.shape
    assert q.shape == (M, N, 2), q.shape
    assert K.shape == (3, 3), K.shape

    # solve PnP for each mirror to get the mirrored 3D pts
    Cp = []
    if N > 3:
        for i in range(M):
            # Use non-negative observations only
            #mask = q[i,:,0] >= 0
            mask = np.isfinite(q[i,:,0])
            mN = np.sum(mask)
            mXp = Xp[mask,:]
            mqi = q[i,mask,:]
            #print(mask.shape, mXp.shape, mqi.shape)
            flags = flags if flags is not None else cv2.SOLVEPNP_ITERATIVE
            retval, rvec, tvec = cv2.solvePnP(mXp, np.ascontiguousarray(mqi).reshape((mN, 1, 2)), K, None, flags=flags)
            assert retval is True
            Cp.append(w2c(Xp, rvec, tvec))  # FIXME: mask 3D points too
    else:
        #assert False, "P3P mode is not implemented yet"
        Cp_candidates = []
        for i in range(M):
            # P3P fails ...
            flags = flags if flags is not None else cv2.SOLVEPNP_P3P
            retval, rvecs, tvecs = cv2.solveP3P(Xp, np.ascontiguousarray(q[i]).reshape((N, 1, 2)), K, None, flags=flags)
            #assert retval == 0, retval
            tmp = []
            for rv, tv in zip(rvecs, tvecs):
                tmp.append(w2c(Xp, rv, tv))
            Cp_candidates.append(tmp)
        Cp = sub_tmn_orth(np.array(Cp_candidates))
    Cp = np.array(Cp)

    # compute the extrinsic camera parameter with the proposed method
    R, T, n, d = sub_tnm_rt(Xp, Cp)  # FIXME: this part should also consider the observation mask

    # compute the reprojection error per pixel
    rep = sub_reproj(Xp, q, R, T, n, d, K)

    return R, T, n, d, rep

def cart2sph(xyz):
    xyz = xyz.reshape((-1, 3))
    theta = np.arccos(xyz[:,2])
    phi = np.sign(xyz[:,1])*np.arccos(xyz[:,0] / np.linalg.norm(xyz[:,:2], axis=1))
    return theta, phi

def sph2cart(theta, phi):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    return np.array([st*cp, st*sp, ct]).T

def tnm_ba(Xp: np.ndarray, q: np.ndarray, K: np.ndarray, R0: np.ndarray, T0: np.ndarray, n0: np.ndarray, d0: np.ndarray, *, verbose=0) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimizes the camera and the mirror poses non-linearly

    Parameters
    ----------
    Xp : numpy.ndarray
        Nx3 matrix representing 3D coordinates of N reference points in the reference coordinate system.
    q : numpy.ndarray
        MxNx2 matrix representing 2D projections of N reference points obeserved via M mirrors.
        NaN values indicate "masked / invalid" points.
    K : numpy.ndarray
        3x3 intrinsic camera parameter.
    R0 : numpy.ndarray
        Initial estimate of 3x3 rotation matrix of the camera.
    T0 : numpy.ndarray
        Initial estimate of 3x1 translation vector of the camera.
    n0 : numpy.ndarray
        Initial estimate of Mx3 normal vectors of the mirror planes.
    d0 : numpy.ndarray
        Initial estimate of Mx1 distances between the camera and the mirrors.

    Returns
    -------
    R : numpy.ndarray
        Optimized 3x3 rotation matrix of the camera.
    T : numpy.ndarray
        Optimized 3x1 translation vector of the camera.
    n : numpy.ndarray
        Optimized Mx3 normal vectors of the mirror planes.
    d : numpy.ndarray
        Optimized Mx1 distances between the camera and the mirrors.
    rep : float
        The average of reprojection error per point
    """

    num_points = len(Xp)
    num_mirrors = len(n0)
    assert Xp.shape == (num_points, 3), Xp.shape
    assert q.shape == (num_mirrors, num_points, 2), q.shape
    assert R0.shape == (3, 3), R0.shape
    assert T0.shape == (3,) or T0.shape == (3, 1), T0.shape
    assert n0.shape == (num_mirrors, 3) or n0.shape == (num_mirrors, 3, 1), n0.shape
    assert d0.shape == (num_mirrors,), d0.shape

    def encode(R, T, n, d):
        x = []
        x.extend(cv2.Rodrigues(R)[0].flatten().tolist())
        x.extend(T.flatten().tolist())
        theta, phi = cart2sph(n)
        x.extend(theta.tolist())
        x.extend(phi.tolist())
        #x.extend(n.flatten().tolist())
        x.extend(d.flatten().tolist())
        return np.array(x)

    def decode(x):
        R = cv2.Rodrigues(x[:3])[0]
        T = x[3:6]
        m = (len(x)-6) // 3
        assert 3*m+6 == len(x)
        theta = x[6:6+m]
        phi = x[6+m:6+2*m]
        d = x[6+2*m:]
        n = sph2cart(theta, phi)
        return R, T, n, d

    def objfun(x, Xp, q, K):
        R_, T_, n_, d_ = decode(x)
        e = sub_reproj_vec(Xp, q, R_, T_, n_, d_, K).flatten()
        return e[~np.isnan(e)]

    x0 = encode(R0, T0, n0, d0)
    rep0 = sub_reproj(Xp, q, R0, T0, n0, d0, K)

    # double-check
    R_, T_, n_, d_ = decode(x0)
    assert np.allclose(R0, R_)
    assert np.allclose(T0, T_)
    assert np.allclose(n0, n_)
    assert np.allclose(d0, d_)

    ret = sp.optimize.least_squares(objfun, x0, args=(Xp, q, K), verbose=verbose)

    R, T, n, d = decode(ret['x'])
    rep = sub_reproj(Xp, q, R, T, n, d, K)

    return R, T, n, d, rep

"""
def 
# fill missing corners by dummy
all_obj_pts = []
all_img_pts = []
all_frames = pd.DataFrame({'frame': ALL_PTS['frame'].unique()}).set_index('frame')

for idx, p in enumerate(OBJ_PTS_PX):
    observation = ALL_PTS.loc[ALL_PTS['id'] == idx]
    if len(observation) == 0:
        print(f'Skipping corner {idx} with {len(observation)} observations.')
        continue

    # fill missing frames by (-1, -1)
    uv = all_frames.merge(observation, on='frame', how='left').fillna(-1)
    all_obj_pts.append(p)
    all_img_pts.append(uv[['u','v']].to_numpy())
"""