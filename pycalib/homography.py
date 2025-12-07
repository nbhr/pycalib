import numpy as np

def calc_homography_ccs(K1, K2, R12, t12, n1, d1):
    H = K2 @ (R12 - (t12 @ n1.T / d1)) @ np.linalg.inv(K1)
    H = H / H[2,2]
    return H

def calc_homography_wcs(K1, R1_w2c, t1_w2c, K2, R2_w2c, t2_w2c, n, d):
    n = n.reshape((3,1))
    t1_w2c = t1_w2c.reshape((3,1))
    t2_w2c = t2_w2c.reshape((3,1))

    R12 = R2_w2c @ R1_w2c.T
    t12 = t2_w2c - R12 @ t1_w2c
    n1 = (n.T @ R1_w2c.T).T
    d1 = d - n.T @ R1_w2c.T @ t1_w2c

    return calc_homography_ccs(K1, K2, R12, t12, n1, d1)
