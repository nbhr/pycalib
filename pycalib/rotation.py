import numpy as np

# Eq (5) of https://arxiv.org/pdf/1812.07035
def rmat2o6d(rmat):
    if rmat.shape == (3,3):
        return rmat.T.flatten()[:6]
    else:
        # Nx3x3
        assert rmat.shape[1:] == (3,3), rmat.shape
        return rmat[:,:,:2].transpose(0, 2, 1).reshape((-1, 6))

# Eq (6) of https://arxiv.org/pdf/1812.07035
def o6d2rmat(o6d):
    if o6d.shape == (6,):
        c0 = o6d[:3]
        c1 = o6d[3:]
        c0 = c0 / np.linalg.norm(c0)
        c2 = np.cross(c0, c1)
        c2 = c2 / np.linalg.norm(c2)
        c1 = np.cross(c2, c0)
        return np.array((c0,c1,c2)).T
    else:
        assert o6d.shape[1] == 6, o6d.shape
        c0 = o6d[:,:3]
        c1 = o6d[:,3:]
        c0 = c0 / np.linalg.norm(c0, axis=1)[:,None]
        c2 = np.cross(c0, c1, axis=1)
        c2 = c2 / np.linalg.norm(c2, axis=1)[:,None]
        c1 = np.cross(c2, c0, axis=1)
        return np.stack([c0, c1, c2], axis=2)


if __name__ == "__main__":

    x = np.arange(1*3*3).reshape((1,3,3))
    print(x)
    print(rmat2o6d(x[0]))
    x = rmat2o6d(x)
    print(x)
    print(o6d2rmat(x[0]))
    x = o6d2rmat(x)
    print(x)

    rvec6 = np.random.rand(6)
    rvec6 = np.arange(6)
    print(rvec6)
    rmat = o6d2rmat(rvec6)
    print(rmat)
    rvec6 = rmat2o6d(rmat)
    print(rvec6)
    rmat = o6d2rmat(rvec6)
    print(rmat)
    rvec6 = rmat2o6d(rmat)
    print(rvec6)

