import unittest
import numpy as np
import pycalib
import cv2

class TestPyCalibDistortion(unittest.TestCase):

    def test_distortion(self):
        np.random.seed(0)

        # N-cameras
        Nc = 15

        # dummy data
        K_all = []
        D_all = []
        R_all = []
        T_all = []
        P_all = []
        for c in range(Nc):
            W = 640 + np.random.normal(0, 10)
            H = 480 + np.random.normal(0, 10)
            fx = 320 + np.random.normal(0, 10)
            fy = fx
            u0 = W // 2 + np.random.normal(0, 10)
            v0 = H // 2 + np.random.normal(0, 10)
            K = np.array( [[ fx, 0, u0 ], [0, fy, v0], [0, 0, 1]])
            D = np.random.normal(0, 0.1, 8)
            tvec = np.random.normal(0, 1, 3)
            tvec = tvec / np.linalg.norm(tvec) * (3 + np.random.rand())
            R, T = pycalib.calib.lookat(tvec, np.zeros(3), np.array([0, 1, 0]))
            P = K @ np.hstack([R, T])

            K_all.append(K)
            D_all.append(D)
            R_all.append(R)
            T_all.append(T)
            P_all.append(P)

        K_all = np.array(K_all)
        D_all = np.array(D_all)
        R_all = np.array(R_all)
        T_all = np.array(T_all)
        P_all = np.array(P_all)

        # grid points in 3D
        X_3d = np.array(np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))).reshape((3, -1)).T  # 3D grid points
        Np = len(X_3d)

        # projections
        X_2d = []
        for K, D, R, T, P in zip(K_all, D_all, R_all, T_all, P_all):
            x, _ = cv2.projectPoints(X_3d, R, T, K, D)
            X_2d.append(x)
        X_2d = np.array(X_2d)
        self.assertEqual(X_2d.shape, (Nc, Np, 1, 2))

        # undistort points
        U = []
        for x, K, D, R, T, P in zip(X_2d, K_all, D_all, R_all, T_all, P_all):
            u = pycalib.calib.undistort_points(x.reshape((-1, 2)), K, D)
            #u = cv2.undistortPointsIter(x, K, D, np.eye(3), K, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-8))

            # n3d = cv2.convertPointsToHomogeneous(u)
            # n3d = np.linalg.inv(K) @ n3d.reshape((-1, 3)).T
            # y, _ = cv2.projectPoints(n3d.T, np.eye(3), np.zeros(3), K, D)
            # print(y)
            d = pycalib.calib.distort_points(u.reshape((-1,1,2)), K, D)
            np.testing.assert_allclose(x, d, rtol=1e-6, atol=1e-6)
            U.append(u)
        U = np.array(U).reshape((Nc, -1, 2))

        # triangulate at once
        X = pycalib.calib.triangulate_Npts(U, P_all)
        np.testing.assert_allclose(X, X_3d, atol=1e-6)

        # triangulate one by one
        X = []
        for i in range(Np):
            x = pycalib.calib.triangulate(U[:,i,:], P_all)
            X.append(x)
        X = np.array(X)[:,:3]
        np.testing.assert_allclose(X, X_3d, atol=1e-6)