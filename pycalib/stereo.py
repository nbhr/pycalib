import cv2
import numpy as np
import matplotlib.pyplot as plt
from .plot import plotCamera, axisEqual3D
from .calib import triangulate_Npts

class StereoPair:
    def __init__(self, K1, d1, R1_w2c, t1_w2c, K2, d2, R2_w2c, t2_w2c, size):
        self.K1 = K1
        self.d1 = d1
        self.R1_w2c = R1_w2c
        self.t1_w2c = t1_w2c.reshape((3,1))
        self.K2 = K2
        self.d2 = d2
        self.R2_w2c = R2_w2c
        self.t2_w2c = t2_w2c.reshape((3,1))
        self.size = size
        self.P1 = self.K1 @ np.hstack([self.R1_w2c, self.t1_w2c])
        self.P2 = self.K2 @ np.hstack([self.R2_w2c, self.t2_w2c])
        self.P = np.array([self.P1, self.P2])
        assert self.P.shape == (2, 3, 4)

        # make cam1 == wcs
        self.R12 = R2_w2c @ R1_w2c.T
        self.t12 = t2_w2c - self.R12 @ t1_w2c

        # R1, R2, P1, P2, Q, validPixROI1, validPixROI2
        ret = cv2.stereoRectify(K1, d1, K2, d2, size, self.R12, self.t12, flags=0, alpha=-1)
        self.rectified_R1 = ret[0]
        self.rectified_R2 = ret[1]
        self.rectified_P1 = ret[2]
        self.rectified_P2 = ret[3]
        self.rectified_K1 = self.rectified_P1[:3,:3]
        self.rectified_K2 = self.rectified_P2[:3,:3]
        self.rectified_t1 = self.rectified_P1[:3,3]
        self.rectified_t2 = self.rectified_P2[:3,3]
        self.rectified_Q = ret[4]
        self.rectified_validPixROI1 = ret[5]
        self.rectified_validPixROI2 = ret[6]

        if np.allclose(self.rectified_P2[1,3], 0):
            self.is_horizontal = True
        else:
            assert np.allclose(self.rectified_P2[0,3], 0)
            self.is_horizontal = False

        # new extrinsic parameters
        self.R1r_w2c = self.rectified_R1@self.R1_w2c
        self.t1r_w2c = self.rectified_R1@self.t1_w2c
        self.R2r_w2c = self.rectified_R2@self.R2_w2c
        self.t2r_w2c = self.rectified_R2@self.t2_w2c
        self.P1r = self.rectified_K1 @ np.hstack([self.R1r_w2c, self.t1r_w2c])
        self.P2r = self.rectified_K2 @ np.hstack([self.R2r_w2c, self.t2r_w2c])
        self.Pr = np.array([self.P1r, self.P2r])
        assert self.Pr.shape == (2, 3, 4)

        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(K1, None, self.rectified_R1, self.rectified_P1, size, cv2.CV_32FC1)
        self.map2_x, self.map2_y = cv2.initUndistortRectifyMap(K2, None, self.rectified_R2, self.rectified_P2, size, cv2.CV_32FC1)

    def rectify_img_1(self, img1):
        return cv2.remap(img1, self.map1_x, self.map1_y, cv2.INTER_NEAREST)

    def rectify_img_2(self, img2):
        return cv2.remap(img2, self.map2_x, self.map2_y, cv2.INTER_NEAREST)

    def rectify_pts3d_1(self, pts3d_Nx3):
        return (self.rectified_R1 @ pts3d_Nx3.T).T

    def rectify_pts3d_2(self, pts3d_Nx3):
        return (self.rectified_R2 @ pts3d_Nx3.T).T

    def unrectify_pts3d_1(self, pts3d_Nx3):
        return (self.rectified_R1.T @ pts3d_Nx3.T).T

    def unrectify_pts3d_2(self, pts3d_Nx3):
        return (self.rectified_R2.T @ pts3d_Nx3.T).T

    def rectify_pts2d_1(self, pts2d_Nx2):
        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.rectified_K1 @ self.rectified_R1 @ np.linalg.inv(self.K1) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def rectify_pts2d_2(self, pts2d_Nx2):
        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.rectified_K2 @ self.rectified_R2 @ np.linalg.inv(self.K2) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def unrectify_pts2d_1(self, pts2d_Nx2):
        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.K1 @ self.rectified_R1.T @ np.linalg.inv(self.rectified_K1) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def unrectify_pts2d_2(self, pts2d_Nx2):
        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.K2 @ self.rectified_R2.T @ np.linalg.inv(self.rectified_K2) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def triangulate_pts(self, p1_rectified_Nx2, p2_rectified_Nx2):
        # given corresponding points in the rectified images, returns the 3D points in WCS
        return triangulate_Npts(np.array([p1_rectified_Nx2, p2_rectified_Nx2]), self.Pr)

    def triangulate_dmap(self, disparity_c1_minus_c2):
        # triangulate in rectified cam1
        X1r = cv2.reprojectImageTo3D(disparity_c1_minus_c2, self.rectified_Q, handleMissingValues=True)
        # reprojectImageTo3D maps points with zero disparity to 10000
        mask = (X1r[:,:,2] != 10000)
        # unrectify
        X1 = self.unrectify_pts3d_1(X1r.reshape((-1,3))).reshape(X1r.shape)
        # to WCS
        X = (self.R1_w2c.T @ X1.reshape((-1,3)).T - (self.R1_w2c.T @ self.t1_w2c)).T.reshape(X1.shape)
        return X, mask

    def plot3d(self, X_MxNx3=[], cam1_name='cam1', cam2_name='cam2', scale=1.0, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        plotCamera(ax, self.R1_w2c,  self.t1_w2c,  color="b", scale=scale, is_w2c=True, legend=f'{cam1_name}')
        plotCamera(ax, self.R1r_w2c, self.t1r_w2c, color="c", scale=scale, is_w2c=True, legend=f'rectified {cam1_name}')
        plotCamera(ax, self.R2_w2c,  self.t2_w2c,  color="r", scale=scale, is_w2c=True, legend=f'{cam2_name}')
        plotCamera(ax, self.R2r_w2c, self.t2r_w2c, color="m", scale=scale, is_w2c=True, legend=f'rectified {cam2_name}')
        for x in X_MxNx3:
            ax.plot(x[:,0], x[:,1], x[:,2], ".", alpha=0.3)
        axisEqual3D(ax)
        ax.set_title(title)
        ax.legend()
        plt.show()