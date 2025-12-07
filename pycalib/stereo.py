import cv2
import numpy as np
import matplotlib.pyplot as plt
from .plot import plotCamera, axisEqual3D
from .calib import triangulate_Npts

class StereoPair:
    def __init__(self, K1, d1, imgsz1, R1_w2c, t1_w2c, K2, d2, imgsz2, R2_w2c, t2_w2c, scale=2, alpha=-1):
        """
        Args:
            K1: 3x3 intrinsic matrix of the first (left) camera
            d1: distortion parameters (4, 5, or 8 elements) of the first (left) camera
            imgsz1: image size (w,h) of the first (left) camera
            R1_w2c: 3x3 rotation matrix of the first (left) camera
            t1_w2c: 3x1 translation vector of the first (left) camera
            K2: 3x3 intrinsic matrix of the second (right) camera
            d2: distortion parameters (4, 5, or 8 elements) of the second (right) camera
            imgsz2: image size (w,h) of the second (right) camera
            R2_w2c: 3x3 rotation matrix of the second (right) camera
            t2_w2c: 3x1 translation vector of the second (right) camera
            scale: scaling factor to magnify the rectified images
            alpha: alpha given to cv2.stereoRectify()
        """

        self.K1 = K1
        self.d1 = d1
        self.R1_w2c = R1_w2c
        self.t1_w2c = t1_w2c.reshape((3,1))
        self.K2 = K2
        self.d2 = d2
        self.R2_w2c = R2_w2c
        self.t2_w2c = t2_w2c.reshape((3,1))
        self.imgsz1 = imgsz1
        self.imgsz2 = imgsz2
        self.rectified_imgsz = (max(imgsz1[0], imgsz2[0]), max(imgsz1[1], imgsz2[1]))
        self.scale = scale
        self.P1 = self.K1 @ np.hstack([self.R1_w2c, self.t1_w2c])
        self.P2 = self.K2 @ np.hstack([self.R2_w2c, self.t2_w2c])
        self.P = np.array([self.P1, self.P2])
        assert self.P.shape == (2, 3, 4)

        # make cam1 == wcs
        self.R12 = R2_w2c @ R1_w2c.T
        self.t12 = t2_w2c - self.R12 @ t1_w2c

        # R1, R2, P1, P2, Q, validPixROI1, validPixROI2
        ret = cv2.stereoRectify(K1, d1, K2, d2, self.rectified_imgsz, self.R12, self.t12, flags=0, alpha=alpha)
        if scale != 1:
            ret = list(ret)
            ret[2][0:2, 0:3] = scale * ret[2][0:2, 0:3] # P1
            ret[3][0:2, 0:3] = scale * ret[3][0:2, 0:3] # P2
            ret[4][:,3] = scale * ret[4][:,3] # Q
            ret[5] = tuple(scale * x for x in ret[5]) # validPixROI1
            ret[6] = tuple(scale * x for x in ret[6]) # validPixROI2
            self.rectified_imgsz = (self.rectified_imgsz[0]*scale, self.rectified_imgsz[1]*scale)

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

        self.map1_x, self.map1_y = cv2.initUndistortRectifyMap(K1, None, self.rectified_R1, self.rectified_K1, self.rectified_imgsz, cv2.CV_32FC1)
        self.map2_x, self.map2_y = cv2.initUndistortRectifyMap(K2, None, self.rectified_R2, self.rectified_K2, self.rectified_imgsz, cv2.CV_32FC1)

    def rectify_img_1(self, img1, interpolation=cv2.INTER_NEAREST):
        """
        Rectify an image in the unrecified first camera to the recified first camera

        Args:
            img1: np.ndarray
                Image in the unrectified first camera.
                The size must be `self.imgsz1`.
        Returns:
            np.ndarray
                Image in the recified first camera.
                The size is self.rectified_imgsz.
        """
        return cv2.remap(img1, self.map1_x, self.map1_y, interpolation)

    def rectify_img_2(self, img2, interpolation=cv2.INTER_NEAREST):
        """
        Rectify an image in the unrecified second camera to the recified second camera

        Args:
            img1: np.ndarray
                Image in the unrectified second camera.
                The size must be `self.imgsz2`.
        Returns:
            np.ndarray
                Image in the recified second camera.
                The size is self.rectified_imgsz.
        """
        return cv2.remap(img2, self.map2_x, self.map2_y, interpolation)

    def rectify_pts3d_1(self, pts3d_Nx3):
        """
        Rectify 3D points in the unrecified first camera to the recified first camera

        Args:
            pts3d_Nx3: np.ndarray
                3D points in the unrectified first camera.
                The size must be Nx3.
        Returns:
            np.ndarray
                3D points in the recified first camera.
                The size is Nx3.
        """
        return (self.rectified_R1 @ pts3d_Nx3.T).T

    def rectify_pts3d_2(self, pts3d_Nx3):
        """
        Rectify 3D points in the unrecified second camera to the recified second camera

        Args:
            pts3d_Nx3: np.ndarray
                3D points in the unrectified second camera.
                The size must be Nx3.
        Returns:
            np.ndarray
                3D points in the recified second camera.
                The size is Nx3.
        """
        return (self.rectified_R2 @ pts3d_Nx3.T).T

    def unrectify_pts3d_1(self, pts3d_Nx3):
        """
        Unrectify 3D points in the recified first camera to the unrecified first camera

        Args:
            pts3d_Nx3: np.ndarray
                3D points in the rectified first camera.
                The size must be Nx3.
        Returns:
            np.ndarray
                3D points in the unrecified first camera.
                The size is Nx3.
        """
        return (self.rectified_R1.T @ pts3d_Nx3.T).T

    def unrectify_pts3d_2(self, pts3d_Nx3):
        """
        Unrectify 3D points in the recified second camera to the unrecified second camera

        Args:
            pts3d_Nx3: np.ndarray
                3D points in the rectified second camera.
                The size must be Nx3.
        Returns:
            np.ndarray
                3D points in the unrecified second camera.
                The size is Nx3.
        """
        return (self.rectified_R2.T @ pts3d_Nx3.T).T

    def rectify_pts2d_1(self, pts2d_Nx2):
        """
        Rectify 2D points in the unrecified first camera to the recified first camera

        Args:
            pts2d_Nx2: np.ndarray
                2D points in the unrectified first camera.
                The size must be Nx2.
        Returns:
            np.ndarray
                2D points in the recified first camera.
                The size is Nx2.
        """

        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.rectified_K1 @ self.rectified_R1 @ np.linalg.inv(self.K1) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def rectify_pts2d_2(self, pts2d_Nx2):
        """
        Rectify 2D points in the unrecified second camera to the recified second camera

        Args:
            pts2d_Nx2: np.ndarray
                2D points in the unrectified second camera.
                The size must be Nx2.
        Returns:
            np.ndarray
                2D points in the recified second camera.
                The size is Nx2.
        """

        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.rectified_K2 @ self.rectified_R2 @ np.linalg.inv(self.K2) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def unrectify_pts2d_1(self, pts2d_Nx2):
        """
        Unrectify 2D points in the recified first camera to the unrecified first camera

        Args:
            pts2d_Nx2: np.ndarray
                2D points in the rectified first camera.
                The size must be Nx2.
        Returns:
            np.ndarray
                2D points in the unrecified first camera.
                The size is Nx2.
        """

        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.K1 @ self.rectified_R1.T @ np.linalg.inv(self.rectified_K1) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def unrectify_pts2d_2(self, pts2d_Nx2):
        """
        Unrectify 2D points in the recified second camera to the unrecified second camera

        Args:
            pts2d_Nx2: np.ndarray
                2D points in the rectified second camera.
                The size must be Nx2.
        Returns:
            np.ndarray
                2D points in the unrecified second camera.
                The size is Nx2.
        """

        p = cv2.convertPointsToHomogeneous(pts2d_Nx2).reshape((-1,3))
        c = (self.K2 @ self.rectified_R2.T @ np.linalg.inv(self.rectified_K2) @ p.T).T
        c = c / c[:,2,None]
        return c[:,:2]

    def triangulate_pts(self, p1_rectified_Nx2, p2_rectified_Nx2):
        """
        Triangulate 3D points in WCS from corresponding points in the rectified images

        Args:
            p1_rectified_Nx2: np.ndarray
                2D points in the rectified first camera.
                The size must be Nx2.
            p2_rectified_Nx2: np.ndarray
                2D points in the rectified second camera.
                The size must be Nx2.
        Returns:
            np.ndarray
                3D points in WCS. The size is Nx3.
        """
        # given corresponding points in the rectified images, returns the 3D points in WCS
        #return triangulate_Npts(np.array([p1_rectified_Nx2, p2_rectified_Nx2]), self.Pr)
        X = cv2.triangulatePoints(self.P1r, self.P2r, p1_rectified_Nx2.T, p2_rectified_Nx2.T)
        X = X / X[3,:]
        return X[:3,:].T

    def triangulate_dmap(self, disparity_c1_minus_c2):
        """
        Triangulate 3D points in WCS from a disparity map in the rectified first camera

        Args:
            disparity_c1_minus_c2: np.ndarray
                Signed disparity map in the first camera.
                The disparity should be given as c1-c2, for each corresponding pixel <c1, c2>.
                Pixels without disparity should have np.nan.
                The size must be `self.rectified_imgsz`.

        Returns:
            np.ndarray
                3D points in WCS. The size is Nx3.
        """

        # triangulate in rectified cam1
        dmap = disparity_c1_minus_c2.astype(np.float32)
        X1r = cv2.reprojectImageTo3D(dmap, self.rectified_Q, handleMissingValues=True)
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
