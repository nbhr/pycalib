import unittest
import os
import numpy as np
import pycalib
import time
import cv2

class TestPyCalibSphere(unittest.TestCase):

    def test_sphere(self):
        # dummy data
        W = 640
        H = 480
        fx = 320
        fy = fx
        u0 = W // 2
        v0 = H // 2
        K = np.array( [[ fx, 0, u0 ], [0, fy, v0], [0, 0, 1]])
        RADIUS = 0.5

        X_gt_all = []
        c_gt_all = []
        img_all = []
        cont_all = []
        for X in range(-8, 9, 2):
            for Y in range(-5, 6, 2):
                X_gt = np.array([X, Y, 10])
                c_gt = K @ X_gt.reshape((3,1))
                c_gt = (c_gt / c_gt[2]).flatten()
        
                img = pycalib.render_sphere(X_gt, RADIUS, K, W, H)
                contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
                X_gt_all.append(X_gt)
                c_gt_all.append(c_gt)
                img_all.append(img)
                cont_all.append(contours[0])
        
        for X_gt, c_gt, img, cnt in zip(X_gt_all, c_gt_all, img_all, cont_all):
            ellipse = cv2.fitEllipse(cnt)
            c = pycalib.calc_sphere_center_from_ellipse(ellipse, K)
            self.assertTrue(np.linalg.norm(c[:2] - c_gt[:2]) < 1.0)

            c3d = pycalib.fit_sphere_center_3d_to_ellipse(cnt, K)
            c = K @ c3d.T
            c = c / c[2]
            self.assertTrue(np.linalg.norm(c[:2] - c_gt[:2]) < 1.0)

            c3d = pycalib.fit_sphere_center_3d_to_ellipse(cnt, K, resample=True)
            c = K @ c3d.T
            c = c / c[2]
            self.assertTrue(np.linalg.norm(c[:2] - c_gt[:2]) < 1.0)

