import unittest
import numpy as np
import cv2
import pycalib

from pycalib.calib import absolute_orientation

class TestPyCalibAo(unittest.TestCase):
    def test_ao(self):
        # synthetic data
        N = 10
        P = np.random.random((3, N))
        R, _ = cv2.Rodrigues(np.random.random(3))
        t = np.random.random(3)
        s = np.random.random(1)
        Q = s * (R @ P) + t[:, None]

        # solve
        R2, t2, s2 = absolute_orientation(P, Q)
        P2 = s2 * (R2 @ P) + t2[:, None]

        # check
        self.assertTrue(np.allclose(R, R2))
        self.assertTrue(np.allclose(t, t2))
        self.assertTrue(np.allclose(s, s2))

    def test_ao2(self):
        # synthetic data
        N = 10
        P = np.random.random((3, N))
        R, _ = cv2.Rodrigues(np.random.random(3))
        t = np.random.random(3)
        s = 1
        Q = s * (R @ P) + t[:, None]

        # solve
        R2, t2, s2 = absolute_orientation(P, Q, no_scaling=True)
        P2 = s2 * (R2 @ P) + t2[:, None]

        # check
        self.assertTrue(np.allclose(R, R2))
        self.assertTrue(np.allclose(t, t2))
        self.assertTrue(np.allclose(s, s2))

