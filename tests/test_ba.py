import unittest
import os
import numpy as np
import bz2
import pycalib
import time
from scipy.optimize import least_squares

class TestPyCalibBa(unittest.TestCase):
    BAL_FILENAME = 'problem-49-7776-pre.txt.bz2'
    BAL_URL = 'https://grail.cs.washington.edu/projects/bal/data/ladybug/problem-49-7776-pre.txt.bz2'

    def test_bal(self):
        if not os.path.isfile(self.BAL_FILENAME):
            import urllib.request
            urllib.request.urlretrieve(self.BAL_URL, self.BAL_FILENAME)

        with bz2.open(self.BAL_FILENAME) as fp:
            camera_params, points_3d, camera_indices, point_indices, points_2d = pycalib.bal.bal_read(fp)

        print('flipping UV')
        points_2d = pycalib.bal.bal_flip_uv(points_2d)
        camera_params, mask = pycalib.bal.bal_cam9_to_cam17(camera_params)

        # allow k1, k2, p1, p2, k3
        mask[9:14] = True

        print(mask, len(mask))

        cam_opt, X_opt, e, ret = pycalib.ba.bundle_adjustment(camera_params, points_3d, camera_indices, point_indices, points_2d, mask=mask)

        print(f'cost = {ret.cost}')
        print(f'reprojection error = {e}')
        self.assertLess(ret.cost, 9963)
        self.assertLess(e, 0.7062)

    def test_sqrt(self):
        N = 5
        for i in range(100):
            A = np.random.randn(N, 2, 2)
            #ic(A)
            #ic(np.transpose(A, (0,2,1)))
            #a = A[1]
            #ic(a)
            #ic(a @ a.T)
            A = np.einsum('nij,njk->nik', A, np.transpose(A, (0,2,1)))
            #ic(A)
            self.assertTrue(np.all(np.linalg.det(A)>=0), 'A must be PSD matrices')
            B = pycalib.ba.sqrt_symmetric_2x2_mat(A)
            #ic(B)
            BB = np.einsum('nij,njk->nik', B, B)
            #ic(BB)
            np.testing.assert_allclose(A, BB)

            iB = np.linalg.inv(B)
            self.assertEqual(iB.shape, (N, 2, 2))
            BiB = np.einsum('nij,njk->nik', B, iB)
            np.testing.assert_allclose(BiB, np.tile(np.eye(2), (N, 1, 1)), atol=1e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
