import unittest
import numpy as np
import cv2
import pycalib


def rotation_angle_deg(R1, R2):
    """Angle (deg) between two rotation matrices."""
    R = R1 @ R2.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


class TestBaMask(unittest.TestCase):
    """
    Verifies that the `mask` parameter of pycalib.ba.bundle_adjustment() correctly
    (1) fixes the parameters it is told to fix, and
    (2) ties together ("shares") the parameters it is told to share,
    as demonstrated in ipynb/ncam_ba.ipynb.
    """

    def make_scene(self, n_cameras=5, seed=0):
        """A synthetic n-camera scene sharing one intrinsic matrix, used as GT."""
        X_gt = np.array(np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))).reshape((3, -1)).T
        Np = X_gt.shape[0]

        K_gt = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        d_gt = np.zeros(8)

        # cameras on a circle around the points, looking at the origin
        angles = 2 * np.pi / n_cameras * np.arange(n_cameras)
        eyes = np.vstack((10 * np.cos(angles), 10 * np.sin(angles), np.zeros(n_cameras))).T

        R_gt = []
        t_gt = []
        for i in range(n_cameras):
            R, t = pycalib.calib.lookat(eyes[i], np.zeros(3), np.array([0., 1., 0.]))
            R_gt.append(R)
            t_gt.append(t)
        R_gt = np.array(R_gt)
        t_gt = np.array(t_gt)

        camera_params_gt = pycalib.ba.encode_camera_params(
            R_gt, t_gt, np.tile(K_gt, (n_cameras, 1, 1)), np.tile(d_gt, (n_cameras, 1)))

        x_gt = np.array([
            cv2.projectPoints(X_gt.reshape((-1, 1, 3)), camera_params_gt[i, :3], t_gt[i], K_gt, None)[0].reshape((-1, 2))
            for i in range(n_cameras)
        ])

        camera_indices = np.repeat(np.arange(n_cameras), Np)
        point_indices = np.tile(np.arange(Np), n_cameras)
        points_2d = x_gt.reshape((-1, 2))

        return {
            'rng': np.random.default_rng(seed),
            'Nc': n_cameras,
            'Np': Np,
            'camera_params_gt': camera_params_gt,
            'R_gt': R_gt,
            't_gt': t_gt,
            'K_gt': K_gt,
            'X_gt': X_gt,
            'camera_indices': camera_indices,
            'point_indices': point_indices,
            'points_2d': points_2d,
        }

    def perturb_initial_guess(self, scene, *, rot_sigma=0.01, t_sigma=0.02, f_sigma_ratio=0.02, X_sigma=0.02, obs_sigma=0.2):
        """Perturb the GT scene to build a (camera_params0, X0, points_2d0) initial guess,
        the same way ipynb/ncam_ba.ipynb injects noise to the initial guess."""
        rng = scene['rng']
        Nc = scene['Nc']

        camera_params0 = scene['camera_params_gt'].copy()
        camera_params0[:, 0:3] += rng.normal(0, rot_sigma, (Nc, 3))
        camera_params0[:, 3:6] += rng.normal(0, t_sigma, (Nc, 3))
        camera_params0[:, 6] += rng.normal(0, scene['K_gt'][0, 0] * f_sigma_ratio, Nc)

        X0 = scene['X_gt'] + rng.normal(0, X_sigma, scene['X_gt'].shape)
        points_2d0 = scene['points_2d'] + rng.normal(0, obs_sigma, scene['points_2d'].shape)

        return camera_params0, X0, points_2d0

    # ------------------------------------------------------------------
    # Unit-level tests of the mask <-> parameter-vector wiring itself
    # (pycalib.ba.x2c / pycalib.ba.c2x), independent of the optimizer.
    # ------------------------------------------------------------------

    def test_x2c_c2x_fixed_params_are_untouched(self):
        rng = np.random.default_rng(1)
        Nc = 3
        c0 = rng.normal(size=(Nc, 17))

        # R, t, f, u0, v0 (cols 0:9) are free and independent per camera;
        # the distortion block (cols 9:17) is fixed (mask == 0).
        mask = np.zeros((Nc, 17), dtype=int)
        mask[:, 0:9] = np.arange(Nc * 9).reshape((Nc, 9)) + 1

        x = pycalib.ba.c2x(c0, mask)
        self.assertEqual(x.shape, (Nc * 9,))

        c_rec, n_params = pycalib.ba.x2c(x, c0, mask)

        # the fixed block must be preserved exactly, bit for bit
        np.testing.assert_array_equal(c_rec[:, 9:17], c0[:, 9:17])
        # the free (independent) block round-trips exactly
        np.testing.assert_allclose(c_rec[:, 0:9], c0[:, 0:9])
        self.assertEqual(n_params, Nc * 9)

    def test_x2c_c2x_shared_params_are_tied(self):
        rng = np.random.default_rng(2)
        Nc = 4
        # deliberately give every camera a DIFFERENT value in what will become
        # the "shared" columns, so a bug that fails to tie them together
        # would be caught by this test.
        c0 = rng.normal(size=(Nc, 17)) * 10 + 500

        mask = np.zeros((Nc, 17), dtype=int)
        idx = 1
        # R, t (cols 0:6): independent per camera
        for i in range(Nc):
            for j in range(6):
                mask[i, j] = idx
                idx += 1
        # f, u0, v0 (cols 6:9): a single value shared by all cameras
        for j in range(6, 9):
            mask[:, j] = idx
            idx += 1
        # distortion (cols 9:17) stays 0 == fixed

        x = pycalib.ba.c2x(c0, mask)
        # Nc*6 independent dof + 1 shared dof (f,u0,v0 are 3 separate shared groups)
        self.assertEqual(x.shape, (Nc * 6 + 3,))

        c_rec, n_params = pycalib.ba.x2c(x, c0, mask)

        # the shared block must be IDENTICAL across all cameras after reconstruction
        for j in range(6, 9):
            np.testing.assert_array_equal(c_rec[:, j], np.full(Nc, c_rec[0, j]))
        # ... and equal to camera 0's original value, since c2x() takes the value
        # of the first occurrence (smallest camera index) as the shared dof's value
        np.testing.assert_array_equal(c_rec[0, 6:9], c0[0, 6:9])

        # the independent R,t block round-trips exactly per camera
        np.testing.assert_allclose(c_rec[:, 0:6], c0[:, 0:6])

        # the fixed distortion block is preserved exactly
        np.testing.assert_array_equal(c_rec[:, 9:17], c0[:, 9:17])

        self.assertEqual(n_params, Nc * 6 + 3)

    # ------------------------------------------------------------------
    # Integration-level tests: run the full optimizer on a synthetic GT
    # scene and check the mask behavior quantitatively.
    # ------------------------------------------------------------------

    def test_ba_independent_mask_converges_to_gt(self):
        """Sanity baseline: with all params free & independent per camera
        (no fixing, no sharing), bundle_adjustment() should still recover the
        GT scene reasonably well.

        Note: k4/k5/k6 (the rational distortion denominator) are fixed at 0
        here, same as ipynb/ncam_ba.ipynb -- freeing them with mask=None on a
        small/central point cloud is a known ill-conditioned setup (the
        classic focal-length / depth "bas-relief" ambiguity per camera) and
        is unrelated to what this file is testing (the mask plumbing)."""
        scene = self.make_scene(n_cameras=5, seed=0)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        mask = pycalib.ba.make_mask(True, True, f=True, u0=True, v0=True,
                                     k1=True, k2=True, p1=True, p2=True, k3=True,
                                     k4=False, k5=False, k6=False)

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=mask, verbose=0)

        # generous margins: this configuration is not tightly constrained per
        # camera, so we only check that optimization clearly improved things
        # and landed in the right ballpark, not pixel/mm-level precision.
        self.assertLess(e, 1.5)
        np.testing.assert_allclose(X_opt, scene['X_gt'], atol=0.3)

        R_opt, T_opt, K_opt, D_opt = pycalib.ba.decode_camera_params(cam_opt)
        for i in range(scene['Nc']):
            self.assertLess(rotation_angle_deg(R_opt[i], scene['R_gt'][i]), 8.0)
            self.assertLess(np.linalg.norm(T_opt[i] - scene['t_gt'][i]), 1.5)

        # fixed distortion (k4:k6) must be untouched
        np.testing.assert_array_equal(cam_opt[:, 14:17], camera_params0[:, 14:17])

    def test_ba_fixed_distortion_is_not_optimized(self):
        """Parameters masked out (0 / False) must stay exactly as given in the
        initial guess, even when that guess is deliberately wrong."""
        scene = self.make_scene(n_cameras=4, seed=1)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        # deliberately set a "wrong" initial k1 (GT k1 == 0) and fix it via mask
        WRONG_K1 = 0.123456
        camera_params0[:, 9] = WRONG_K1

        mask = pycalib.ba.make_mask(True, True, f=True, u0=True, v0=True,
                                     k1=False, k2=False, p1=False, p2=False,
                                     k3=False, k4=False, k5=False, k6=False)

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=mask, verbose=0)

        # the whole distortion block (including our deliberately wrong k1) must
        # be untouched, bit for bit
        np.testing.assert_array_equal(cam_opt[:, 9:17], camera_params0[:, 9:17])
        self.assertTrue(np.all(cam_opt[:, 9] == WRONG_K1))

        # the free parameters should still have compensated reasonably well
        self.assertLess(e, 3.0)

    def test_ba_shared_intrinsics_are_tied_and_converge_to_gt(self):
        """Reproduces the "moving camera" scenario of ipynb/ncam_ba.ipynb:
        R, t are optimized independently per camera, while f, u0, v0 are
        optimized as a single value shared by all cameras."""
        scene = self.make_scene(n_cameras=5, seed=2)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        mask = pycalib.ba.make_mask(True, True, f=True, u0=True, v0=True,
                                     k1=False, k2=False, p1=False, p2=False,
                                     k3=False, k4=False, k5=False, k6=False)
        mask = np.tile(mask, (scene['Nc'], 1))
        m = np.arange(mask.size).reshape(mask.shape) + 1
        m[mask == False] = 0
        mask = m
        mask[:, 6:9] = mask[0, 6:9]  # share f, u0, v0 across all cameras

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=mask, verbose=0)

        # the shared block must be EXACTLY identical across all cameras
        for j in range(6, 9):
            np.testing.assert_array_equal(cam_opt[:, j], np.full(scene['Nc'], cam_opt[0, j]))

        # ... and it must converge close to the (single, shared) GT value
        self.assertLess(abs(cam_opt[0, 6] - scene['K_gt'][0, 0]), 10.0)
        self.assertLess(abs(cam_opt[0, 7] - scene['K_gt'][0, 2]), 10.0)
        self.assertLess(abs(cam_opt[0, 8] - scene['K_gt'][1, 2]), 10.0)

        # the fixed distortion block is untouched
        np.testing.assert_array_equal(cam_opt[:, 9:17], camera_params0[:, 9:17])

        # overall reprojection error is small
        self.assertLess(e, 1.0)

    def test_ba_shared_vs_independent_intrinsics_contrast(self):
        """Contrast check: unlike the shared-mask case, independently optimized
        per-camera intrinsics are NOT forced to be bit-identical to each other
        (they only happen to converge to similar values)."""
        scene = self.make_scene(n_cameras=5, seed=3)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=None, verbose=0)

        f_opt = cam_opt[:, 6]
        # independently optimized focal lengths are close to each other (since
        # they share the same GT), but not required to be exactly equal
        self.assertLess(np.std(f_opt), 5.0)
        self.assertFalse(np.all(f_opt == f_opt[0]), 'independent params should not be bit-identical by construction')


if __name__ == '__main__':
    unittest.main()
