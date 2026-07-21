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

    def test_x2c_no_fixed_entries_at_all(self):
        """Regression guard: if a mask has NO fixed (0) entry anywhere, 0 is
        absent from the mask's unique-value set. x2c() used to derive each
        entry's column as "rank among all unique values, minus one", which
        is only valid when 0 is present -- otherwise it silently shifts
        every column by one (wrapping around for the smallest value)."""
        rng = np.random.default_rng(7)
        Nc = 3
        c0 = rng.normal(size=(Nc, 17))
        mask = np.arange(Nc * 17).reshape((Nc, 17)) + 1  # every entry free, none fixed

        x = np.arange(1, Nc * 17 + 1) * 100.0  # distinct, recognizable values
        c_rec, n_params = pycalib.ba.x2c(x, c0, mask)

        np.testing.assert_array_equal(c_rec.flatten(), x)
        self.assertEqual(n_params, Nc * 17)

    def test_x2c_c2x_generalize_to_3d_points(self):
        """The same fixed/shared mechanism, applied to a (Np, 3) 3D point
        array instead of a (Nc, 17) camera-parameter array."""
        rng = np.random.default_rng(8)
        Np = 5
        c0 = rng.normal(size=(Np, 3)) * 10

        mask = np.zeros((Np, 3), dtype=int)
        idx = 1
        # points 0,1: independent (free) per-coordinate
        for p in range(2):
            for s in range(3):
                mask[p, s] = idx
                idx += 1
        # points 2,3: share a single XYZ value (e.g. a duplicated observation
        # of the same physical point under two different labels)
        for s in range(3):
            mask[2, s] = idx
            mask[3, s] = idx
            idx += 1
        # point 4 stays 0 == fixed

        x = pycalib.ba.c2x(c0, mask)
        self.assertEqual(x.shape, (2 * 3 + 3,))

        c_rec, n_params = pycalib.ba.x2c(x, c0, mask)

        np.testing.assert_allclose(c_rec[0:2], c0[0:2])
        np.testing.assert_array_equal(c_rec[2], c_rec[3])
        np.testing.assert_array_equal(c_rec[2], c0[2])
        np.testing.assert_array_equal(c_rec[4], c0[4])
        self.assertEqual(n_params, 2 * 3 + 3)

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

    def test_ba_fixed_point_is_not_optimized(self):
        """A 3D point masked out (0 / False) via `point_mask` must stay
        exactly as given in the initial guess, even when that guess is
        deliberately wrong. (With all cameras free and only one point
        fixed, the rest of the reconstruction is free to drift/rotate/scale
        to compensate -- that's expected gauge freedom, not a bug, so this
        test only asserts the fixed point itself is untouched.)"""
        scene = self.make_scene(n_cameras=5, seed=4)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        WRONG_POINT = np.array([999.0, -999.0, 999.0])
        X0 = X0.copy()
        X0[0] = WRONG_POINT

        point_mask = np.ones(X0.shape, dtype=bool)
        point_mask[0, :] = False

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            point_mask=point_mask, verbose=0)

        np.testing.assert_array_equal(X_opt[0], WRONG_POINT)

    def test_ba_fixed_control_points_anchor_reconstruction(self):
        """Realistic use case: some 3D points are known in advance (e.g.
        surveyed control points) and fixed at their true GT location, while
        the rest of the points AND all camera poses are free. The fixed
        points must stay exactly at GT, and -- because they properly anchor
        the scene's gauge freedom (unlike a single arbitrarily-placed fixed
        point) -- the free points and camera poses should converge very
        close to GT too."""
        scene = self.make_scene(n_cameras=5, seed=6)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        fixed_idx = np.arange(0, scene['Np'], 2)  # every other point is a known control point
        free_idx = np.setdiff1d(np.arange(scene['Np']), fixed_idx)
        X0[fixed_idx] = scene['X_gt'][fixed_idx]  # snap control points back to their true position

        point_mask = np.ones(X0.shape, dtype=bool)
        point_mask[fixed_idx] = False

        mask = pycalib.ba.make_mask(True, True, f=True, u0=True, v0=True,
                                     k1=True, k2=True, p1=True, p2=True, k3=True,
                                     k4=False, k5=False, k6=False)

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=mask, point_mask=point_mask, verbose=0)

        # fixed control points are untouched, bit for bit
        np.testing.assert_array_equal(X_opt[fixed_idx], scene['X_gt'][fixed_idx])

        # free points converge tightly to GT, thanks to the fixed anchors
        np.testing.assert_allclose(X_opt[free_idx], scene['X_gt'][free_idx], atol=0.05)

        # camera poses converge reasonably too
        R_opt, T_opt, K_opt, D_opt = pycalib.ba.decode_camera_params(cam_opt)
        for i in range(scene['Nc']):
            self.assertLess(rotation_angle_deg(R_opt[i], scene['R_gt'][i]), 8.0)
            self.assertLess(np.linalg.norm(T_opt[i] - scene['t_gt'][i]), 1.5)

        self.assertLess(e, 1.0)

    def test_ba_all_points_and_intrinsics_fixed_reduces_to_pnp(self):
        """All 3D points fixed at their (known) GT location, and camera
        intrinsics (K, distortion) fixed too: only R, t are free per camera.
        This is functionally the same problem PnP solves (pose-only
        refinement against known 3D structure and fixed calibration), so
        camera poses should converge tightly to GT."""
        scene = self.make_scene(n_cameras=5, seed=10)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        # 3D points are fully known -- fix all of them at their true position
        X0 = scene['X_gt'].copy()
        point_mask = np.zeros(X0.shape, dtype=bool)

        # intrinsics (f, u0, v0, k1:k6) fixed; only R, t are free
        mask = pycalib.ba.make_mask(True, True, f=False, u0=False, v0=False,
                                     k1=False, k2=False, p1=False, p2=False,
                                     k3=False, k4=False, k5=False, k6=False)

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=mask, point_mask=point_mask, verbose=0)

        # 3D points and intrinsics are untouched, bit for bit
        np.testing.assert_array_equal(X_opt, X0)
        np.testing.assert_array_equal(cam_opt[:, 6:17], camera_params0[:, 6:17])

        # only R, t were free to move, and they should converge tightly to
        # GT since both the 3D structure and the intrinsics are exact (this
        # is exactly the PnP setup: known 3D points + known K -> pose only)
        R_opt, T_opt, K_opt, D_opt = pycalib.ba.decode_camera_params(cam_opt)
        for i in range(scene['Nc']):
            self.assertLess(rotation_angle_deg(R_opt[i], scene['R_gt'][i]), 1.0)
            self.assertLess(np.linalg.norm(T_opt[i] - scene['t_gt'][i]), 0.5)

        self.assertLess(e, 1.0)

    def test_ba_all_points_fixed_updates_intrinsics_and_extrinsics(self):
        """All 3D points fixed at their (known) GT location, but intrinsics
        left free this time: both R, t (extrinsics) AND f, u0, v0, k1:k3
        (intrinsics) should be optimized and converge close to GT -- unlike
        the PnP-like test above where intrinsics stay untouched."""
        scene = self.make_scene(n_cameras=5, seed=11)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        X0 = scene['X_gt'].copy()
        point_mask = np.zeros(X0.shape, dtype=bool)

        # k4:k6 (rational terms) stay fixed for the same reason as the other
        # tests in this file; everything else (R, t, f, u0, v0, k1:k3) is free
        mask = pycalib.ba.make_mask(True, True, f=True, u0=True, v0=True,
                                     k1=True, k2=True, p1=True, p2=True, k3=True,
                                     k4=False, k5=False, k6=False)

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=mask, point_mask=point_mask, verbose=0)

        # 3D points and the fixed k4:k6 block are untouched, bit for bit
        np.testing.assert_array_equal(X_opt, X0)
        np.testing.assert_array_equal(cam_opt[:, 14:17], camera_params0[:, 14:17])

        # unlike the PnP-like test, intrinsics were free and must have
        # actually moved from their (noisy) initial guess
        self.assertFalse(np.allclose(cam_opt[:, 6], camera_params0[:, 6]))

        # both intrinsics and extrinsics should converge reasonably close to GT
        R_opt, T_opt, K_opt, D_opt = pycalib.ba.decode_camera_params(cam_opt)
        for i in range(scene['Nc']):
            self.assertLess(rotation_angle_deg(R_opt[i], scene['R_gt'][i]), 8.0)
            self.assertLess(np.linalg.norm(T_opt[i] - scene['t_gt'][i]), 1.5)
            self.assertLess(abs(K_opt[i, 0, 0] - scene['K_gt'][0, 0]), 10.0)

        self.assertLess(e, 1.0)

    def test_ba_shared_vs_independent_intrinsics_contrast(self):
        """Contrast check: unlike the shared-mask case, independently optimized
        per-camera intrinsics are NOT forced to be bit-identical to each other
        (they only happen to converge to similar values)."""
        scene = self.make_scene(n_cameras=5, seed=3)
        camera_params0, X0, points_2d0 = self.perturb_initial_guess(scene)

        # same "sensible" mask as the other tests (k4:k6 fixed at 0) -- see
        # the note in test_ba_independent_mask_converges_to_gt about why
        # mask=None (freeing the rational k4:k6 terms too) is a poorly
        # constrained setup unrelated to what this test checks.
        mask = pycalib.ba.make_mask(True, True, f=True, u0=True, v0=True,
                                     k1=True, k2=True, p1=True, p2=True, k3=True,
                                     k4=False, k5=False, k6=False)

        cam_opt, X_opt, e, res = pycalib.ba.bundle_adjustment(
            camera_params0, X0, scene['camera_indices'], scene['point_indices'], points_2d0,
            mask=mask, verbose=0)

        # reprojection error is small (this is a valid local solution, not a
        # divergence) even though each camera's focal length individually
        # wanders around the shared GT value (the classic per-camera
        # focal-length / depth ambiguity) -- that spread is expected and is
        # exactly the point of this contrast test.
        self.assertLess(e, 1.0)
        f_opt = cam_opt[:, 6]
        self.assertLess(np.std(f_opt), 30.0)
        self.assertFalse(np.all(f_opt == f_opt[0]), 'independent params should not be bit-identical by construction')


if __name__ == '__main__':
    unittest.main()
