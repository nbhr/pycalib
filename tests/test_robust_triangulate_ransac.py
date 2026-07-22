import unittest
import numpy as np
import cv2
import pandas as pd
import pycalib


class TestRobustTriangulateRansac(unittest.TestCase):
    """
    Verifies pycalib.robust.triangulate_ransac(), following the same synthetic
    GT scene design as tests/test_robust_triangulate_consensus.py (a real,
    bundled 15-GoPro-camera calibration; random 3D points; occlusion; Gaussian
    pixel noise; injected large outliers), plus checks specific to the RANSAC
    (randomly sampled pairs) behavior itself: it must degenerate to the exact
    same result as triangulate_consensus() when `n_trials` covers every pair,
    and must be reproducible given a fixed seed.
    """

    CALIB_FILE = 'data/ba/output.json'

    def make_scene(self, seed=0, n_frames=10, n_points=3):
        K, D, R, T, _ = pycalib.util.load_calib(self.CALIB_FILE)
        P = np.array([k @ np.hstack([r, t]) for k, r, t in zip(K, R, T)])
        Nc = len(P)
        rng = np.random.default_rng(seed)
        labels = [chr(ord('A') + i) for i in range(n_points)]

        X_gt = (rng.random((n_frames * n_points, 3)) - 0.5) * 100
        X_gt[:, 2] = np.abs(X_gt[:, 2])

        frames = np.repeat(np.arange(n_frames), n_points)
        pt_labels = np.tile(labels, n_frames)
        X_gt_df = pd.DataFrame({
            'frame': frames, 'label': pt_labels,
            'X': X_gt[:, 0], 'Y': X_gt[:, 1], 'Z': X_gt[:, 2],
        })

        rows = []
        for c, (r, t, k, d) in enumerate(zip(R, T, K, D)):
            x, _ = cv2.projectPoints(X_gt.reshape((-1, 1, 3)), cv2.Rodrigues(r)[0], t, k, d)
            x = x.reshape((-1, 2))
            w = np.ones((len(x), 1))
            rows.append(np.hstack([x, w]))
        df_gt = pd.DataFrame(np.array(rows).reshape((-1, 3)), columns=['x', 'y', 'w'])
        df_gt['camera'] = np.repeat(np.arange(Nc), n_frames * n_points)
        df_gt['frame'] = np.tile(frames, Nc)
        df_gt['label'] = np.tile(pt_labels, Nc)

        return {
            'rng': rng, 'K': K, 'D': D, 'R': R, 'T': T, 'P': P, 'Nc': Nc,
            'X_gt_df': X_gt_df, 'df_gt': df_gt,
            'scale': np.mean(np.abs(X_gt)),
        }

    def error_to_gt(self, X_est_df, X_gt_df, cols=('X', 'Y', 'Z')):
        est = X_est_df[['frame', 'label', *cols]].rename(columns=dict(zip(cols, ['X_est', 'Y_est', 'Z_est'])))
        gt = X_gt_df.rename(columns={'X': 'X_gt', 'Y': 'Y_gt', 'Z': 'Z_gt'})
        m = pd.merge(gt, est, on=['frame', 'label'])
        d = m[['X_est', 'Y_est', 'Z_est']].to_numpy().astype(float) - m[['X_gt', 'Y_gt', 'Z_gt']].to_numpy().astype(float)
        return np.linalg.norm(d, axis=1)

    def drop_random_rows(self, df, rng, ratio):
        idx = rng.choice(df.index, int(len(df) * ratio), replace=False)
        return df.drop(idx).reset_index(drop=True)

    def add_gaussian_noise(self, df, rng, sigma_px):
        df = df.copy()
        df['x'] += rng.normal(scale=sigma_px, size=len(df))
        df['y'] += rng.normal(scale=sigma_px, size=len(df))
        return df

    def add_outliers(self, df, rng, ratio, *, low=100, high=1000):
        df = df.copy()
        idx = rng.choice(df.index, int(len(df) * ratio), replace=False)
        df.loc[idx, 'x'] += rng.uniform(low, high, len(idx)) * rng.choice([-1, 1], len(idx))
        df.loc[idx, 'y'] += rng.uniform(low, high, len(idx)) * rng.choice([-1, 1], len(idx))
        return df, idx

    # ------------------------------------------------------------------
    # RANSAC-specific behavior
    # ------------------------------------------------------------------

    def test_exhaustive_fallback_matches_consensus(self):
        """When n_trials covers every possible camera pair, triangulate_ransac()
        must degenerate to exactly the same result as triangulate_consensus()."""
        scene = self.make_scene(seed=0)
        total_pairs = scene['Nc'] * (scene['Nc'] - 1) // 2

        X_consensus, _ = pycalib.robust.triangulate_consensus(
            scene['df_gt'], scene['P'], distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)
        X_ransac, _ = pycalib.robust.triangulate_ransac(
            scene['df_gt'], scene['P'], n_trials=total_pairs + 1, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)

        m = pd.merge(X_consensus, X_ransac, on=['frame', 'label'], suffixes=('_c', '_r'))
        self.assertEqual(len(m), len(X_consensus))
        np.testing.assert_allclose(
            m[['X_c', 'Y_c', 'Z_c']].to_numpy().astype(float),
            m[['X_r', 'Y_r', 'Z_r']].to_numpy().astype(float))

    def test_reproducible_with_seed(self):
        """The same seed must give bit-identical results; with noisy data and
        few trials, a different seed generally samples different pairs and
        so should (not guaranteed, but overwhelmingly likely) give a
        different result."""
        scene = self.make_scene(seed=1)
        df = self.add_gaussian_noise(scene['df_gt'], scene['rng'], sigma_px=5)

        X1, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], n_trials=5, seed=7, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)
        X2, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], n_trials=5, seed=7, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)
        X3, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], n_trials=5, seed=99, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)

        m12 = pd.merge(X1, X2, on=['frame', 'label'], suffixes=('_1', '_2'))
        np.testing.assert_array_equal(
            m12[['X_1', 'Y_1', 'Z_1']].to_numpy(), m12[['X_2', 'Y_2', 'Z_2']].to_numpy())

        m13 = pd.merge(X1, X3, on=['frame', 'label'], suffixes=('_1', '_3'))
        same = np.array_equal(m13[['X_1', 'Y_1', 'Z_1']].to_numpy(), m13[['X_3', 'Y_3', 'Z_3']].to_numpy())
        self.assertFalse(same, 'different seeds sampling different pairs should generally change the result')

    # ------------------------------------------------------------------
    # ransac_n_trials() formula and the n_trials / success_prob / inlier_ratio
    # priority rule in triangulate_ransac()
    # ------------------------------------------------------------------

    def test_ransac_n_trials_matches_known_values(self):
        """Cross-check against the standard RANSAC trial-count table
        (e.g. Hartley & Zisserman): for a 2-point sample, 50% inlier ratio,
        99% success probability, the textbook answer is 17 trials."""
        self.assertEqual(pycalib.robust.ransac_n_trials(0.99, 0.5, sample_size=2), 17)
        self.assertEqual(pycalib.robust.ransac_n_trials(0.99, 1.0, sample_size=2), 1)

        # monotonicity: higher confidence or lower inlier ratio both demand
        # more trials
        self.assertLess(
            pycalib.robust.ransac_n_trials(0.90, 0.5),
            pycalib.robust.ransac_n_trials(0.99, 0.5))
        self.assertLess(
            pycalib.robust.ransac_n_trials(0.99, 0.9),
            pycalib.robust.ransac_n_trials(0.99, 0.1))

        with self.assertRaises(AssertionError):
            pycalib.robust.ransac_n_trials(1.0, 0.5)   # success_prob must be < 1
        with self.assertRaises(AssertionError):
            pycalib.robust.ransac_n_trials(0.99, 0.0)  # inlier_ratio must be > 0

    def test_explicit_n_trials_takes_priority(self):
        """When n_trials is given explicitly, it must be honored as-is and
        success_prob / inlier_ratio must be ignored entirely -- even absurd
        values for the latter two must not change the result."""
        scene = self.make_scene(seed=20)
        df = self.add_gaussian_noise(scene['df_gt'], scene['rng'], sigma_px=5)

        X1, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], n_trials=5, seed=1, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)
        X2, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], n_trials=5, success_prob=0.999999, inlier_ratio=0.01, seed=1,
            distorted=True, camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)

        m = pd.merge(X1, X2, on=['frame', 'label'], suffixes=('_1', '_2'))
        np.testing.assert_array_equal(m[['X_1', 'Y_1', 'Z_1']].to_numpy(), m[['X_2', 'Y_2', 'Z_2']].to_numpy())

    def test_n_trials_none_derives_from_success_prob_and_inlier_ratio(self):
        """When n_trials is not given, triangulate_ransac() must derive it
        from success_prob/inlier_ratio via ransac_n_trials() -- verified
        indirectly by comparing against an explicit call using the value
        ransac_n_trials() itself returns for the same inputs."""
        scene = self.make_scene(seed=21)
        df = self.add_gaussian_noise(scene['df_gt'], scene['rng'], sigma_px=5)

        success_prob, inlier_ratio = 0.9, 0.4
        expected_n_trials = pycalib.robust.ransac_n_trials(success_prob, inlier_ratio)

        X1, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], success_prob=success_prob, inlier_ratio=inlier_ratio, seed=1,
            distorted=True, camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)
        X2, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], n_trials=expected_n_trials, seed=1,
            distorted=True, camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)

        m = pd.merge(X1, X2, on=['frame', 'label'], suffixes=('_1', '_2'))
        np.testing.assert_array_equal(m[['X_1', 'Y_1', 'Z_1']].to_numpy(), m[['X_2', 'Y_2', 'Z_2']].to_numpy())

    # ------------------------------------------------------------------
    # Same accuracy scenarios as test_robust_triangulate_consensus.py, but
    # using only a subset of camera pairs.
    # ------------------------------------------------------------------

    def test_exact_recovery_no_noise_no_occlusion(self):
        scene = self.make_scene(seed=10)

        X, _ = pycalib.robust.triangulate_ransac(
            scene['df_gt'], scene['P'], n_trials=20, seed=1, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)

        np.testing.assert_equal(X['n_outliers'].to_numpy(), 0)
        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.max(), 1e-3)

    def test_exact_recovery_with_occlusion(self):
        scene = self.make_scene(seed=11)
        df_occ = self.drop_random_rows(scene['df_gt'], scene['rng'], ratio=0.5)

        X, _ = pycalib.robust.triangulate_ransac(
            df_occ, scene['P'], n_trials=20, seed=2, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)

        np.testing.assert_equal(X['n_outliers'].to_numpy(), 0)
        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.max(), 1e-3)

    def test_gaussian_noise_bounded_error(self):
        scene = self.make_scene(seed=12)
        df = self.drop_random_rows(scene['df_gt'], scene['rng'], ratio=0.5)
        df = self.add_gaussian_noise(df, scene['rng'], sigma_px=5)

        X, _ = pycalib.robust.triangulate_ransac(
            df, scene['P'], n_trials=25, seed=3, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)

        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.mean() / scene['scale'], 0.2)

    def test_outlier_rejection_recall_and_accuracy(self):
        scene = self.make_scene(seed=13)
        df = self.drop_random_rows(scene['df_gt'], scene['rng'], ratio=0.5)
        df_noisy = self.add_gaussian_noise(df, scene['rng'], sigma_px=5)
        df_out, outlier_idx = self.add_outliers(df_noisy, scene['rng'], ratio=0.1)

        X, _ = pycalib.robust.triangulate_ransac(
            df_out, scene['P'], n_trials=40, seed=4, distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'], nproc=1, show_pbar=False)

        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.mean() / scene['scale'], 0.2)

        is_outlier = pd.Series(False, index=df_noisy.index)
        is_outlier.loc[outlier_idx] = True
        df_marked = df_noisy.copy()
        df_marked['is_outlier'] = is_outlier

        tp = fn = 0
        for (frame, label), grp in df_marked.groupby(['frame', 'label']):
            gt_cams = set(grp.loc[grp['is_outlier'], 'camera'].tolist())
            if not gt_cams:
                continue
            row = X[(X['frame'] == frame) & (X['label'] == label)]
            est_cams = set(row.iloc[0]['outliers']) if len(row) else set()
            tp += len(gt_cams & est_cams)
            fn += len(gt_cams - est_cams)

        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        self.assertIsNotNone(recall)
        self.assertGreaterEqual(recall, 0.7)


if __name__ == '__main__':
    unittest.main()
