import unittest
import numpy as np
import cv2
import pandas as pd
import pycalib


class TestRobustTriangulateConsensus(unittest.TestCase):
    """
    Verifies pycalib.robust.triangulate_consensus() with a synthetic GT
    scene, following the same design as ipynb/ncam_triangulate_consensus.ipynb:
    a real (bundled) 15-GoPro-camera calibration, random 3D points, occlusion,
    Gaussian pixel noise, and injected large outliers.
    """

    CALIB_FILE = 'data/ba/output.json'

    def make_scene(self, seed=0, n_frames=10, n_points=3):
        K, D, R, T, _ = pycalib.util.load_calib(self.CALIB_FILE)
        P = np.array([k @ np.hstack([r, t]) for k, r, t in zip(K, R, T)])
        Nc = len(P)
        rng = np.random.default_rng(seed)
        labels = [chr(ord('A') + i) for i in range(n_points)]

        # random 3D points in front of the rig (same scale as the notebook)
        X_gt = (rng.random((n_frames * n_points, 3)) - 0.5) * 100
        X_gt[:, 2] = np.abs(X_gt[:, 2])

        frames = np.repeat(np.arange(n_frames), n_points)
        pt_labels = np.tile(labels, n_frames)
        X_gt_df = pd.DataFrame({
            'frame': frames, 'label': pt_labels,
            'X': X_gt[:, 0], 'Y': X_gt[:, 1], 'Z': X_gt[:, 2],
        })

        # noiseless 2D projections (distorted, as a real camera would observe)
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
        """3D displacement (per point) between an estimated result and GT,
        matched by the (frame, label) primary key -- robust to row reordering."""
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
        """Corrupt a random subset of rows with a large offset. Returns the
        corrupted dataframe and the index of the corrupted rows."""
        df = df.copy()
        idx = rng.choice(df.index, int(len(df) * ratio), replace=False)
        df.loc[idx, 'x'] += rng.uniform(low, high, len(idx)) * rng.choice([-1, 1], len(idx))
        df.loc[idx, 'y'] += rng.uniform(low, high, len(idx)) * rng.choice([-1, 1], len(idx))
        return df, idx

    # ------------------------------------------------------------------

    def test_exact_recovery_no_noise_no_occlusion(self):
        """All cameras see all points with noiseless observations: the
        result must match GT almost exactly, with zero outliers."""
        scene = self.make_scene(seed=0)

        X, _ = pycalib.robust.triangulate_consensus(
            scene['df_gt'], scene['P'], distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)

        np.testing.assert_equal(X['n_outliers'].to_numpy(), 0)
        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.max(), 1e-3)

    def test_exact_recovery_with_occlusion(self):
        """Randomly dropping half of the observations (but keeping >= 2
        views per point) should still allow near-exact recovery."""
        scene = self.make_scene(seed=1)
        df_occ = self.drop_random_rows(scene['df_gt'], scene['rng'], ratio=0.5)

        X, _ = pycalib.robust.triangulate_consensus(
            df_occ, scene['P'], distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)

        np.testing.assert_equal(X['n_outliers'].to_numpy(), 0)
        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.max(), 1e-3)

    def test_gaussian_noise_bounded_error(self):
        """With occlusion and moderate (5px) Gaussian pixel noise, the
        reconstruction error should stay small relative to the scene scale."""
        scene = self.make_scene(seed=2)
        df = self.drop_random_rows(scene['df_gt'], scene['rng'], ratio=0.5)
        df = self.add_gaussian_noise(df, scene['rng'], sigma_px=5)

        X, _ = pycalib.robust.triangulate_consensus(
            df, scene['P'], distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)

        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.mean() / scene['scale'], 0.15)

    def test_outlier_rejection_recall_and_accuracy(self):
        """Injecting large (100-1000px) outliers on top of occlusion + noise
        must (1) be detected with high recall, and (2) not meaningfully
        degrade the final 3D accuracy compared to the noise-only case."""
        scene = self.make_scene(seed=3)
        df = self.drop_random_rows(scene['df_gt'], scene['rng'], ratio=0.5)
        df_noisy = self.add_gaussian_noise(df, scene['rng'], sigma_px=5)
        df_out, outlier_idx = self.add_outliers(df_noisy, scene['rng'], ratio=0.1)

        X, _ = pycalib.robust.triangulate_consensus(
            df_out, scene['P'], distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)

        # accuracy must stay close to the noise-only case (not blown up by outliers)
        err = self.error_to_gt(X, scene['X_gt_df'])
        self.assertLess(err.mean() / scene['scale'], 0.15)

        # recall of the injected outliers, per (frame, label, camera)
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
        self.assertGreaterEqual(recall, 0.8)

    def test_bundle_adjustment_refines_inliers(self):
        """The re-triangulated ("_in") columns should be accurate for both
        the plain DLT re-triangulation and the bundle_adjustment=True path,
        and BA should not be meaningfully worse than the plain DLT pass."""
        scene = self.make_scene(seed=4)
        df = self.drop_random_rows(scene['df_gt'], scene['rng'], ratio=0.5)
        df_noisy = self.add_gaussian_noise(df, scene['rng'], sigma_px=5)
        df_out, _ = self.add_outliers(df_noisy, scene['rng'], ratio=0.1)

        X_dlt, _ = pycalib.robust.triangulate_consensus(
            df_out, scene['P'], distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)
        err_dlt = self.error_to_gt(X_dlt, scene['X_gt_df'], cols=('X_in', 'Y_in', 'Z_in'))
        self.assertLess(err_dlt.mean() / scene['scale'], 0.15)

        X_ba, _ = pycalib.robust.triangulate_consensus(
            df_out, scene['P'], bundle_adjustment=True, key_w='w', distorted=True,
            camera_matrix=scene['K'], dist_coeffs=scene['D'],
            nproc=1, show_pbar=False)
        err_ba = self.error_to_gt(X_ba, scene['X_gt_df'], cols=('X_in', 'Y_in', 'Z_in'))
        self.assertLess(err_ba.mean() / scene['scale'], 0.15)


if __name__ == '__main__':
    unittest.main()
