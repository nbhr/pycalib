import cv2
import numpy as np
import pandas as pd
import pycalib
import multiprocessing
import itertools
from tqdm.auto import tqdm
from scipy.optimize import least_squares

def undistort_inplace(df, cid, camera_matrix, dist_coeffs, *, key_cid='camera', key_x='x', key_y='y'):
    """
    Undistort 2D points in DataFrame in place

    Parameters
    ----------
    df : pandas.DataFrame
        list of 2D points
    cid : int
        target camera id
    camera_matrix : ndarray
        camera matrix
    dist_coeffs : ndarray
        distortion coefficients
    key_cid : str
        column name of camera id
    key_x : str
        column name of x
    key_y : str
        column name of y
    """

    df_c = df[df[key_cid] == cid]
    if len(df_c) == 0:
        return
    pt2d = df_c[[key_x, key_y]].to_numpy()
    df.loc[df_c.index, [key_x, key_y]] = pycalib.calib.undistort_points(pt2d.reshape((-1, 1, 2)), camera_matrix, dist_coeffs).reshape((-1, 2))


def triangulate(df, P, cid1, cid2, *, reproj_th=10, primary_key=['frame', 'label'], key_cid='camera', key_x='x', key_y='y'):
    if 'index' in df:
        df = df.drop(columns='index')

    # Valid pairs == inner-join on frame and keypoint
    d = pd.merge(df[df[key_cid] == cid1], df[df[key_cid] == cid2], on=primary_key, suffixes=[f'_{cid1}', f'_{cid2}'])
    if len(d) == 0:
        raise Exception(f'No valid corresponding points between {cid1} and {cid2}')
    #print(f'Found {len(d)} valid corresponding points between {cid1} and {cid2}')

    # Triangulate
    p1 = d[[f'{key_x}_{cid1}', f'{key_y}_{cid1}']].to_numpy()
    p2 = d[[f'{key_x}_{cid2}', f'{key_y}_{cid2}']].to_numpy()
    X = cv2.triangulatePoints(P[cid1], P[cid2], p1.T, p2.T)
    X = X / X[3]

    # Reproject to all cameras (un-normalized; the 3rd row is each camera's depth)
    x = np.einsum('cij,jp->cip', P, X)

    # Cheirality check: discard points that are not in front of BOTH triangulating views.
    # (An all-behind pair, e.g. two opposite-facing cameras in a 360-degree rig, is a
    # normal occurrence -- not an error -- so we don't raise, just contribute nothing.)
    front = (x[cid1, 2, :] > 0) & (x[cid2, 2, :] > 0)
    d = d[front].reset_index(drop=True)
    X = X[:, front]
    x = x[:, :, front]

    if len(d) == 0:
        for col in ['X', 'Y', 'Z', 'reproj', 'outliers', 'inliers', 'n_outliers', 'n_inliers']:
            d[col] = []
        return d.convert_dtypes()

    x = x / x[:,2:3,:]

    # Merge the other 2D observation by left-outer-join on frame and keypoint
    for c in range(len(P)):
        if c in [cid1, cid2]:
            continue
        d = pd.merge(d, df[df[key_cid] == c].rename(columns={key_cid:f'{key_cid}_{c}', key_x:f'{key_x}_{c}', key_y:f'{key_y}_{c}'}), on=primary_key, how='left')

    # Extract N x (x0, y0, x1, y1, ...)
    colnames = [x for c in range(len(P)) for x in [f'{key_x}_{c}', f'{key_y}_{c}']]
    y = d[colnames].to_numpy()

    # Reshape to N x C x 2
    y = y.reshape((-1, len(P), 2))

    # Transpose to C x 2 x N
    y = np.rollaxis(y, 0, 3)

    # Reprojection error
    e = x[:,:2,:] - y
    e = np.linalg.norm(e, axis=1)

    # Save the triangulated 3D points, inliers, and outliers
    d['X'] = X[0]
    d['Y'] = X[1]
    d['Z'] = X[2]

    d['reproj'] = list(map(lambda x : tuple(x), e.T))
    d['outliers'] = [tuple(np.where(i)[0]) for i in (e >= reproj_th).T]
    d['inliers'] = [tuple(np.where(i)[0]) for i in (e < reproj_th).T]
    d['n_outliers'] = np.sum(e >= reproj_th, axis=0).astype(int)
    d['n_inliers'] = np.sum(e < reproj_th, axis=0).astype(int)
    #d['n_visibles'] = np.sum(np.sum(np.isnan(y), axis=1) == 0, axis=0).astype(int)

    return d.convert_dtypes()


def merge(dfs, *, primary_key=['frame', 'label'], key_n_inliers='n_inliers', key_others=['X', 'Y', 'Z', 'reproj', 'outliers', 'inliers', 'n_outliers']):
    SUFFIX = 'tmp'
    KEYS = primary_key
    VALS = key_others + [key_n_inliers]
    VALS_y = [f'{v}_{SUFFIX}' for v in VALS]

    # merge points
    df = None
    for d in dfs:
        for c in KEYS + VALS:
            if not c in d:
                print(d)
                raise Exception(f'{d} does not have "{c}" column')

        d = d[KEYS+VALS]
        if df is None:
            df = d
            continue

        di = pd.merge(df, d, on=KEYS, how='outer', suffixes=['', f'_{SUFFIX}'])

        # import if the input has more inliers, or the point does not in the current list
        idx = (di[key_n_inliers] < di[f'{key_n_inliers}_{SUFFIX}']) | di[key_n_inliers].isna()
        #print(f'Found {len(di)} points in total (new {np.sum(idx)} points from {f})')
        di.loc[idx,VALS] = di.loc[idx,VALS_y].rename(columns={f'{v}_{SUFFIX}':v for v in VALS})
        df = di.loc[:,~di.columns.str.contains(f'_{SUFFIX}$', case=False)]

    df = df.sort_values(by=KEYS)
    return df


def triangulate_wrapper(x):
    df, P, c1, c2, reproj_th, primary_key, key_cid, key_x, key_y = x
    return triangulate(df, P, c1, c2, reproj_th=reproj_th, primary_key=primary_key, key_cid=key_cid, key_x=key_x, key_y=key_y)

def reprojection_error_1pt_nview(pt3d, pt2d, P, weight=None):
    pt3dh = np.append(pt3d.flatten(), 1.0)
    p = np.einsum('ijk,k->ij', P, pt3dh)
    p = p / p[:,2:3]
    e = p[:,:2] - pt2d

    if weight is not None:
        e = e * weight[:,None]

    return e.flatten()

def triangulate_inliers(df_undistorted, X, P, *, primary_key=['frame', 'label'], key_cid='camera', key_x='x', key_y='y', bundle_adjustment=False, key_w=None):
    inliers = X[primary_key + ['inliers']]
    df = pd.merge(df_undistorted, inliers, how='inner', on=primary_key)

    for c in sorted(df[key_cid].unique()):
        idx = np.array([(c in i and c == j) for i,j in zip(df['inliers'].to_list(), df[key_cid].to_list())])
        dfc = df.loc[idx]
        if key_w is None:
            df = pd.merge(df, dfc[primary_key+[key_x, key_y]], how='left', on=primary_key, suffixes=['', f'_{c}'])
        else:
            df = pd.merge(df, dfc[primary_key+[key_x, key_y, key_w]], how='left', on=primary_key, suffixes=['', f'_{c}'])
    if key_w is None:
        df = df.drop(columns=[key_x, key_y, key_cid])
    else:
        df = df.drop(columns=[key_x, key_y, key_cid, key_w])
    df = df.drop_duplicates(primary_key)
    df = df.sort_values(primary_key)

    # do triangulation for each set of inliers
    df_all = []
    for inliers, d in df.groupby(by='inliers'):
        Pg = P[list(inliers)] # Nc x 3 x 4
        pt2d = []
        for c in inliers:
            p = d[[f'{key_x}_{c}', f'{key_y}_{c}']]
            pt2d.append(p.to_numpy())
        pt2d = np.array(pt2d) # Nc x Np x 2
        pt3d = pycalib.calib.triangulate_Npts(pt2d, Pg)
        if bundle_adjustment:
            if key_w is None:
                weights = np.ones(len(inliers))
            else:
                weights = []
                for c in inliers:
                    w = d[f'{key_w}_{c}']
                    weights.append(w.to_numpy())
                weights = np.array(weights).flatten() # Nc
#            weights = None
#            weights = np.ones(len(inliers))
            for i in range(pt2d.shape[1]):
                p2 = pt2d[:,i,:]
                p3 = pt3d[i].flatten()
                assert p3.shape == (3,), p3
                res = least_squares(reprojection_error_1pt_nview, p3, verbose=False, ftol=1e-4, method='trf', loss='linear', args=(p2, Pg, weights))
                pt3d[i] = res['x'].flatten()
        d[['X', 'Y', 'Z']] = pt3d
        df_all.append(d)
    df = pd.concat(df_all).sort_values(primary_key)
    df = pd.merge(X, df[primary_key+['X','Y','Z']], how='inner', on=primary_key, suffixes=['', '_in'])
    return df

def _triangulate_pairs(df, P, pairs, *, nproc, reproj_th, re_triangulate, bundle_adjustment, show_pbar, primary_key, key_cid, key_x, key_y, key_w, distorted, camera_matrix, dist_coeffs):
    """Shared core of triangulate_consensus() / triangulate_ransac(): triangulate
    every point visible in each of the given camera `pairs`, merge the per-point
    best (highest inlier count) result across pairs, and optionally re-triangulate
    using all of that point's identified inlier cameras."""

    cids = df[key_cid].unique()
    if distorted:
        df = df.copy()
        assert camera_matrix is not None and dist_coeffs is not None, 'camera_matrix and dist_coeffs must be provided'
        for c in cids:
            undistort_inplace(df, c, camera_matrix[c], dist_coeffs[c])

    if key_w is not None:
        dfw = df[primary_key+[key_cid, key_x, key_y, key_w]].copy()

    df = df[primary_key+[key_cid, key_x, key_y]].copy()

    # two-view triangulation in parallel
    p = multiprocessing.Pool(multiprocessing.cpu_count() if nproc <= 0 else nproc)
    args = [(df, P, c1, c2, reproj_th, primary_key, key_cid, key_x, key_y) for c1, c2 in pairs]
    dfs = []
    for d in tqdm(p.imap_unordered(triangulate_wrapper, args), total=len(args), disable=not show_pbar):
        dfs.append(d)
    p.close()
    p.join()

    # merge results
    X = merge(dfs, primary_key=primary_key)

    # re-triangulation using all inliers
    if re_triangulate or bundle_adjustment:
        X = triangulate_inliers(df if key_w is None else dfw, X, P, primary_key=primary_key, key_cid=key_cid, key_x=key_x, key_y=key_y, bundle_adjustment=bundle_adjustment, key_w=key_w)

    return X, df


def triangulate_consensus(df, P, *, nproc=0, reproj_th=10, re_triangulate=True, bundle_adjustment=False, show_pbar=True, primary_key=['frame', 'label'], key_cid='camera', key_x='x', key_y='y', key_w=None, distorted=False, camera_matrix=None, dist_coeffs=None):
    """
    Robust n-view triangulation, trying every possible camera pair.

    Parameters
    ----------
    df: pandas.DataFrame
        list of 2D points with columns [primary_key, key_cid, key_x, key_y]
    P: numpy.array
        projection matrices (Nc x 3 x 4)
    nproc: int
        number of concurrent processes
    reproj_th: float
        reprojection error threshold
    re_triangulate: Bool
        do final triangulation using identified inliers
    bundle_adjustment: Bool
        do bundle adjustment to minimize reprojection errors (implies re_triangulate)
    show_pbar: Bool
        show progress bar
    primary_key: list of str
        primary key columns to identify points
    key_cid: str
        column name of camera id
    key_x: str
        column name of x
    key_y: str
        column name of y
    key_w: str
        colum name of weighting factor of each 2D point for scaling its reprojection error
    distorted: Bool
        do undistortion before triangulation (requires camera_matrix and dist_coeffs)
    camera_matrix: numpy.array
        camera matrix (Nc x 3 x 3)
    dist_coeffs: numpy.array
        distortion coefficients (Nc x {4,5,8})

    Returns
    -------
    X: pandas.DataFrame
        triangulated 3D points with columns [primary_key, key_cid, key_x, key_y, X, Y, Z]
    df: pandas.DataFrame
        undistorted 2D points with columns [primary_key, key_cid, key_x, key_y]
    """

    cids = sorted(df[key_cid].unique())
    pairs = list(itertools.combinations(cids, 2))

    return _triangulate_pairs(df, P, pairs, nproc=nproc, reproj_th=reproj_th, re_triangulate=re_triangulate,
                               bundle_adjustment=bundle_adjustment, show_pbar=show_pbar, primary_key=primary_key,
                               key_cid=key_cid, key_x=key_x, key_y=key_y, key_w=key_w, distorted=distorted,
                               camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


def ransac_n_trials(success_prob, inlier_ratio, *, sample_size=2):
    """
    Number of RANSAC trials needed so that, with probability >= `success_prob`,
    at least one trial's random sample of `sample_size` observations is made
    entirely of inliers.

    N >= log(1 - success_prob) / log(1 - inlier_ratio**sample_size)

    Parameters
    ----------
    success_prob: float
        desired probability (0, 1) of drawing at least one all-inlier sample
    inlier_ratio: float
        assumed probability (0, 1] that a single randomly chosen observation
        (here: a single camera, for a given 3D point) is an inlier
    sample_size: int
        number of observations per trial (2 for two-view triangulation)

    Returns
    -------
    n_trials: int
    """

    assert 0 < success_prob < 1, success_prob
    assert 0 < inlier_ratio <= 1, inlier_ratio

    if inlier_ratio >= 1:
        return 1

    n_trials = np.log(1 - success_prob) / np.log(1 - inlier_ratio**sample_size)
    return max(1, int(np.ceil(n_trials)))


def triangulate_ransac(df, P, *, n_trials=None, success_prob=0.99, inlier_ratio=0.5, seed=None, nproc=0, reproj_th=10, re_triangulate=True, bundle_adjustment=False, show_pbar=True, primary_key=['frame', 'label'], key_cid='camera', key_x='x', key_y='y', key_w=None, distorted=False, camera_matrix=None, dist_coeffs=None):
    """
    Robust n-view triangulation via RANSAC: instead of exhaustively trying every
    camera pair like triangulate_consensus() does (an O(Nc^2) cost that becomes
    expensive as the number of cameras grows), only a limited number of randomly
    sampled camera pairs are tried.

    For each sampled pair, triangulate() already (1) triangulates the point,
    (2) discards it if it isn't in front of both cameras of the pair, and
    (3) reprojects it to the remaining cameras to count reprojection-error
    inliers. After all trials, merge() keeps -- independently for each 2D
    point -- whichever trial produced the most inliers, and (if
    `re_triangulate` or `bundle_adjustment`) triangulate_inliers() does the
    final triangulation using every camera in that best inlier set. This is
    the same "pick 2 views -> triangulate -> cheirality check -> count
    inliers -> repeat -> keep the best inlier set -> re-triangulate with all
    of it" procedure as triangulate_consensus(), just over a random subset of
    pairs rather than all of them.

    The number of trials can be given either directly via `n_trials`, or
    derived from a target `success_prob` and an assumed `inlier_ratio` using
    the standard RANSAC formula (see ransac_n_trials()). If `n_trials` is
    given explicitly, it always takes precedence and `success_prob` /
    `inlier_ratio` are ignored. Note that `inlier_ratio` here is a single,
    global assumption shared by every 3D point in `df` (all trials sample
    from the same set of camera pairs across all points), so treat it as a
    conservative estimate of the worst-case inlier ratio among your points,
    not a per-point guarantee.

    Parameters
    ----------
    df: pandas.DataFrame
        list of 2D points with columns [primary_key, key_cid, key_x, key_y]
    P: numpy.array
        projection matrices (Nc x 3 x 4)
    n_trials: int or None
        number of randomly sampled camera pairs to try. If this is >= the
        total number of possible pairs, all pairs are used (falls back to
        the same exhaustive behavior as triangulate_consensus()). If None
        (default), it is computed from `success_prob` and `inlier_ratio`.
    success_prob: float
        used only when `n_trials` is None: desired probability (0, 1) of
        sampling at least one all-inlier camera pair. Default 0.99.
    inlier_ratio: float
        used only when `n_trials` is None: assumed probability (0, 1] that a
        randomly chosen camera observes a given 3D point as an inlier.
        Default 0.5.
    seed: int or None
        seed for the random pair sampling, for reproducibility
    nproc: int
        number of concurrent processes
    reproj_th: float
        reprojection error threshold
    re_triangulate: Bool
        do final triangulation using identified inliers
    bundle_adjustment: Bool
        do bundle adjustment to minimize reprojection errors (implies re_triangulate)
    show_pbar: Bool
        show progress bar
    primary_key: list of str
        primary key columns to identify points
    key_cid: str
        column name of camera id
    key_x: str
        column name of x
    key_y: str
        column name of y
    key_w: str
        colum name of weighting factor of each 2D point for scaling its reprojection error
    distorted: Bool
        do undistortion before triangulation (requires camera_matrix and dist_coeffs)
    camera_matrix: numpy.array
        camera matrix (Nc x 3 x 3)
    dist_coeffs: numpy.array
        distortion coefficients (Nc x {4,5,8})

    Returns
    -------
    X: pandas.DataFrame
        triangulated 3D points with columns [primary_key, key_cid, key_x, key_y, X, Y, Z]
    df: pandas.DataFrame
        undistorted 2D points with columns [primary_key, key_cid, key_x, key_y]
    """

    if n_trials is None:
        n_trials = ransac_n_trials(success_prob, inlier_ratio, sample_size=2)
    assert n_trials >= 1, n_trials

    cids = sorted(df[key_cid].unique())
    all_pairs = list(itertools.combinations(cids, 2))

    if n_trials >= len(all_pairs):
        pairs = all_pairs
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(all_pairs), size=n_trials, replace=False)
        pairs = [all_pairs[i] for i in idx]

    return _triangulate_pairs(df, P, pairs, nproc=nproc, reproj_th=reproj_th, re_triangulate=re_triangulate,
                               bundle_adjustment=bundle_adjustment, show_pbar=show_pbar, primary_key=primary_key,
                               key_cid=key_cid, key_x=key_x, key_y=key_y, key_w=key_w, distorted=distorted,
                               camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)


def main_undistort():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib', type=str, help='Calibration file (e.g., calib.json)')
    parser.add_argument('pt2d_distorted', type=str, help='2D points file (e.g., keypoint2d_allframe.csv)')
    parser.add_argument('output', type=str, help='Output filename (e.g., keypoint2d_allframe_undistorted.csv)')
    args = parser.parse_args()

    # Load calib
    K, D, R, T, _ = pycalib.util.load_calib(args.calib)
    print(f'Found {len(K)} cameras from {args.calib}')

    # Load 2D points
    df = pd.read_csv(args.pt2d_distorted).dropna()
    print(f'Found {len(df)} 2D points from {args.pt2d_distorted}')

    # Undistort for each camera
    for c in df['camera'].unique():
        undistort_inplace(df, c, K[c], D[c])

    # Save
    df.to_csv(args.output, index=False, float_format="%.16f")


def main_triangulate():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib', type=str, help='Calibration file (e.g., calib.json)')
    parser.add_argument('pt2d_undistorted', type=str, help='2D points file (e.g., keypoint2d_allframe_undistorted.csv)')
    parser.add_argument('cam1', type=int, help='1st camera for triangulation')
    parser.add_argument('cam2', type=int, help='2nd camera for triangulation')
    parser.add_argument('output', type=str, help='Output csv file')
    parser.add_argument('-r', '--reproj_th', type=float, default=5, help='Max reprojection error to accept')
    args = parser.parse_args()

    # Load calib
    K, D, R, T, _ = pycalib.util.load_calib(args.calib)
    print(f'Found {len(K)} cameras from {args.calib}')
    P = []
    for k, r, t in zip(K, R, T):
        p = k @ np.hstack([r, t])
        P.append(p)
    P = np.array(P)

    # Load 2D points
    df = pd.read_csv(args.pt2d_undistorted).dropna()
    print(f'Found {len(df)} 2D points from {args.pt2d_undistorted}')
    if 'index' in df:
        df = df.drop(columns='index')

    d = triangulate(df, P, args.cam1, args.cam2, reproj_th=args.reproj_th)

    # Save
    d.to_csv(args.output, index=False)
    print(f'Saved to {args.output}')

def main_merge():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+', required=True, type=str, help='CSV files with inliers and visible columns')
    parser.add_argument('-o', '--output', required=True, type=str, help='output CSV file')
    args = parser.parse_args()
    dfs = [pd.read_csv(f) for f in args.input]
    df = merge(dfs)
    df.to_csv(args.output, index=False, float_format="%.16f")
    print(f'Saved to {args.output}')

def main_triangulate_consensus():
    parser = argparse.ArgumentParser()
    parser.add_argument('calib', type=str, help='Calibration file (e.g., calib.json)')
    parser.add_argument('pt2d_undistorted', type=str, help='2D points file (e.g., keypoint2d_allframe_undistorted.csv)')
    parser.add_argument('output', type=str, help='Output csv file')
    parser.add_argument('-r', '--reproj_th', type=float, default=5, help='Max reprojection error to accept')
    args = parser.parse_args()

    # Load calib
    K, D, R, T, _ = pycalib.util.load_calib(args.calib)
    print(f'Found {len(K)} cameras from {args.calib}')
    P = []
    for k, r, t in zip(K, R, T):
        p = k @ np.hstack([r, t])
        P.append(p)
    P = np.array(P)

    # Load 2D points
    df = pd.read_csv(args.pt2d_undistorted).dropna()
    print(f'Found {len(df)} 2D points from {args.pt2d_undistorted}')
    if 'index' in df:
        df = df.drop(columns='index')

    d = triangulate_consensus(df, P, reproj_th=args.reproj_th)

    # Save
    d.to_csv(args.output, index=False)
    print(f'Saved to {args.output}')

if __name__ == '__main__':
    import argparse
    #main_undistort()
    #main_triangulate()
    #main_merge()
    main_triangulate_consensus()
