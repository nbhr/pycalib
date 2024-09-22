import cv2
import os
import sys
import numpy as np
import scipy as sp
import argparse
from tqdm import tqdm
import maxflow

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import pycalib

def fit_ellipse(gray, *, min_sz=0, max_sz=-1, force_convex_hull=True, verbose=False):
    ellipses = pycalib.fit_ellipse(gray, min_sz=min_sz, max_sz=max_sz, force_convex_hull=force_convex_hull)
    buf = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for e, c in ellipses:
        buf = cv2.ellipse(buf, e, (255,0,0), 4)
        for p in c:
            x, y = p[0]
            buf = cv2.rectangle(buf, (x-1, y-1), (x+1, y+1), (0, 255, 0), thickness=-1)
    return buf

def diff_to_cost(diff, scale):
    assert diff.shape[2] == 2
    return (255 - np.linalg.norm(diff, axis=2) / np.sqrt(2)) * scale

def get_silhouette(img, fg_tree, bg_tree, fg_scale, bg_scale, smoothness):
    fg_dist, fg_idx = fg_tree.query(img.reshape((-1, 3)))
    bg_dist, bg_idx = bg_tree.query(img.reshape((-1, 3)))
    fg_dist = fg_dist.reshape(img.shape[:2]) * bg_scale
    bg_dist = bg_dist.reshape(img.shape[:2]) * fg_scale

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,1:].astype(float)
    b_dist = diff_to_cost(np.roll(lab, -1, axis=0) - lab, smoothness)
    r_dist = diff_to_cost(np.roll(lab, -1, axis=1) - lab, smoothness)
    br_dist = diff_to_cost(np.roll(lab, (-1, -1), axis=(0,1)) - lab, smoothness)
    ur_dist = diff_to_cost(np.roll(lab, (1, -1), axis=(0,1)) - lab, smoothness)

    # pix-wise
    #sil = np.zeros(img.shape[:2], dtype=np.uint8)
    #sil[fg_dist < bg_dist] = 255
    #return sil

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(img.shape[:2])
    #g.add_grid_edges(nodeids, smoothness, symmetric=True) # fixme: different weights for each pixel
    # see https://github.com/pmneila/PyMaxflow/issues/49
    g.add_grid_edges(nodeids, weights=b_dist, structure=np.array([[0,0,0],[0,0,0],[0,1,0]]), symmetric=True)
    g.add_grid_edges(nodeids, weights=r_dist, structure=np.array([[0,0,0],[0,0,1],[0,0,0]]), symmetric=True)
    g.add_grid_edges(nodeids, weights=br_dist, structure=np.array([[0,0,0],[0,0,0],[0,0,1]]), symmetric=True)
    g.add_grid_edges(nodeids, weights=ur_dist, structure=np.array([[0,0,1],[0,0,0],[0,0,0]]), symmetric=True)
    g.add_grid_tedges(nodeids, fg_dist, bg_dist) # fg == sink:1, bg == source:0
    g.maxflow()
    sil = g.get_grid_segments(nodeids)
    sil = sil.astype(np.uint8)*255

    return sil

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Source MKV or MP4 file')
    parser.add_argument('state', help='State file')
    parser.add_argument('output_fmt', help='Output file format string (e.g., foo_%%08d.png)')
    parser.add_argument('-d', '--debug', action='store_true', help='Show debug window')
    parser.add_argument('-f', '--fg_scale', type=int, choices=range(1, 256), metavar="[1-255]", default=1, help='Scaling factor')
    parser.add_argument('-b', '--bg_scale', type=int, choices=range(1, 256), metavar="[1-255]", default=1, help='Scaling factor')
    parser.add_argument('-m', '--min_blob', type=int, choices=range(1, 1024), metavar="[1-1023]", default=1, help='Minimum blob size (px)')
    parser.add_argument('-M', '--max_blob', type=int, choices=range(1, 1024), metavar="[1-1023]", default=256, help='Maxmum blob size (px)')
    parser.add_argument('-s', '--smoothness', type=int, choices=range(0, 1001), metavar="[0-1000]", default=10, help='Smoothness factor (%)')
    args = parser.parse_args()


    state = np.load(args.state)
    fg_tree = sp.spatial.KDTree(state['fg_pix'])
    bg_tree = sp.spatial.KDTree(state['bg_pix'])
    fg_scale = state['fg_scale']
    bg_scale = state['bg_scale']
    min_sz = state['min_blob']**2
    max_sz = state['max_blob']**2
    smoothness = state['smoothness'] / 100.0

    if args.debug:
        cv2.namedWindow("debug")

    cap = cv2.VideoCapture(args.input)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for curr in tqdm(range(count)):
        ret, img = cap.read()
        assert img is not None
        sil = get_silhouette(img, fg_tree, bg_tree, fg_scale, bg_scale, smoothness)
        cv2.imwrite(args.output_fmt % curr, sil)
        curr = curr + 1

        if args.debug:
            img[sil == 0] = img[sil == 0] // 16
            cv2.imshow("debug", img)
            cv2.waitKey(10)
