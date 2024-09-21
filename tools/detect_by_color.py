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
    print(ellipses)
    buf = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for e in ellipses:
        buf = cv2.ellipse(buf, e, (255,0,0), 4)
    return buf

def get_silhouette(img, fg_tree, bg_tree, fg_scale, bg_scale, smoothness):
    fg_dist, fg_idx = fg_tree.query(img.reshape((-1, 3)))
    bg_dist, bg_idx = bg_tree.query(img.reshape((-1, 3)))
    fg_dist = fg_dist.reshape(img.shape[:2]) * fg_scale
    bg_dist = bg_dist.reshape(img.shape[:2]) * bg_scale

    # pix-wise
    #sil = np.zeros(img.shape[:2], dtype=np.uint8)
    #sil[fg_dist < bg_dist] = 255
    #return sil

    g = maxflow.Graph[float]()
    nodeids = g.add_grid_nodes(img.shape[:2])
    g.add_grid_edges(nodeids, smoothness)
    g.add_grid_tedges(nodeids, fg_dist, bg_dist)
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
    parser.add_argument('-s', '--smoothness', type=int, choices=range(0, 256), metavar="[0-255]", default=16, help='Smoothness factor')
    args = parser.parse_args()


    state = np.load(args.state)
    fg_tree = sp.spatial.KDTree(state['fg_pix'])
    bg_tree = sp.spatial.KDTree(state['bg_pix'])
    fg_scale = state['fg_scale']
    bg_scale = state['bg_scale']
    smoothness = state['smoothness']

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
