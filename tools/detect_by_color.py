import cv2
import numpy as np
import scipy as sp
import argparse
from detect_by_color_gui import get_silhouette
from tqdm import tqdm

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
