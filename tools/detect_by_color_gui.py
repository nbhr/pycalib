import cv2
import numpy as np
import scipy as sp
from enum import Enum
import argparse
from detect_by_color import get_silhouette, fit_ellipse

def save_state(filename):
    fg_scale = cv2.getTrackbarPos("fg_scale", "main")
    bg_scale = cv2.getTrackbarPos("bg_scale", "main")
    min_blob = cv2.getTrackbarPos("min_blob", "main")
    max_blob = cv2.getTrackbarPos("max_blob", "main")
    smoothness = cv2.getTrackbarPos("smoothness", "main")
    np.savez_compressed(filename, fg_pix=fg_pix, bg_pix=bg_pix, fg_scale=fg_scale, bg_scale=bg_scale, min_blob=min_blob, max_blob=max_blob, smoothness=smoothness)

class MouseState:
    bx: int = -1
    by: int = -1
    ex: int = -1
    ey: int = -1
    is_dragging: bool = False

class Mode(Enum):
    FG = 1
    BG = 2
    SEG = 3

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Source MKV or MP4 file')
parser.add_argument('output', help='Output NPZ file')
parser.add_argument('-f', '--fg_scale', type=int, choices=range(1, 256), metavar="[1-255]", default=1, help='Scaling factor')
parser.add_argument('-b', '--bg_scale', type=int, choices=range(1, 256), metavar="[1-255]", default=1, help='Scaling factor')
parser.add_argument('-m', '--min_blob', type=int, choices=range(1, 1024), metavar="[1-1023]", default=1, help='Minimum blob size (px)')
parser.add_argument('-M', '--max_blob', type=int, choices=range(1, 1024), metavar="[1-1023]", default=256, help='Maxmum blob size (px)')
parser.add_argument('-s', '--smoothness', type=int, choices=range(0, 1001), metavar="[0-1000]", default=10, help='Smoothness factor (%)')
args = parser.parse_args()

cap = cv2.VideoCapture(args.input)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

main_buf = None
main_seg = None
fg_pix = []
bg_pix = []
fg_tree = None
bg_tree = None

def show_help():
    print(f'\n\n\n')
    print(f'\t,\tprev frame')
    print(f'\t.\tnext frame')
    print(f'\tf\tforeground sampling mode (by mouse)')
    print(f'\tb\tbackground sampling mode (by mouse)')
    print(f'\ts\tsegmentation mode')
    print(f'\tt\toverlay the segmentation result over the video frame')
    print(f'\tw\tsave to {args.output}')
    print(f'\n\n\n')
    print(f'1. Sample foreground colors by mouse')
    print(f'   1. Press `f` to switch to FG mode')
    print(f'   2. Select FG area by the left mouse button')
    print(f'      - Shift + L-mouse de-selects the pixel colors')
    print(f'2. Sample background colors by mouse')
    print(f'   1. Press `b` to switch to BG mode')
    print(f'3. Check segmentation result')
    print(f'   1. Press `s` to switch to SEG mode')
    print(f'   2. Add FG / BG colors by `f` and `b` modes as needed.')
    print(f'4. Check other frames')
    print(f'   1. Press `,` and `.` to go back and forth')
    print(f'5. Save the sampled FG/BG colors')
    print(f'   1. Press `w` to save as `{args.output}`')
    print(f'6. Use `detect_by_color.py` and `{args.output}` to save results.')
    print(f'\n\n\n')

def on_change_frame(val):
    global main_buf
    global redraw

    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, main_buf = cap.read()
    assert main_buf is not None
    cv2.imshow("main", main_buf)
    redraw = True

def on_change_trackbar(val):
    global redraw
    redraw = True

def to_uint32(bgr8x3):
    bgr = bgr8x3.astype(np.uint32)
    return (bgr[:,0] << 16) + (bgr[:,1] << 8) + (bgr[:,2])

def to_uint8x3(bgr32):
    bgr = bgr32.astype(np.uint32)
    return np.dstack([(bgr>>16)&0xff, (bgr>>8)&0xff, bgr&0xff])[0].astype(np.uint8)

def add_pix(curr_pix, new_pix):
    if len(curr_pix) == 0:
        return new_pix
    else:
        a = np.union1d(to_uint32(curr_pix), to_uint32(new_pix))
        return to_uint8x3(a)
        return np.unique(np.concatenate((curr_pix, add_pix)), axis=0)

def add_pix_tree(msg, pix, tree, new_pix):
    print(f'add {msg} {len(pix)} -> ', end='')
    pix = add_pix(pix, new_pix)
    tree = sp.spatial.KDTree(pix)
    print(f'{len(pix)}')
    return pix, tree

def del_pix(curr_pix, new_pix):
    if len(curr_pix) == 0:
        return
    else:
        a = np.setdiff1d(to_uint32(curr_pix), to_uint32(new_pix), assume_unique=True)
        return to_uint8x3(a)

def del_pix_tree(msg, pix, tree, new_pix):
    print(f'del {msg} {len(pix)} -> ', end='')
    pix = del_pix(pix, new_pix)
    tree = sp.spatial.KDTree(pix)
    print(f'{len(pix)}')
    return pix, tree

def on_mouse(event, x, y, flags, params):
    global main_buf
    global mode
    global fg_pix
    global bg_pix
    global fg_tree
    global bg_tree
    global redraw

    mouse_state = params
    match event:
        case cv2.EVENT_LBUTTONDOWN:
            mouse_state.bx = x
            mouse_state.by = y
            mouse_state.is_dragging = True
        case cv2.EVENT_LBUTTONUP:
            mouse_state.ex = x
            mouse_state.ey = y
            mouse_state.is_dragging = False
            if mouse_state.bx > mouse_state.ex:
                mouse_state.bx, mouse_state.ex = mouse_state.ex, mouse_state.bx
            if mouse_state.by > mouse_state.ey:
                mouse_state.by, mouse_state.ey = mouse_state.ey, mouse_state.by

            # unique pixel colors
            roi = main_buf[mouse_state.by:mouse_state.ey+1,mouse_state.bx:mouse_state.ex+1,:]
            roi = roi.reshape((-1, 3))
            roi = np.unique(roi, axis=0)

            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                if mode == Mode.FG:
                    fg_pix, fg_tree = del_pix_tree('fg', fg_pix, fg_tree, roi)
                elif mode == Mode.BG:
                    bg_pix, bg_tree = del_pix_tree('bg', bg_pix, bg_tree, roi)
            else:
                if mode == Mode.FG:
                    fg_pix, fg_tree = add_pix_tree('fg', fg_pix, fg_tree, roi)
                elif mode == Mode.BG:
                    bg_pix, bg_tree = add_pix_tree('bg', bg_pix, bg_tree, roi)

            cv2.imshow("main", main_buf)
            redraw = True
        case cv2.EVENT_MOUSEMOVE:
            if mouse_state.is_dragging:
                buf = main_buf.copy()
                cv2.rectangle(buf, (mouse_state.bx, mouse_state.by), (x,y), (0,0,255), 2, cv2.LINE_8)
                cv2.imshow("main", buf)


mouse_state = MouseState()
mode = Mode.FG
redraw = True

cv2.namedWindow("main")
cv2.createTrackbar("frame", "main", 0, count-1, on_change_frame)
cv2.createTrackbar("fg_scale", "main", args.fg_scale, 256, on_change_trackbar)
cv2.createTrackbar("bg_scale", "main", args.bg_scale, 256, on_change_trackbar)
cv2.createTrackbar("min_blob", "main", args.min_blob, 1024, on_change_trackbar)
cv2.createTrackbar("max_blob", "main", args.max_blob, 1024, on_change_trackbar)
cv2.createTrackbar("smoothness", "main", args.smoothness, 1000, on_change_trackbar)
cv2.setMouseCallback('main', on_mouse, mouse_state)
on_change_frame(0)
show_help()

while True:
    if redraw:
        if mode == Mode.SEG:
            fg_scale = cv2.getTrackbarPos("fg_scale", "main")
            bg_scale = cv2.getTrackbarPos("bg_scale", "main")
            min_sz = cv2.getTrackbarPos("min_blob", "main")**2
            max_sz = cv2.getTrackbarPos("max_blob", "main")**2
            smoothness = cv2.getTrackbarPos("smoothness", "main") / 100.0
            print('segmentation ... ', end='', flush=True)
            main_seg = get_silhouette(main_buf, fg_tree, bg_tree, fg_scale, bg_scale, smoothness)
            buf = fit_ellipse(main_seg, min_sz=min_sz, max_sz=max_sz, force_convex_hull=True, verbose=True)
            print('done', flush=True)
            cv2.imshow("main", buf)
        redraw = False

    key = cv2.waitKey(10)
    if key == ord('q') or key == 27: # ESC
        print('quit')
        break
    elif key == ord('f'):
        mode = Mode.FG
        print('fg mode')
        cv2.imshow("main", main_buf)
    elif key == ord('b'):
        mode = Mode.BG
        print('bg mode')
        cv2.imshow("main", main_buf)
    elif key == ord('s'):
        if fg_tree is None:
            print('FG colorspace is not specified yet')
        elif bg_tree is None:
            print('BG colorspace is not specified yet')
        else:
            mode = Mode.SEG
            print('segmentation mode')
            redraw = True
    elif key == ord('t'):
        if main_buf is not None and main_seg is not None:
            buf = main_buf.copy()
            buf[main_seg == 0] = buf[main_seg == 0] // 16
            cv2.imshow("main", buf)
    elif key == 83 or key == ord('.'): # right
        curr = cv2.getTrackbarPos("frame", "main")
        if curr < count-1:
            cv2.setTrackbarPos("frame", "main", curr+1)
    elif key == 81 or key == ord(','): # left
        curr = cv2.getTrackbarPos("frame", "main")
        if curr > 0:
            cv2.setTrackbarPos("frame", "main", curr-1)
    elif key == ord('w'):
        print(f'saving to {args.output}')
        save_state(args.output)
    elif key == ord('h') or key == ord('?'):
        show_help()
    elif key>0:
        pass
        #print(f'unknown key {key}')

