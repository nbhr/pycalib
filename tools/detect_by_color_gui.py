import cv2
import numpy as np
import scipy as sp
from enum import Enum
import maxflow
import argparse

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
    def save_state(filename):
        fg_scale = cv2.getTrackbarPos("fg_scale", "main")
        bg_scale = cv2.getTrackbarPos("bg_scale", "main")
        smoothness = cv2.getTrackbarPos("smoothness", "main")
        np.savez_compressed(filename, fg_pix=fg_pix, bg_pix=bg_pix, fg_scale=fg_scale, bg_scale=bg_scale, smoothness=smoothness)

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
    parser.add_argument('-s', '--smoothness', type=int, choices=range(0, 256), metavar="[0-255]", default=16, help='Smoothness factor')
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

    def del_pix(curr_pix, new_pix):
        if len(curr_pix) == 0:
            return
        else:
            a = np.setdiff1d(to_uint32(curr_pix), to_uint32(new_pix), assume_unique=True)
            return to_uint8x3(a)

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

                # unique pixel colors
                roi = main_buf[mouse_state.by:mouse_state.ey+1,mouse_state.bx:mouse_state.ex+1,:]
                roi = roi.reshape((-1, 3))
                roi = np.unique(roi, axis=0)

                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    if mode == Mode.FG:
                        print(f'del fg {len(fg_pix)} -> ', end='')
                        fg_pix = del_pix(fg_pix, roi)
                        fg_tree = sp.spatial.KDTree(fg_pix)
                        print(f'{len(fg_pix)}')
                    elif mode == Mode.BG:
                        print(f'del bg {len(bg_pix)} -> ', end='')
                        bg_pix = del_pix(bg_pix, roi)
                        bg_tree = sp.spatial.KDTree(bg_pix)
                        print(f'{len(bg_pix)}')
                else:
                    if mode == Mode.FG:
                        print(f'add fg {len(fg_pix)} -> ', end='')
                        fg_pix = add_pix(fg_pix, roi)
                        fg_tree = sp.spatial.KDTree(fg_pix)
                        print(f'{len(fg_pix)}')
                    elif mode == Mode.BG:
                        print(f'add bg {len(fg_pix)} -> ', end='')
                        bg_pix = add_pix(bg_pix, roi)
                        bg_tree = sp.spatial.KDTree(bg_pix)
                        print(f'{len(bg_pix)}')
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
    cv2.createTrackbar("smoothness", "main", args.smoothness, 256, on_change_trackbar)
    cv2.setMouseCallback('main', on_mouse, mouse_state)
    on_change_frame(0)

    while True:
        if redraw:
            if mode == Mode.SEG:
                fg_scale = cv2.getTrackbarPos("fg_scale", "main")
                bg_scale = cv2.getTrackbarPos("bg_scale", "main")
                smoothness = cv2.getTrackbarPos("smoothness", "main")
                print('segmentation ... ', end='', flush=True)
                sil = get_silhouette(main_buf, fg_tree, bg_tree, fg_scale, bg_scale, smoothness)
                print('done', flush=True)
                cv2.imshow("main", sil)
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
        elif key>0:
            pass
            #print(f'unknown key {key}')

