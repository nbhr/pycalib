import numpy as np
import cv2
import dateutil.parser
from datetime import datetime
from tqdm import tqdm

def str2datetime_gopro(s):
    return datetime.strptime(s + '000', 'oT%y%m%d%H%M%S.%f')

def str2datetime_iso(s):
    return dateutil.parser.isoparse(s) 

def detect_qrcode_gopro(filename, *, begin=0, end=-1, fmt='gopro', verbose_lv=0):
    """ Detect QR code for each frame, and parse it as timestamp

    Use ./data/qrtimecode/gopro+js.html to show the QR code.

    Parameters
    ----------
    filename: str
        source video filename
    begin: int
        frame to start detection
    end: int
        frame to stop detection
    fmt: str
        either 'gopro' or 'iso'
    verbose_lv: int
        verbosity level

    Returns
    -------
    X: ndarray
        frame indices (int)
    Y: ndarray
        detected timestamps (float)
    FPS: float
        framerate
    """

    cap = cv2.VideoCapture(filename)
    qcd = cv2.QRCodeDetector()
    X = []
    Y = []
    
    str2datetime = str2datetime_gopro
    if fmt == 'iso':
        str2datetime = str2datetime_iso

    idx_max = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end < 0:
        end = idx_max
    else:
        end = min(end, idx_max)

    idx = 0

    pbar = tqdm(total=end-begin, disable=(verbose_lv==0))
    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        if begin <= idx:
            ret_qr, decoded_info, points, _ = qcd.detectAndDecodeMulti(frame)
            if ret_qr:
                for s, p in zip(decoded_info, points):
                    if s:
                        ts = str2datetime(s)
                        X.append(idx)
                        Y.append(ts.timestamp())
                        if verbose_lv > 1:
                            print(f'{idx:12}, {ts}')
                pbar.set_description(f'Found in {len(X)} frames')
            pbar.update(1)
        idx += 1

        if idx >= end:
            break
    pbar.close()

    return np.array(X), np.array(Y), cap.get(cv2.CAP_PROP_FPS)


def estimate_offset_only(X_all, Y_all, alpha):
    X = X_all * alpha
    beta = np.mean(Y_all - X_all)
    return beta
    
def estimate_offset(X_all, Y_all, *, fps=0):
    """ Estimate time offsets between cameras from detected timestamps

    Parameters
    ----------
    X_all: list of np.ndarray
        frame indices of each timestamp
    Y_all: list of np.ndarray
        timestamps at frame indices given as X

    Returns
    -------
    offset: float
        frame offset between the cameras
    alpha: float
        slope alpha to map frame index to timestamp
    beta: ndarray
        camera-specific intercept beta to map frame index to timestamp
    rmse: ndarray
        root-mean-square residuals between Y and alpha*X+beta
    """

    if fps>0:
        alpha = 1.0 / fps
        beta = []
        for c, (X, Y) in enumerate(zip(X_all, Y_all)):
            b = estimate_offset_only(X, Y, alpha)
            beta.append(b)
        beta = np.array(beta)
    else:
        N = np.sum([ len(X) for X in X_all ])
        A = np.zeros((N, len(X_all)+1))
        b = np.zeros(N)
        
        curr = 0
        for c, (X, Y) in enumerate(zip(X_all, Y_all)):
            N = len(X)
            assert N == len(Y), f'len(X) != len(Y) for camera {c}'
    
            A[curr:curr+N, 0] = X
            A[curr:curr+N, 1+c] = 1
            b[curr:curr+N] = Y
            curr += N
        
        x = np.linalg.solve(A.T@A, A.T@b.reshape((-1,1)))
        x = x.flatten()
        alpha = x[0]
        beta = x[1:]

    offset = (beta - beta[0]) / alpha
    
    rmse_all = []
    for c, (X, Y) in enumerate(zip(X_all, Y_all)):
        Y_est = alpha * X + beta[c]
        e = np.sqrt(np.mean((Y-Y_est)**2))
        rmse_all.append(e)
        
    return offset, alpha, beta, np.array(rmse_all)
