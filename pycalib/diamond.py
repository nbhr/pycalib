import cv2
import numpy as np

def detect_charuco_diamond(gray, aruco_dict, square_length, marker_length, target_diamond_ids, *, verbose=0):
    # detect Aruco
    detected_corners, detected_ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)
    if detected_ids is None:
        if verbose>0: print('Aruco markers are not found')
        return None, None
    if len(detected_ids) < 4:
        if verbose>0: print(f'Four Aruco markers are not found. Detected IDs are: {detected_ids.flatten()}')
        return None, None
    
    # detect Aruco diamonds
    detected_diamond_corners, detected_diamond_ids = cv2.aruco.detectCharucoDiamond(gray, detected_corners, detected_ids, square_length / marker_length)
    if detected_diamond_ids is None:
        if verbose>0: print('Diamond markers are not found')
        return None, None

    # check if the detected one is the target
    target_diamond_ids = np.array(target_diamond_ids).reshape((-1, 1, 4)) # detectCharucoDiamond returns (N, 1, 4) array
    ret_corners = []
    ret_ids = []
    for tdi in target_diamond_ids:
        found = False
        for ddc, ddi in zip(detected_diamond_corners, detected_diamond_ids):
            if np.array_equal(tdi, ddi):
                ret_corners.append(ddc)
                ret_ids.append(ddi)
                found = True
                break
        if not found:
            if verbose>0: print(f'Diamond {tdi.flatten()} not found')
            return None, None

    return np.array(ret_corners), np.array(ret_ids)


