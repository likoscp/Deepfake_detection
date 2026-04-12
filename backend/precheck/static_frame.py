import cv2
import numpy as np

def detect_static_video(frames):
    if len(frames) < 2:
        return False, 0.0
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [np.mean(cv2.absdiff(grays[i], grays[i+1])) 
             for i in range(len(grays)-1)]
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    return mean_diff < 0.30, mean_diff