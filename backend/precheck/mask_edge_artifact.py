import cv2
import numpy as np

def detect_mask_edges(face_cache):
    scores = []
    for _, frame, (x, y, w, h) in face_cache:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        scores.append(np.mean(edges > 0))
    if not scores:
        return False, 0.0
    mean_edge = float(np.mean(scores))
    return mean_edge > 0.09, mean_edge