import cv2
import numpy as np


def detect_temporal_texture(face_cache):

    if len(face_cache) < 5:
        return False, 0.0

    histograms = []

    for _, frame, (x, y, w, h) in face_cache:
        x1 = max(0, x);          y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0 or (x2 - x1) < 32 or (y2 - y1) < 32:
            continue

        face_resized = cv2.resize(face, (64, 64))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist /= (hist.sum() + 1e-6)
        histograms.append(hist)

    if len(histograms) < 5:
        return False, 0.0

    hist_arr = np.array(histograms)
    score = float(np.mean(np.std(hist_arr, axis=0)))

    return score > 0.015, score
