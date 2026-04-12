import cv2
import numpy as np


def detect_color_depth(face_cache):
 
    scores = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0 or (x2 - x1) < 32 or (y2 - y1) < 32:
            continue

        face = cv2.resize(face, (64, 64))

        gap_ratios = []
        for ch in cv2.split(face):
            hist = np.bincount(ch.ravel(), minlength=256).astype(np.float32)
            nonzero_idx = np.where(hist > 0)[0]
            if len(nonzero_idx) < 5:
                continue
            start, end = nonzero_idx[0], nonzero_idx[-1]
            span = end - start + 1
            if span < 10:
                continue
            gaps = int(np.sum(hist[start:end + 1] == 0))
            gap_ratios.append(gaps / span)

        if not gap_ratios:
            continue
        scores.append(float(np.mean(gap_ratios)))

    if not scores:
        return False, 0.0

    score = float(np.mean(scores))
    return score > 0.30, score
