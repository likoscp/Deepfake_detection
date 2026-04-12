import cv2
import numpy as np
from .deepfake import _compute_lbp


def detect_lbp_entropy(face_cache):

    entropies = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0 or (x2 - x1) < 48 or (y2 - y1) < 48:
            continue

        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.uint8)

        lbp = _compute_lbp(gray)

        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)

        nonzero = hist[hist > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        entropies.append(entropy)

    if not entropies:
        return False, 0.0

    mean_entropy = float(np.mean(entropies))

    score = float(max(0.0, 7.0 - mean_entropy))
    return score > 2.5, score
