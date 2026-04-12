import cv2
import numpy as np


def detect_blending_boundary(face_cache):

    scores = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]

        fw, fh = x2 - x1, y2 - y1
        if fw < 64 or fh < 64:
            continue

        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        lap = np.abs(lap)

        border_mask = np.zeros((128, 128), dtype=bool)
        border_mask[:12, :]   = True
        border_mask[-12:, :]  = True
        border_mask[:, :12]   = True
        border_mask[:, -12:]  = True

        cy, cx = 64, 64
        inner_mask = np.zeros((128, 128), dtype=bool)
        inner_mask[26:102, 26:102] = True
        inner_mask[border_mask]    = False

        border_var = float(np.var(lap[border_mask]))
        centre_var = float(np.var(lap[inner_mask]))

        if centre_var < 1.0:
            continue

        ratio = border_var / centre_var
        score = float(max(0.0, ratio - 1.2))
        scores.append(score)

    if not scores:
        return False, 0.0

    score = float(np.mean(scores))
    return score > 0.8, score
