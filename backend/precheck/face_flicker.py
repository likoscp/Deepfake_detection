import cv2
import numpy as np


def detect_face_flicker(face_cache):

    if len(face_cache) < 6:
        return False, 0.0

    face_lum = []
    bg_lum   = []

    for _, frame, (x, y, w, h) in face_cache:
        x1 = max(0, x);          y1 = max(0, y)
        x2 = min(frame.shape[1], x + w)
        y2 = min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        pad = max(int(min(w, h) * 0.5), 20)
        bx1 = max(0, x1 - pad);  by1 = max(0, y1 - pad)
        bx2 = min(frame.shape[1], x2 + pad)
        by2 = min(frame.shape[0], y2 + pad)

        region = frame[by1:by2, bx1:bx2].copy()
        ry1, rx1 = y1 - by1, x1 - bx1
        ry2, rx2 = y2 - by1, x2 - bx1
        region[ry1:ry2, rx1:rx2] = 0

        bg_pixels = region[region.sum(axis=2) > 0]
        if len(bg_pixels) < 100:
            continue

        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_lum.append(float(face_gray.mean()))
        bg_lum.append(float(bg_pixels.mean()))

    if len(face_lum) < 6:
        return False, 0.0

    fa = np.diff(np.array(face_lum))
    ba = np.diff(np.array(bg_lum))

    if fa.std() < 0.3 or ba.std() < 0.3:
        return False, 0.0

    corr  = float(np.corrcoef(fa, ba)[0, 1])
    score = float(max(0.0, 1.0 - corr))

    return score > 0.8, score
