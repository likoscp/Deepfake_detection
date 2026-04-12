import cv2
import numpy as np


def detect_eye_region_temporal(face_cache):

    eye_means = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]

        fw, fh = x2 - x1, y2 - y1
        if fw < 48 or fh < 48:
            continue

        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)

        ey1, ey2 = int(128 * 0.28), int(128 * 0.50)
        eye_band = gray[ey1:ey2, :]

        eye_means.append(float(np.mean(eye_band)))

    if len(eye_means) < 4:
        return False, 0.0

    score = float(np.std(eye_means))
    return score > 6.0, score
