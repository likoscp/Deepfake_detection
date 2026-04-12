import cv2
import numpy as np


def detect_specular_consistency(face_cache):

    scores = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]

        fw, fh = x2 - x1, y2 - y1
        if fw < 64 or fh < 64:
            continue

        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        ey1, ey2 = int(128 * 0.28), int(128 * 0.50)
        lx1, lx2 = int(128 * 0.12), int(128 * 0.46)
        rx1, rx2 = int(128 * 0.54), int(128 * 0.88)

        left_eye = gray[ey1:ey2, lx1:lx2].astype(np.float32)
        right_eye = gray[ey1:ey2, rx1:rx2].astype(np.float32)

        if left_eye.size < 50 or right_eye.size < 50:
            continue

        l_thresh = float(np.percentile(left_eye, 90))
        r_thresh = float(np.percentile(right_eye, 90))

        if l_thresh < 10 and r_thresh < 10:
            continue

        left_hl = (left_eye > l_thresh).astype(np.float32)
        right_hl = (right_eye > r_thresh).astype(np.float32)

        right_hl_m = np.fliplr(cv2.resize(right_hl, (left_hl.shape[1], left_hl.shape[0])))

        diff = float(np.mean(np.abs(left_hl - right_hl_m)))
        scores.append(diff)

    if not scores:
        return False, 0.0

    score = float(np.mean(scores))
    return score > 0.42, score
