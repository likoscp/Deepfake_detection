import cv2
import numpy as np


def detect_halftone_pattern(face_cache):

    scores = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]
        if face.size == 0 or (x2 - x1) < 32 or (y2 - y1) < 32:
            continue

        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)

        fft_shift = np.fft.fftshift(np.fft.fft2(gray))
        magnitude = np.log1p(np.abs(fft_shift))

        h_m, w_m = magnitude.shape
        cy, cx = h_m // 2, w_m // 2

        magnitude[cy - 4:cy + 4, cx - 4:cx + 4] = 0

        y_grid, x_grid = np.ogrid[:h_m, :w_m]
        dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        band = (dist >= 8) & (dist <= 48)
        mid = magnitude[band]

        if mid.size == 0 or mid.mean() < 1e-6:
            continue

        peak_to_mean = float(np.percentile(mid, 99.5) / mid.mean())
        scores.append(peak_to_mean)

    if not scores:
        return False, 0.0

    score = float(np.mean(scores))
    return score > 18.0, score
