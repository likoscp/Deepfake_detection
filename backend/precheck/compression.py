import cv2
import numpy as np

def detect_compression_artifacts(frames):
    source = [f for f in frames if isinstance(f, np.ndarray) and f.ndim == 3]

    if not source:
        return False, 0.0

    scores = []
    for img in source:
        if not isinstance(img, np.ndarray) or img.ndim != 3:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h, w = gray.shape
        h8 = (h // 8) * 8
        w8 = (w // 8) * 8
        if h8 < 16 or w8 < 16:
            continue
        gray = gray[:h8, :w8]

        border_left  = gray[:, 7::8]
        border_right = gray[:, 8::8]
        min_w = min(border_left.shape[1], border_right.shape[1])
        h_diff = np.abs(border_left[:, :min_w] - border_right[:, :min_w]).mean()

        border_top    = gray[7::8, :]
        border_bottom = gray[8::8, :]
        min_h = min(border_top.shape[0], border_bottom.shape[0])
        v_diff = np.abs(border_top[:min_h, :] - border_bottom[:min_h, :]).mean()

        inner_h = np.abs(gray[:, 6::8][:, :min_w] - gray[:, 7::8][:, :min_w]).mean()
        inner_v = np.abs(gray[6::8, :][:min_h, :] - gray[7::8, :][:min_h, :]).mean()

        h_ratio = h_diff / (inner_h + 1e-6)
        v_ratio = v_diff / (inner_v + 1e-6)
        scores.append((h_ratio + v_ratio) / 2.0)

    if not scores:
        return False, 0.0

    avg = float(np.mean(scores))
    return bool(avg > 1.05), avg