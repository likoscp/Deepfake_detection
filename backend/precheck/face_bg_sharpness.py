import cv2
import numpy as np

def detect_face_bg_sharpness(face_cache):

    scores = []

    for _, frame, (x, y, w, h) in face_cache:
        if w < 48 or h < 48:
            continue

        fx1, fy1 = max(0, x), max(0, y)
        fx2, fy2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[fy1:fy2, fx1:fx2]
        if face.size == 0:
            continue

        margin = int(max(w, h) * 0.5)
        bx1 = max(0, x - margin)
        by1 = max(0, y - margin)
        bx2 = min(frame.shape[1], x + w + margin)
        by2 = min(frame.shape[0], y + h + margin)

        bg_frame = frame[by1:by2, bx1:bx2].copy()
        fy_rel = fy1 - by1
        fx_rel = fx1 - bx1
        fh_clip = fy2 - fy1
        fw_clip = fx2 - fx1
        bg_frame[fy_rel:fy_rel + fh_clip, fx_rel:fx_rel + fw_clip] = 0

        face_gray = cv2.cvtColor(cv2.resize(face, (64, 64)), cv2.COLOR_BGR2GRAY)
        face_lap = cv2.Laplacian(face_gray.astype(np.float32), cv2.CV_32F)
        face_var = float(np.var(face_lap))

        bg_mask = bg_frame.sum(axis=2) > 0
        if int(bg_mask.sum()) < 500:
            continue

        bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_lap = cv2.Laplacian(bg_gray, cv2.CV_32F)
        bg_var = float(np.var(bg_lap[bg_mask]))

        if bg_var < 2.0 or face_var < 2.0:
            continue

        ratio = face_var / bg_var
        log_r = float(np.log(ratio))
        deviation = max(0.0, abs(log_r - 0.18) - 0.50)
        scores.append(deviation)

    if not scores:
        return False, 0.0

    score = float(np.mean(scores))
    return score > 0.35, score
