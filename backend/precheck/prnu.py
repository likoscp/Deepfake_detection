import cv2
import numpy as np


def detect_prnu_inconsistency(face_cache):

    face_residuals = []
    bg_residuals = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

        fw, fh = x2 - x1, y2 - y1
        if fw < 48 or fh < 48:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        residual = gray - blurred 

        face_res = residual[y1:y2, x1:x2]
        face_res = cv2.resize(face_res, (32, 32))
        face_residuals.append(face_res)

        margin = int(max(fw, fh) * 0.5)
        bx1 = max(0, x - margin)
        by1 = max(0, y - margin)
        bx2 = min(frame.shape[1], x + w + margin)
        by2 = min(frame.shape[0], y + h + margin)

        bg_patch = residual[by1:by2, bx1:bx2].copy()
        fy_rel = y1 - by1
        fx_rel = x1 - bx1
        bg_patch[fy_rel:fy_rel + fh, fx_rel:fx_rel + fw] = 0.0

        bg_mask = (bg_patch != 0.0)
        if int(bg_mask.sum()) < 200:
            continue

        bh, bw = bg_patch.shape
        best_sum = -1
        best_crop = None
        for cy_start in [0, bh // 4, bh // 2]:
            for cx_start in [0, bw // 4, bw // 2]:
                cy_end = cy_start + 32
                cx_end = cx_start + 32
                if cy_end > bh or cx_end > bw:
                    continue
                crop = bg_patch[cy_start:cy_end, cx_start:cx_end]
                s = float(np.sum(crop != 0))
                if s > best_sum:
                    best_sum = s
                    best_crop = crop

        if best_crop is None or best_sum < 100:
            continue

        bg_residuals.append(best_crop.astype(np.float32))

    n = min(len(face_residuals), len(bg_residuals))
    if n < 4:
        return False, 0.0

    face_fp = np.mean(face_residuals[:n], axis=0)
    bg_fp   = np.mean(bg_residuals[:n],   axis=0)

    face_flat = face_fp.ravel()
    bg_flat   = bg_fp.ravel()

    face_std = float(np.std(face_flat))
    bg_std   = float(np.std(bg_flat))

    if face_std < 1e-4 or bg_std < 1e-4:
        return False, 0.0

    face_n = (face_flat - face_flat.mean()) / face_std
    bg_n   = (bg_flat   - bg_flat.mean())   / bg_std

    corr = float(np.dot(face_n, bg_n) / len(face_n))

    score = float(max(0.0, 0.5 - corr))
    return score > 0.40, score
