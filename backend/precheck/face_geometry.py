import cv2
import numpy as np
import dlib

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

NOSE_IDX = list(range(27, 36))
LEYE_IDX = list(range(36, 42))
REYE_IDX = list(range(42, 48))
LJAW_IDX = list(range(0, 8))
RJAW_IDX = list(range(16, 8, -1)) 
LBROW_IDX = list(range(17, 22))
RBROW_IDX = list(range(26, 21, -1))

def _get_landmarks(rgb_crop):
    ch, cw = rgb_crop.shape[:2]
    rect = dlib.rectangle(0, 0, cw - 1, ch - 1)
    shape = predictor(rgb_crop, rect)
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float32)

def _normalize_landmarks(pts):
    ref = pts[NOSE_IDX].mean(axis=0)
    scale = np.linalg.norm(
        pts[LEYE_IDX].mean(0) - pts[REYE_IDX].mean(0)
    ) + 1e-6
    return (pts - ref) / scale

def _asymmetry_score(pts_norm):

    left_jaw  = pts_norm[LJAW_IDX]
    right_jaw = pts_norm[RJAW_IDX].copy()
    right_jaw[:, 0] *= -1  

    left_brow  = pts_norm[LBROW_IDX]
    right_brow = pts_norm[RBROW_IDX].copy()
    right_brow[:, 0] *= -1

    jaw_asym  = float(np.mean(np.linalg.norm(left_jaw - right_jaw, axis=1)))
    brow_asym = float(np.mean(np.linalg.norm(left_brow - right_brow, axis=1)))
    return (jaw_asym + brow_asym) / 2.0

def detect_face_warping(face_cache, threshold=0.08):

    asym_scores = []

    for _, frame, (x, y, w, h) in face_cache:
        try:
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < 48 or crop.shape[1] < 48:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            pts = _get_landmarks(rgb)
            pts_norm = _normalize_landmarks(pts)
            asym_scores.append(_asymmetry_score(pts_norm))
        except Exception:
            continue

    if len(asym_scores) < 3:
        return False, 0.0

    mean_asym = float(np.mean(asym_scores))
    asym_variance = float(np.std(asym_scores))

    score = mean_asym * 0.5 + asym_variance * 2.0

    return score > threshold, score