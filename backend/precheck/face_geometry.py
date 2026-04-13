import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh

_face_mesh = FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5
)

LEFT_JAW  = [152, 148, 176, 149, 150, 136, 172, 58, 132, 93]
RIGHT_JAW = [152, 377, 400, 378, 379, 365, 397, 288, 361, 323]

LEFT_BROW  = [70, 63, 105, 66, 107]
RIGHT_BROW = [300, 293, 334, 296, 336]

LEFT_EYE_CORNER  = 33
RIGHT_EYE_CORNER = 263
NOSE_TIP         = 1


def _get_landmarks(rgb_crop):
    result = _face_mesh.process(rgb_crop)
    if not result.multi_face_landmarks:
        return None
    ih, iw = rgb_crop.shape[:2]
    lm = result.multi_face_landmarks[0].landmark
    return np.array([[l.x * iw, l.y * ih] for l in lm], dtype=np.float32)


def _normalize(pts):
    ref   = pts[NOSE_TIP]
    scale = np.linalg.norm(pts[LEFT_EYE_CORNER] - pts[RIGHT_EYE_CORNER]) + 1e-6
    return (pts - ref) / scale


def _asymmetry_score(pts_norm):
    lj = pts_norm[LEFT_JAW]
    rj = pts_norm[RIGHT_JAW].copy()
    rj[:, 0] *= -1

    lb = pts_norm[LEFT_BROW]
    rb = pts_norm[RIGHT_BROW].copy()
    rb[:, 0] *= -1

    jaw_asym  = float(np.mean(np.linalg.norm(lj - rj, axis=1)))
    brow_asym = float(np.mean(np.linalg.norm(lb - rb, axis=1)))
    return (jaw_asym + brow_asym) / 2.0


def detect_face_warping(face_cache, threshold=0.08):
    asym_scores = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 48 or crop.shape[1] < 48:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pts = _get_landmarks(rgb)
        if pts is None:
            continue

        pts_norm = _normalize(pts)
        asym_scores.append(_asymmetry_score(pts_norm))

    if len(asym_scores) < 3:
        return False, 0.0

    mean_asym     = float(np.mean(asym_scores))
    asym_variance = float(np.std(asym_scores))
    score         = mean_asym * 0.5 + asym_variance * 2.0

    return score > threshold, score