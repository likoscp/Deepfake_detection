import cv2
import numpy as np
from mediapipe.python.solutions.face_mesh import FaceMesh

_face_mesh = FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def _ear(landmarks, eye_idx, iw, ih):
    pts = np.array([[landmarks[i].x * iw, landmarks[i].y * ih] for i in eye_idx], dtype=np.float32)
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C + 1e-6)

def detect_blinks(face_cache, ear_threshold=0.22):
    if len(face_cache) < 5:
        return False, 0.5

    ear_values = []

    for _, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 48 or crop.shape[1] < 48:
            continue

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = _face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            continue

        lm = result.multi_face_landmarks[0].landmark
        ih, iw = crop.shape[:2]

        left_ear  = _ear(lm, _LEFT_EYE, iw, ih)
        right_ear = _ear(lm, _RIGHT_EYE, iw, ih)
        ear_values.append((left_ear + right_ear) / 2.0)

    if len(ear_values) < 5:
        return False, 0.5

    arr = np.array(ear_values)
    closed = (arr < ear_threshold).astype(int)

    blink_count = 0
    i = 1
    while i < len(closed):
        if closed[i - 1] == 0 and closed[i] == 1:
            j = i
            while j < len(closed) and closed[j] == 1:
                j += 1
            if (j - i) <= 4 and j < len(closed):
                blink_count += 1
            i = j + 1
        else:
            i += 1

    has_blink = blink_count >= 1
    no_blink_score = 1.0 if blink_count == 0 else max(0.0, 1.0 - blink_count * 0.4)

    return not has_blink, no_blink_score
