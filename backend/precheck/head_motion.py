import numpy as np
from .face_iterator import iterate_faces

def detect_head_motion(video_path, max_faces=100):
    centers = []

    for _, (x, y, w, h) in iterate_faces(video_path, max_faces):
        cx = x + w / 2
        cy = y + h / 2
        centers.append((cx / w, cy / h))

    if len(centers) < 10:
        return True, 0.0

    centers = np.array(centers)
    motion = np.mean(
        np.linalg.norm(np.diff(centers, axis=0), axis=1)
    )

    is_static = motion < 0.02
    return is_static, float(motion)
