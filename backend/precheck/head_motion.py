import numpy as np

def detect_head_motion(face_cache):
    centers = []
    for _, frame, (x, y, w, h) in face_cache:
        frame_h, frame_w = frame.shape[:2]
        centers.append(((x + w/2) / frame_w, (y + h/2) / frame_h))
        
    if len(centers) < 2:
        return True, 0.0
    centers = np.array(centers)
    motion = np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))
    is_static = motion < 0.015
    return is_static, float(motion)