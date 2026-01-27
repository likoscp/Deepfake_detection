import cv2
import numpy as np
from face_iterator import iterate_faces

def detect_mask_edges(video_path, max_frames=50):
    scores = []

    for frame, (x, y, w, h) in iterate_faces(video_path, max_frames):
        face = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        scores.append(edge_density)

    if not scores:
        return False, 0.0

    mean_edge = float(np.mean(scores))
    return mean_edge > 0.12, mean_edge
