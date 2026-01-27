import cv2
from .face_iterator import iterate_faces
import numpy as np

def detect_skin_tone_mismatch(video_path, max_frames=80):
    diffs = []

    for frame, (x, y, w, h) in iterate_faces(video_path, max_frames):

        face = frame[y:y+h, x:x+w]

        neck_y1 = min(frame.shape[0], y + h)
        neck_y2 = min(frame.shape[0], y + int(h * 1.3))  
        neck = frame[neck_y1:neck_y2, x:x+w]

        if face.size == 0 or neck.size == 0:
            continue

        face_ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        neck_ycrcb = cv2.cvtColor(neck, cv2.COLOR_BGR2YCrCb)

        face_mask = cv2.inRange(face_ycrcb[:, :, 1:], np.array([135, 85]), np.array([180, 135]))
        neck_mask = cv2.inRange(neck_ycrcb[:, :, 1:], np.array([135, 85]), np.array([180, 135]))

        if np.count_nonzero(face_mask) == 0 or np.count_nonzero(neck_mask) == 0:
            continue

        face_skin = face_ycrcb[:, :, 1:][face_mask > 0]
        neck_skin = neck_ycrcb[:, :, 1:][neck_mask > 0]

        face_median = np.median(face_skin, axis=0)
        neck_median = np.median(neck_skin, axis=0)

        diff = np.linalg.norm(face_median - neck_median)
        diffs.append(diff)

    avg_diff = float(np.median(diffs)) if diffs else 0.0
    is_mismatch = avg_diff > 15.0  

    return is_mismatch, avg_diff
