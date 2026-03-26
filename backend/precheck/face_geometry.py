import cv2
import numpy as np
from .face_iterator import iterate_faces
import dlib

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def detect_face_warping(video_path, max_frames=50):
    distortions = []
    
    for frame, (x, y, w, h) in iterate_faces(video_path, max_frames=max_frames):
        try:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            fh, fw = frame.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(fw, x + w)
            y2 = min(fh, y + h)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = np.ascontiguousarray(gray)

            ch, cw = gray.shape
            rect = dlib.rectangle(0, 0, cw - 1, ch - 1)
            shape = predictor(gray, rect)
            points = np.array([[p.x, p.y] for p in shape.parts()])

            width  = np.max(points[:, 0]) - np.min(points[:, 0])
            height = np.max(points[:, 1]) - np.min(points[:, 1])

            if height < 1:
                continue  

            ratio = width / height
            distortions.append(abs(ratio - 1.0)) 
            
        except Exception as e:
            continue

    if not distortions:
        return False, 0.0
    
    score = float(np.mean(distortions))
    return score > 0.15, score