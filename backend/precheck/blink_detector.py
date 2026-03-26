import cv2
import numpy as np
from .face_iterator import iterate_faces

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_blinks(video_path, max_frames=30):
    eye_counts = []
    
    for frame, (x, y, w, h) in iterate_faces(video_path, max_frames):
        face = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eyes = EYE_CASCADE.detectMultiScale(gray_face, 1.1, 3)
        eye_counts.append(len(eyes))
    
    if len(eye_counts) == 0:
        return True, 0.0
    
    eye_arr = np.array(eye_counts)
    eyes_found_ratio = float((eye_arr > 0).sum()) / len(eye_arr)
    variance = float(np.std(eye_arr))
    
    is_no_blink = variance < 0.1 and eyes_found_ratio < 0.3
    score = eyes_found_ratio
    return is_no_blink, score