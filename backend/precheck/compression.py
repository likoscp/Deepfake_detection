import cv2
import numpy as np

def detect_compression_artifacts(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    scores = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)
    cap.release()
    if not scores:
        return False, 0.0
    avg_score = np.mean(scores)
    is_artifacted = avg_score > 433  
    return is_artifacted, float(avg_score)