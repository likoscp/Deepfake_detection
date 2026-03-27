import cv2
import numpy as np

def detect_temporal_inconsistency(
    video_path, 
    max_frames=50, 
    resize=(224, 224), 
    threshold=0.08
):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            continue

        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        frame = cv2.resize(frame, resize)
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

    cap.release()

    if len(frames) < 2:
        return False, 0.0

    frames_array = np.stack(frames, axis=0)

    diffs = np.abs(np.diff(frames_array, axis=0))
    score = float(np.mean(diffs))

    return score > threshold, score