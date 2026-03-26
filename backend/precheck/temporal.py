import cv2
import numpy as np

def detect_temporal_inconsistency(video_path, max_frames=50, resize=(224, 224), threshold=1300.0):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None or frame.size == 0:
            continue
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        frame_resized = cv2.resize(frame, resize)
        frames.append(frame_resized.astype(np.float32))
        
        
        if len(frames) == 5:
            test = np.stack(frames, axis=0)
            d = np.diff(test, axis=0).reshape(4, -1)
            if np.linalg.norm(d, axis=1).mean() < 1.0:
                cap.release()
                return True, 0.0  
    
    cap.release()
    
    if len(frames) < 2:
        return False, 0.0
    
    frames_array = np.stack(frames, axis=0)
    diffs = np.diff(frames_array, axis=0).reshape(len(frames)-1, -1)
    diffs = np.linalg.norm(diffs, axis=1)
    score = float(np.mean(diffs))
    
    return score > threshold, score