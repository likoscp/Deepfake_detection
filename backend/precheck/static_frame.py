import cv2
import numpy as np

def detect_static_video(video_path, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    prev = None
    diffs = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev is not None:
            diff = np.mean(cv2.absdiff(prev, gray))
            diffs.append(diff)

        prev = gray

    cap.release()

    mean_diff = np.mean(diffs) if diffs else 0
    is_static = mean_diff < 0.5

    return is_static, mean_diff
