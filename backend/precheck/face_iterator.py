import cv2
import numpy as np
from ultralytics import YOLO

from pathlib import Path
YOLO_PATH = str(Path(__file__).resolve().parents[1] / "yolo26n-face.pt")
_face_model = None

def _resize_frame(frame, max_dim=640):
    h, w = frame.shape[:2]
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

def _load_phase1_and_faces(video_path, sample_frames=40, max_seconds=5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_needed = int(fps * max_seconds)
    step = max(1, total_needed // sample_frames)
    model = get_face_model()

    frames = []
    sampled_720 = []
    sampled_indices = []

    idx = 0
    while idx < total_needed and len(sampled_720) < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frame_720 = _resize_frame(frame, max_dim=640)
            frames.append(cv2.resize(frame_720, (320, 240)))
            sampled_720.append(frame_720)
            sampled_indices.append(idx)
        idx += 1
    cap.release()

    face_cache = []
    if sampled_720:
        results = model(sampled_720, verbose=False, device=0)
        for frame_idx, frame_720, result in zip(sampled_indices, sampled_720, results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes.xyxy.tolist():
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                if w > 10 and h > 10:
                    face_cache.append((frame_idx, frame_720, (x1, y1, w, h)))
                    break

    return frames, fps, face_cache

def _detect_faces_yolo_frame(frame, model=None):
    if model is None:
        model = get_face_model()
    results = model(frame, verbose=False, device=0)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    faces = []
    for box in boxes.xyxy.tolist():
        x1, y1, x2, y2 = map(int, box)
        w, h = x2 - x1, y2 - y1
        if w > 10 and h > 10:
            faces.append((x1, y1, w, h))
    return faces

def get_face_model():
    global _face_model
    if _face_model is None:
        _face_model = YOLO(YOLO_PATH)
    return _face_model

def detect_no_face(face_cache, min_face_ratio=0.5, total_frames=15):
    if not face_cache:
        return True, 0.0
    ratio = len(face_cache) / max(total_frames, 1)
    return ratio < min_face_ratio, float(ratio)