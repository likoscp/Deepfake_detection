import cv2
import numpy as np
from ultralytics import YOLO

YOLO_PATH = r"e:\Projects\diploma\доп скрипты\yolo26n-face.pt"


_face_model = None

def get_face_model():
    global _face_model
    if _face_model is None:
        _face_model = YOLO(YOLO_PATH)
    return _face_model


def _detect_faces_yolo(frame):
    
    model = get_face_model()
    results = model(frame, verbose=False, device=0)
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return []

    faces = []
    for box in boxes.xyxy.tolist():
        x1, y1, x2, y2 = map(int, box)
        w = x2 - x1
        h = y2 - y1
        if w > 10 and h > 10:  
            faces.append((x1, y1, w, h))

    return faces


def iterate_faces(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while frame_id < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        faces = _detect_faces_yolo(frame)

        if len(faces) == 0:
            frame_id += 1
            continue

        
        x, y, w, h = faces[0]

        yield frame, (x, y, w, h)

        frame_id += 1

    cap.release()

def detect_no_face(video_path, max_frames=30, min_face_ratio=0.5):
    cap = cv2.VideoCapture(video_path)
    frames_checked = 0
    frames_with_face = 0

    while frames_checked < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        faces = _detect_faces_yolo(frame)
        if len(faces) > 0:
            frames_with_face += 1

        frames_checked += 1

    cap.release()

    if frames_checked == 0:
        return True, 0.0

    face_ratio = frames_with_face / frames_checked
    no_face = face_ratio < min_face_ratio

    return no_face, face_ratio