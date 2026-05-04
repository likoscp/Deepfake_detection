import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from ultralytics import YOLO
import torch.nn as nn

from pathlib import Path
_BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = str(_BASE / "efficientnet_b4_v2" / "best_model.pth")
YOLO_PATH  = str(_BASE.parent / "yolo26n-face.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_model():
    model = efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 2),
    )
    return model


face_model = YOLO(YOLO_PATH)

model = build_model()
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict_video_file(video_path, max_frames=30, threshold=0.65, max_seconds=30):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25

    max_by_time   = int(fps * max_seconds)
    effective_total = min(total, max_by_time)
    step = max(1, effective_total // max_frames)

    scores    = []
    frame_idx = 0

    while frame_idx < effective_total:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            scale = min(640 / w, 640 / h, 1.0)
            yolo_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame

            results = face_model(yolo_frame, verbose=False)
            boxes   = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                largest = max(
                    boxes.xyxy.tolist(),
                    key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
                )
                x1, y1, x2, y2 = map(int, largest)
                w2, h2 = x2 - x1, y2 - y1
                x1 = max(0, x1 - int(w2 * 0.2))
                y1 = max(0, y1 - int(h2 * 0.2))
                x2 = min(yolo_frame.shape[1], x2 + int(w2 * 0.2))
                y2 = min(yolo_frame.shape[0], y2 + int(h2 * 0.2))

                face_crop = yolo_frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    frame_idx += 1
                    continue

                face_crop = cv2.resize(face_crop, (256, 256))
                face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img   = Image.fromarray(face_rgb)

                img = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img)
                    prob    = torch.softmax(outputs, dim=1)
                    scores.append(prob[0, 1].item())

        frame_idx += 1

    cap.release()

    if not scores:
        return {
            "video_score": None,
            "prediction":  "ERROR",
            "reason":      "No faces detected"
        }

    video_score = float(np.median(scores))
    video_score = min(video_score, 0.99)
    prediction  = "FAKE" if video_score >= threshold else "REAL"

    return {
        "video_score":     video_score,
        "prediction":      prediction,
        "frames_analyzed": len(scores)
    }


def predict_from_face_cache(face_cache, threshold=0.65):

    scores = []
    for _, frame_640, (x1, y1, w, h) in face_cache:
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_640.shape[1], x1 + w + pad_x)
        cy2 = min(frame_640.shape[0], y1 + h + pad_y)

        face_crop = frame_640[cy1:cy2, cx1:cx2]
        if face_crop.size == 0:
            continue

        face_crop = cv2.resize(face_crop, (256, 256))
        face_rgb  = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img   = Image.fromarray(face_rgb)

        img = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            prob    = torch.softmax(outputs, dim=1)
            scores.append(prob[0, 1].item())

    if not scores:
        return {
            "video_score": None,
            "prediction":  "ERROR",
            "reason":      "No faces in cache"
        }

    video_score = float(np.median(scores))
    video_score = min(video_score, 0.99)
    prediction  = "FAKE" if video_score >= threshold else "REAL"

    return {
        "video_score":     video_score,
        "prediction":      prediction,
        "frames_analyzed": len(scores)
    }