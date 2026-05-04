
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import torch.nn as nn

from pathlib import Path
_BASE = Path(__file__).resolve().parents[2]
MODEL_PATH = str(_BASE / "cnn_scratch" / "best_model.pth")
YOLO_PATH  = str(_BASE.parent / "yolo26n-face.pt")

device = "cuda" if torch.cuda.is_available() else "cpu"

class DeepfakeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 512),
            nn.Sigmoid(),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        att = self.attention(x).unsqueeze(-1).unsqueeze(-1)
        x = x * att

        x = self.gap(x)
        x = self.classifier(x)
        return x

face_model = YOLO(YOLO_PATH)

model = DeepfakeCNN(num_classes=2)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict_video_file(video_path, max_frames=30, threshold=0.65, max_seconds=30):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    max_by_time = int(fps * max_seconds)
    effective_total = min(total, max_by_time)
    step = max(1, effective_total // max_frames)

    scores = []
    frame_idx = 0

    while frame_idx < effective_total:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            h, w = frame.shape[:2]
            scale = min(640 / w, 640 / h, 1.0)

            if scale < 1.0:
                yolo_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                yolo_frame = frame

            results = face_model(yolo_frame, verbose=False)
            boxes = results[0].boxes

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

                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(face_rgb)
                img = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img)
                    prob = torch.softmax(outputs, dim=1)
                    scores.append(prob[0, 1].item())
    
        frame_idx += 1

    cap.release()

    if not scores:
        return {
            "video_score": None,
            "prediction": "ERROR",
            "reason": "No faces detected"
        }

    video_score = float(np.median(scores))
    video_score = min(video_score, 0.99)

    prediction = "FAKE" if video_score >= threshold else "REAL"

    return {
        "video_score": video_score,
        "prediction": prediction,
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