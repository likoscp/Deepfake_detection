import cv2
import torch
import timm
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

MODEL_PATH = r"E:\Projects\diploma\Deepfake_detection\backend\models\efficientnet_b4\best_model.pth"
YOLO_PATH  = r"e:\Projects\diploma\доп скрипты\yolo26n-face.pt"

face_model = YOLO(YOLO_PATH)
model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def predict_video_file(video_path, max_frames=100, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    step  = max(1, total // max_frames)

    scores = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            results = face_model(frame, verbose=False)
            boxes = results[0].boxes

            if boxes is not None and len(boxes) > 0:
                largest = max(boxes.xyxy.tolist(),
                              key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                x1, y1, x2, y2 = map(int, largest)

                w, h = x2-x1, y2-y1
                x1 = max(0, x1 - int(w*0.2))
                y1 = max(0, y1 - int(h*0.2))
                x2 = min(frame.shape[1], x2 + int(w*0.2))
                y2 = min(frame.shape[0], y2 + int(h*0.2))

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    frame_idx += 1
                    continue

                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                pil_img  = Image.fromarray(face_rgb)

                img = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(img)
                    prob = torch.softmax(outputs, dim=1)
                    scores.append(prob[0, 1].item())

        frame_idx += 1

    cap.release()

    if not scores:
        return {"video_score": None, "prediction": "ERROR", "reason": "No faces detected"}

    video_score = float(np.median(scores))
    video_score = min(video_score, 0.99)
    prediction  = "FAKE" if video_score >= threshold else "REAL"

    return {"video_score": video_score, "prediction": prediction}