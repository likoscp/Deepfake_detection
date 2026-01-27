import cv2
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import numpy as np

MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_video_file(video_path, max_frames=500, threshold=0.8):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break
        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)

    cap.release()

    if not frames:
        return {"video_score": None, "prediction": "ERROR", "reason": "No frames read"}

    scores = []
    for frame in frames:
        inputs = feature_extractor(images=frame, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)
            scores.append(prob[0, 1].item())

    video_score = float(np.median(scores))
    video_score_clipped = min(video_score, 0.99)
    prediction = "FAKE" if video_score_clipped >= threshold else "REAL"

    return {"video_score": video_score_clipped, "prediction": prediction}
