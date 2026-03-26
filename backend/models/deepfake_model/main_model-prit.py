import cv2
import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np

MODEL_PATH = r"E:\Projects\diploma\Deepfake_detection\backend\models\efficientnet_b4\best_model.pth"

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
        img = transform(frame).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            prob = torch.softmax(outputs, dim=1)
            scores.append(prob[0, 1].item())

    video_score = float(np.median(scores))
    video_score_clipped = min(video_score, 0.99)
    prediction = "FAKE" if video_score_clipped >= threshold else "REAL"

    return {"video_score": video_score_clipped, "prediction": prediction}