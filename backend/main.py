from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from models.predict_fake import predict_frame, predict_video
import numpy as np
import cv2

app = FastAPI(title="Deepfake Detection API", version="0.1.0")

class Prediction(BaseModel):
    prob: float
    
    
 
@app.post("/predict-frame")
async def predict_frame_endpoint(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prob = predict_frame(img)
    return {"fake_probability": prob}


@app.post("/predict-video")

async def predict_video_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp_video.mp4", "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture("temp_video.mp4")
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        prob = predict_video(frame)
        results.append(prob)

    cap.release()
    return results

@app.post("/predict-websocket")

async def predict_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Websocket client connecet")
    
    contents = websocket.receive_bytes()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prob = predict_video(frame)
    
    await websocket.send_json({"fake_probability": prob})
    