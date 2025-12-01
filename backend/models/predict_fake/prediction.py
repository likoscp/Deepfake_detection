import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("./models/saved_model")

def predict_frame(frame: np.ndarray) -> float:
    
    return

def predict_video(frame: np.ndarray) -> float:
    
    return

def predict_websocket(frame: np.ndarray) -> float:
    
    return