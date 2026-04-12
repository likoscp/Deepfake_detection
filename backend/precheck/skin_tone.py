import cv2
import numpy as np
def detect_skin_tone_mismatch(face_cache):
    diffs = []
    for _, frame, (x, y, w, h) in face_cache:
        face = frame[y:y+h, x:x+w]
        
        neck_y1 = min(frame.shape[0], y + h)
        neck_y2 = min(frame.shape[0], y + int(h * 1.3))
        neck = frame[neck_y1:neck_y2, x:x+w]
        
        if face.size == 0 or neck.shape[0] < 15 or neck.shape[1] < 15:
            continue
            
        face_ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        neck_ycrcb = cv2.cvtColor(neck, cv2.COLOR_BGR2YCrCb)
        
        face_mask = cv2.inRange(face_ycrcb[:,:,1:], np.array([135,85]), np.array([180,135]))
        neck_mask = cv2.inRange(neck_ycrcb[:,:,1:], np.array([135,85]), np.array([180,135]))
        
        face_pixels = face_mask.size
        neck_pixels = neck_mask.size
        
        if np.count_nonzero(face_mask) < face_pixels * 0.15:
            continue
        if np.count_nonzero(neck_mask) < neck_pixels * 0.15:
            continue
            
        face_skin = face_ycrcb[:,:,1:][face_mask > 0]
        neck_skin = neck_ycrcb[:,:,1:][neck_mask > 0]
        
        diff = np.linalg.norm(
            np.median(face_skin, axis=0) - np.median(neck_skin, axis=0)
        )
        diffs.append(diff)
    
    if len(diffs) < 3:
        return False, 0.0
        
    avg_diff = float(np.median(diffs))
    return avg_diff > 12.0, avg_diff