import cv2
import numpy as np

def detect_color_inconsistency(face_cache):
    scores = []
    for _, frame, (x, y, w, h) in face_cache:
        face = frame[y:y+h, x:x+w]
        if face.size == 0: continue
        
        pad = int(w * 0.3)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        surround = frame[y1:y2, x1:x2].copy()

        surround[y-y1:y-y1+h, x-x1:x-x1+w] = 0
        
        if surround[surround.sum(axis=2) > 0].shape[0] < 100:
            continue
            
        face_ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb).astype(np.float32)
        surr_pixels = surround[surround.sum(axis=2) > 0]
        if len(surr_pixels) < 100: continue
        surr_ycrcb = cv2.cvtColor(
            surr_pixels.reshape(1, -1, 3), cv2.COLOR_BGR2YCrCb
        ).reshape(-1, 3).astype(np.float32)
        
        face_cb = face_ycrcb[:,:,1].mean()
        face_cr = face_ycrcb[:,:,2].mean()
        surr_cb = surr_ycrcb[:,1].mean()
        surr_cr = surr_ycrcb[:,2].mean()
        
        diff = np.sqrt((face_cb - surr_cb)**2 + (face_cr - surr_cr)**2)
        scores.append(float(diff))
    
    if len(scores) < 3:
        return False, 0.0
    score = float(np.mean(scores))
    return bool(score > 8.0), score

def detect_noise_inconsistency(face_cache):
    scores = []
    for _, frame, (x, y, w, h) in face_cache:
        face = frame[y:y+h, x:x+w]
        if face.size == 0 or w < 64 or h < 64: continue
        
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        denoised = cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
        noise = np.abs(gray - denoised)
        noise_level = float(np.std(noise))
        
        pad = int(w * 0.3)
        x1, y1 = max(0, x-pad), max(0, y-pad)
        x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)
        bg = frame[y1:y2, x1:x2]
        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg_denoised = cv2.medianBlur(bg_gray.astype(np.uint8), 3).astype(np.float32)
        bg_noise = float(np.std(np.abs(bg_gray - bg_denoised)))
        
        if bg_noise < 0.5: continue
        ratio = noise_level / (bg_noise + 1e-6)
        scores.append(ratio)
    
    if len(scores) < 3:
        return False, 0.0
    score = float(np.mean(scores))
    return bool(score < 0.65), float(1.0 - score)