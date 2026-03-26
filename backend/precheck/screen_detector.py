import cv2
import numpy as np

def detect_screen_display(video_path, max_frames=30):

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if len(frames) < 2:
        return False, 0.0
    
    scores = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        h, w = magnitude.shape
        
        magnitude[h//2-5:h//2+5, w//2-5:w//2+5] = 0
        
        mid_band = magnitude[h//4:3*h//4, w//4:3*w//4]
        moire_score = float(np.percentile(mid_band, 99)) / (np.mean(magnitude) + 1e-6)
        
        
        brightness_std = float(np.std(gray.astype(np.float32)))
        uniformity_score = 1.0 / (brightness_std + 1e-6) * 100
        
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        pixel_structure = float(np.var(laplacian))
        
        frame_score = moire_score * 0.5 + uniformity_score * 0.3 + (pixel_structure / 1000) * 0.2
        scores.append(frame_score)
    
    final_score = float(np.mean(scores))
    
    is_screen = final_score > 21
    
    return is_screen, final_score


def detect_screen_flicker_pattern(video_path, max_frames=60):

    cap = cv2.VideoCapture(video_path)
    brightness_over_time = []
    
    while len(brightness_over_time) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness_over_time.append(np.mean(gray))
    cap.release()
    
    if len(brightness_over_time) < 10:
        return False, 0.0
    
    b = np.array(brightness_over_time)
    
    fft = np.abs(np.fft.rfft(b - np.mean(b)))
    fps = 25  
    freqs = np.fft.rfftfreq(len(b), d=1.0/fps)
    
    
    mask = (freqs >= 8) & (freqs <= 15)
    flicker_power = float(np.max(fft[mask])) if mask.any() else 0.0
    total_power = float(np.sum(fft)) + 1e-6
    flicker_score = flicker_power / total_power
    
    is_flicker = flicker_score > 0.09
    return is_flicker, flicker_score

def detect_screen_flatness(video_path):
    cap = cv2.VideoCapture(video_path)
    variances = []

    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variances.append(np.var(gray))

    cap.release()

    avg_var = np.mean(variances)

    
    is_flat = avg_var < 500

    return is_flat, avg_var