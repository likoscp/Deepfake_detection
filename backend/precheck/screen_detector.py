import cv2
import numpy as np

def _load_phase1_frames(video_path, sample_frames=25, max_seconds=3):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_needed = int(fps * max_seconds)
    step = max(1, total_needed // sample_frames)
    
    frames = []
    i = 0
    while len(frames) < sample_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0:
            frames.append(cv2.resize(frame, (320, 240)))
        i += 1
        if i > total_needed:
            break
    cap.release()
    return frames, fps


def detect_static_from_frames(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    diffs = [np.mean(cv2.absdiff(grays[i], grays[i+1])) for i in range(len(grays)-1)]
    mean_diff = float(np.mean(diffs)) if diffs else 0.0
    return mean_diff < 0.5, mean_diff


def detect_screen_display_from_frames(frames):
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
        scores.append(moire_score * 0.5 + uniformity_score * 0.3 + (pixel_structure / 1000) * 0.2)
    final_score = float(np.mean(scores))
    return final_score > 21, final_score


def detect_screen_flicker_from_frames(frames, fps):
    brightness = []
    for frame in frames:
        small = cv2.resize(frame, (160, 120))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        brightness.append(np.mean(gray))
    b = np.array(brightness)
    fft = np.abs(np.fft.rfft(b - np.mean(b)))
    freqs = np.fft.rfftfreq(len(b), d=1.0 / fps)
    mask = (freqs >= 8) & (freqs <= 15)
    flicker_power = float(np.max(fft[mask])) if mask.any() else 0.0
    flicker_score = flicker_power / (float(np.sum(fft)) + 1e-6)
    return flicker_score > 0.09, flicker_score


def detect_screen_flatness_from_frames(frames):
    variances = [np.var(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    avg_var = float(np.mean(variances))
    return avg_var < 500, avg_var