import cv2
import numpy as np

def detect_screen_flicker(video_path, max_frames=120):
    cap = cv2.VideoCapture(video_path)
    intensities = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        intensities.append(np.mean(gray))

    cap.release()

    intensities = np.array(intensities)
    if len(intensities) < 30:
        return False, 0.0

    fft = np.abs(np.fft.rfft(intensities - intensities.mean()))
    fft_energy = np.sum(fft[5:]) / (np.sum(fft) + 1e-6)

    is_flicker = fft_energy > 0.35
    return is_flicker, fft_energy
