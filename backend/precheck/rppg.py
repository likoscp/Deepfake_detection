import numpy as np
import cv2

def detect_rppg_absence(face_cache, fps=25.0):

    g_signal = []
    frame_indices = []

    for idx, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]

        fw, fh = x2 - x1, y2 - y1
        if fw < 32 or fh < 32:
            continue

        forehead = face[:max(1, int(fh * 0.35)), :]
        if forehead.size == 0:
            continue

        b = float(np.mean(forehead[:, :, 0]))
        g = float(np.mean(forehead[:, :, 1]))
        r = float(np.mean(forehead[:, :, 2]))
        total = r + g + b + 1e-6

        g_signal.append(g / total)
        frame_indices.append(int(idx))

    if len(g_signal) < 10:
        return False, 0.0

    sig = np.array(g_signal, dtype=np.float64)
    idxs = np.array(frame_indices, dtype=np.float64)

    t = np.arange(len(sig), dtype=np.float64)
    poly = np.polyfit(t, sig, 1)
    sig = sig - np.polyval(poly, t)
    n = len(sig)
    if n > 1 and fps > 0:
        span_frames = idxs[-1] - idxs[0]
        span_seconds = span_frames / fps if span_frames > 0 else (n - 1) / fps
        eff_fps = (n - 1) / max(span_seconds, 0.5)
    else:
        eff_fps = fps
    eff_fps = min(eff_fps, fps)

    freqs = np.fft.rfftfreq(n, d=1.0 / eff_fps)
    fft_mag = np.abs(np.fft.rfft(sig))

    pulse_mask = (freqs >= 0.75) & (freqs <= 2.5)
    noise_mask = ((freqs > 0.05) & (freqs < 0.75)) | ((freqs > 2.5) & (freqs <= 5.0))

    if not pulse_mask.any() or not noise_mask.any():
        return False, 0.0

    pulse_peak = float(fft_mag[pulse_mask].max())
    noise_mean = float(np.mean(fft_mag[noise_mask])) + 1e-9

    snr = pulse_peak / noise_mean

    score = float(max(0.0, 1.0 - min(snr / 2.5, 1.0)))
    return score > 0.60, score
