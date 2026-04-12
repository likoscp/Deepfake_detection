import cv2
import numpy as np

def detect_temporal_freq(face_cache, fps=25.0):

    region_signals = [[] for _ in range(4)]
    frame_indices = []

    for idx, frame, (x, y, w, h) in face_cache:
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
        face = frame[y1:y2, x1:x2]

        fw, fh = x2 - x1, y2 - y1
        if fw < 32 or fh < 32:
            continue

        face = cv2.resize(face, (64, 64))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)

        region_signals[0].append(float(np.mean(gray[:32, :32])))
        region_signals[1].append(float(np.mean(gray[:32, 32:])))
        region_signals[2].append(float(np.mean(gray[32:, :32])))
        region_signals[3].append(float(np.mean(gray[32:, 32:])))
        frame_indices.append(int(idx))

    n = len(frame_indices)
    if n < 8:
        return False, 0.0

    idxs = np.array(frame_indices, dtype=np.float64)
    span_frames = idxs[-1] - idxs[0]
    eff_fps = float((n - 1) / max(span_frames / fps, 0.5)) if span_frames > 0 else fps
    eff_fps = min(eff_fps, fps)

    snr_list = []
    freqs = np.fft.rfftfreq(n, d=1.0 / eff_fps)

    micro_mask = (freqs >= 3.0) & (freqs <= 15.0)
    broad_mask = (freqs >= 0.5) & (freqs <= 30.0)

    if not micro_mask.any() or not broad_mask.any():
        return False, 0.0

    for sig_raw in region_signals:
        if len(sig_raw) < n:
            continue
        sig = np.array(sig_raw, dtype=np.float64)
        t = np.arange(n, dtype=np.float64)
        sig -= np.polyval(np.polyfit(t, sig, 1), t)

        mag = np.abs(np.fft.rfft(sig))
        micro_energy = float(np.sum(mag[micro_mask] ** 2))
        broad_energy = float(np.sum(mag[broad_mask] ** 2)) + 1e-9
        snr_list.append(micro_energy / broad_energy)

    if not snr_list:
        return False, 0.0

    snr_micro = float(np.mean(snr_list))
    score = float(max(0.0, 0.20 - snr_micro))
    return score > 0.12, score
