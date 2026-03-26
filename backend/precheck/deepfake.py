
import cv2
import numpy as np
from .face_iterator import iterate_faces

def detect_gan_fingerprint(video_path, max_frames=30, threshold=0.55):

    scores = []

    for frame, (x, y, w, h) in iterate_faces(video_path, max_frames):
        face = frame[y:y+h, x:x+w]
        if face.size == 0:
            continue

        
        face = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)

        
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log(np.abs(fft_shift) + 1)

        
        magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

        h_m, w_m = magnitude.shape
        cy, cx = h_m // 2, w_m // 2

        
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[:cy//2, :] = 1
        high_freq_mask[cy + cy//2:, :] = 1
        high_freq_mask[:, :cx//2] = 1
        high_freq_mask[:, cx + cx//2:] = 1

        
        low_freq_mask = np.zeros_like(magnitude)
        r = min(cy, cx) // 3
        y_grid, x_grid = np.ogrid[:h_m, :w_m]
        circle = (y_grid - cy)**2 + (x_grid - cx)**2 <= r**2
        low_freq_mask[circle] = 1

        high_energy = (magnitude * high_freq_mask).sum()
        low_energy  = (magnitude * low_freq_mask).sum() + 1e-8

        
        ratio = high_energy / low_energy
        scores.append(ratio)

    if not scores:
        return False, 0.0

    mean_score = float(np.mean(scores))
    is_gan = mean_score > threshold

    return is_gan, mean_score









def _compute_lbp(gray, radius=1, n_points=8):
    
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)

    angles = [2 * np.pi * i / n_points for i in range(n_points)]
    neighbors = [
        (int(round(radius * np.sin(a))),
         int(round(radius * np.cos(a))))
        for a in angles
    ]

    for i, (dy, dx) in enumerate(neighbors):
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp += ((gray >= shifted).astype(np.uint8)) << i

    return lbp


def detect_texture_consistency(video_path, max_frames=60, threshold=0.5):

    scores = []

    for frame, (x, y, w, h) in iterate_faces(video_path, max_frames):
        face = frame[y:y+h, x:x+w]
        if face.size == 0 or w < 64 or h < 64:
            continue

        face_resized = cv2.resize(face, (128, 128))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY).astype(np.float32)

        
        
        lbp = _compute_lbp(gray.astype(np.uint8))
        lbp_var = float(np.var(lbp))
        
        lbp_score = 1.0 / (1.0 + lbp_var / 1000.0)

        
        
        center_region = gray[32:96, 32:96]
        edge_top    = gray[:16, :]
        edge_bottom = gray[112:, :]
        edge_left   = gray[:, :16]
        edge_right  = gray[:, 112:]

        center_std = float(np.std(center_region))
        edge_std   = float(np.mean([
            np.std(edge_top), np.std(edge_bottom),
            np.std(edge_left), np.std(edge_right)
        ]))

        
        if center_std + edge_std > 0:
            boundary_score = abs(center_std - edge_std) / (center_std + edge_std)
        else:
            boundary_score = 0.0

        
        
        laplacian = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F)
        lap_var = float(np.var(laplacian))
        
        lap_score = 1.0 / (1.0 + lap_var / 500.0)

        
        combined = (lbp_score * 0.4 +
                    boundary_score * 0.4 +
                    lap_score * 0.2)
        scores.append(combined)

    if not scores:
        return False, 0.0

    mean_score = float(np.mean(scores))
    is_artifact = mean_score > threshold

    return is_artifact, mean_score
