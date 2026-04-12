import cv2
import numpy as np
def detect_temporal_inconsistency(face_cache, threshold=0.035):
    if len(face_cache) < 4:
        return False, 0.0

    face_flows = []
    bg_flows = []

    def _small(frame, box, max_dim=320):
        h, w = frame.shape[:2]
        scale = min(max_dim/w, max_dim/h, 1.0)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            x, y, bw, bh = box
            box = (int(x*scale), int(y*scale), int(bw*scale), int(bh*scale))
        return frame, box

    _, prev_frame, prev_box = face_cache[0]
    prev_frame, prev_box = _small(prev_frame, prev_box)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    for _, frame, box in face_cache[1:]:
        frame, (x, y, w, h) = _small(frame, box)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mask_face = np.zeros(gray.shape, dtype=bool)
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(gray.shape[1], x+w), min(gray.shape[0], y+h)
        mask_face[y1:y2, x1:x2] = True

        pad = 10
        mask_bg = ~mask_face.copy()
        mask_bg[max(0,y1-pad):min(gray.shape[0],y2+pad),
                max(0,x1-pad):min(gray.shape[1],x2+pad)] = False

        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        face_mag = float(np.mean(mag[mask_face])) if mask_face.any() else 0.0
        bg_mag   = float(np.mean(mag[mask_bg]))   if mask_bg.any()   else 0.0

        face_flows.append(face_mag)
        bg_flows.append(bg_mag)
        prev_gray = gray

    if not face_flows:
        return False, 0.0

    face_arr = np.array(face_flows)
    bg_arr   = np.array(bg_flows)

    if face_arr.mean() < 0.3 or bg_arr.mean() < 0.3:
        return False, 0.0

    if bg_arr.std() < 1e-6 or face_arr.std() < 1e-6:
        return False, 0.0

    corr = float(np.corrcoef(face_arr, bg_arr)[0, 1])
    score = float(max(0.0, 1.0 - corr))
    return bool(score > threshold), score