import cv2
from face_iterator import iterate_faces

EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

def detect_blinks(video_path, max_faces=100):
    closed_eye_frames = 0
    valid_frames = 0

    for frame, (x, y, w, h) in iterate_faces(video_path, max_faces):
        face = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        eyes = EYE_CASCADE.detectMultiScale(gray_face, 1.3, 5)
        valid_frames += 1
        if len(eyes) == 0:
            closed_eye_frames += 1

    if valid_frames == 0:
        return True, 0.0  

    blink_ratio = closed_eye_frames / valid_frames
    is_no_blink = blink_ratio < 0.05 #я бы уменьшила или поставила обработку след слайдов, но тогда дольше препроверка была бы, соу соу

    return is_no_blink, blink_ratio
