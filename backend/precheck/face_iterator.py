import cv2
#мб для оптимизации кидать рез в модель? пока так оставлю
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def iterate_faces(video_path, max_frames=100):
    cap = cv2.VideoCapture(video_path)

    while max_frames > 0:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

        if len(faces):
            yield frame, faces[0]
            max_frames -= 1

    cap.release()
