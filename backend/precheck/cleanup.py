import os
import shutil
# НЕ ЮЗАТЬ ПОКА ЧТО ЭТО НА НОВЫЙ ГОД (удалится твое видео)

def cleanup(VIDEO_PATH):
    if os.path.exists(VIDEO_PATH):
        os.remove(VIDEO_PATH)
        print(f"Video {VIDEO_PATH} removed after processing.")

def cleanup_pycache():
    for root, dirs, files in os.walk("./precheck/", topdown=False):
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d))
                print(f"Removed {os.path.join(root, d)}")
