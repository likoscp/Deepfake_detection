from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles

import os
from precheck.precheck_runner import run_full_check

app = FastAPI()
TEMP_VIDEO_PATH = "temp/temp.mp4"

@app.post("/verify-video")
async def verify_video_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open(TEMP_VIDEO_PATH, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    results = run_full_check(TEMP_VIDEO_PATH)

    return results

app.mount(
    "/frontend",
    StaticFiles(directory="../frontend_MVP"),
    name="frontend"
)
