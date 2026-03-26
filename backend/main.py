from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.staticfiles import StaticFiles

from precheck.precheck_runner import run_full_check
from auth.auth import send_code, check_code

import os

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_VIDEO_PATH = "temp/temp.mp4"
os.makedirs("temp", exist_ok=True)


@app.post("/request-access")
async def request_access(email: str = Form(...)):
    try:
        send_code(email)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


@app.post("/verify-video")
async def verify_video(
    email: str = Form(...),
    code: str = Form(...),
    file: UploadFile = File(...),
):
    # check_code(email, code)

    contents = await file.read()
    with open(TEMP_VIDEO_PATH, "wb") as f:
        f.write(contents)

    return run_full_check(TEMP_VIDEO_PATH)


app.mount("/", StaticFiles(directory="../frontend_MVP"), name="frontend")
