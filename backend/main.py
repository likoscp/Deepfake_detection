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

os.makedirs("temp", exist_ok=True)


@app.post("/request-access")
async def request_access(email: str = Form(...)):
    try:
        send_code(email)
    except Exception as e:
        raise HTTPException(500, str(e))
    return {"ok": True}


import uuid

@app.post("/verify-video")
async def verify_video(email: str = Form(...), code: str = Form(...), file: UploadFile = File(...)):
    check_code(email, code)

    video_id = uuid.uuid4().hex
    path = f"temp/{video_id}.mp4"

    contents = await file.read()
    with open(path, "wb") as f:
        f.write(contents)

    try:
        return run_full_check(path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass



app.mount("/", StaticFiles(directory="../frontend_MVP"), name="frontend")
