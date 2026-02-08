import time, secrets, smtplib
from email.message import EmailMessage
from fastapi import HTTPException
from dotenv import load_dotenv
import os

load_dotenv()

OTP_TTL = 10 * 60
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

codes: dict[str, dict] = {}

def send_code(email: str):
    code = f"{secrets.randbelow(1_000_000):06d}"
    codes[email] = {
        "code": code,
        "expires_at": time.time() + OTP_TTL}

    msg = EmailMessage()
    msg["Subject"] = "Deepfake verification access code"
    msg["From"] = SMTP_USER
    msg["To"] = email
    msg.set_content(
        f"Your access code: {code}\n"
        f"It is valid for {OTP_TTL//60} minutes."
    )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


def check_code(email: str, code: str):
    rec = codes.get(email)
    if not rec:
        raise HTTPException(403, "Code not requested")

    if time.time() > rec["expires_at"]:
        codes.pop(email, None)
        raise HTTPException(403, "Code expired")

    if rec["code"] != code:
        raise HTTPException(403, "Invalid code")

    codes.pop(email, None)
