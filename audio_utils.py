
from fastapi import HTTPException
import base64

def decode_audio(audio_base64: str):
    if not audio_base64:
        raise HTTPException(status_code=400, detail="audioBase64 is empty")

    try:
        raw = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    # Return proxy signal length (no decoding)
    return len(raw)
