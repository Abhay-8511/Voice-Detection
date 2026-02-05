import base64
import io
import librosa
from fastapi import HTTPException

def decode_audio(audio_base64: str):
    if not audio_base64:
        raise HTTPException(status_code=400, detail="audioBase64 is empty")

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    try:
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None,
            mono=True
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode MP3 audio")

    if audio is None or len(audio) == 0:
        raise HTTPException(status_code=400, detail="Empty or corrupted audio")

    return audio, sr
