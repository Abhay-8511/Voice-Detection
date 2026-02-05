import base64
import io
import soundfile as sf
from fastapi import HTTPException
import numpy as np

def decode_audio(audio_base64: str):
    if not audio_base64:
        raise HTTPException(status_code=400, detail="audioBase64 is empty")

    try:
        audio_bytes = base64.b64decode(audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio")

    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to decode MP3 audio")

    if audio is None or len(audio) == 0:
        raise HTTPException(status_code=400, detail="Empty or corrupted audio")

    # Convert stereo → mono if needed
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr
