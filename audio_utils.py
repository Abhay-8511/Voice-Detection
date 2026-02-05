import base64, io
import soundfile as sf
import numpy as np
from fastapi import HTTPException

def decode_audio(audio_base64: str):
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) < sr * 0.5:
        raise HTTPException(status_code=400, detail="Audio too short")

    return audio, sr
