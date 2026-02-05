from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from auth import validate_api_key
from audio_utils import decode_audio
from features import extract_features
from model_utils import predict_voice

app = FastAPI()

SUPPORTED_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
def health():
    return {"status": "Voice Detection API is running"}

@app.post("/api/voice-detection")
def voice_detection(payload: VoiceRequest, x_api_key: str = Header(None)):
    validate_api_key(x_api_key)

    if payload.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    if payload.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 format supported")

    try:
        audio, sr = decode_audio(payload.audioBase64)
        features = extract_features(audio, sr)
        classification, confidence, explanation = predict_voice(features)
    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(status_code=500, detail="Audio processing failed")

    return {
        "status": "success",
        "language": payload.language,
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
