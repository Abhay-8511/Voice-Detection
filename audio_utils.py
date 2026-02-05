import base64
import io
import librosa

def decode_audio(audio_base64: str):
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=None,
            mono=True
        )
        return audio, sr
    except Exception:
        raise ValueError("Invalid audio data")
