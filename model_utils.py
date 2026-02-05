import os
import joblib

# Absolute path to model file (cloud-safe)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "voice_detector.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

def predict_voice(features):
    probabilities = model.predict_proba(features)[0]

    if probabilities[1] > probabilities[0]:
        return (
            "AI_GENERATED",
            float(probabilities[1]),
            "Synthetic speech patterns and unnaturally stable pitch detected"
        )
    else:
        return (
            "HUMAN",
            float(probabilities[0]),
            "Natural pitch variation and human voice irregularities detected"
        )
