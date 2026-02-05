import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "voice_detector.pkl")

model = joblib.load(MODEL_PATH)
EXPECTED_FEATURES = model.n_features_in_

def predict_voice(features):
    # Ensure correct feature size
    current_len = features.shape[1]

    if current_len < EXPECTED_FEATURES:
        pad_width = EXPECTED_FEATURES - current_len
        features = np.pad(features, ((0, 0), (0, pad_width)))
    elif current_len > EXPECTED_FEATURES:
        features = features[:, :EXPECTED_FEATURES]

    probs = model.predict_proba(features)[0]

    if probs[1] > probs[0]:
        return "AI_GENERATED", float(probs[1]), "Synthetic speech characteristics detected"
    else:
        return "HUMAN", float(probs[0]), "Natural human voice patterns detected"
