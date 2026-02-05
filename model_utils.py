import joblib
import numpy as np

model = joblib.load("voice_detector.pkl")

def predict_voice(features):
    probabilities = model.predict_proba(features)[0]

    if probabilities[1] > probabilities[0]:
        classification = "AI_GENERATED"
        explanation = "Unnaturally consistent pitch and low jitter detected"
        confidence = probabilities[1]
    else:
        classification = "HUMAN"
        explanation = "Natural pitch variation and breathing patterns detected"
        confidence = probabilities[0]

    return classification, float(confidence), explanation
