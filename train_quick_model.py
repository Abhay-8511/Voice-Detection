import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# MUST MATCH features.py (13 MFCC + 1 spectral)
NUM_FEATURES = 14

X = np.random.rand(200, NUM_FEATURES)
y = np.array([0]*100 + [1]*100)  # 0=HUMAN, 1=AI

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "voice_detector.pkl")
print("voice_detector.pkl regenerated with correct feature size")
