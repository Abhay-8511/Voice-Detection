import os
import numpy as np
import librosa
import joblib
from features import extract_features

DATASET_PATH = "dataset"
X, y = [], []

LABELS = {
    "human": 0,
    "ai": 1
}

for label in LABELS:
    for language in os.listdir(f"{DATASET_PATH}/{label}"):
        folder = f"{DATASET_PATH}/{label}/{language}"
        for file in os.listdir(folder):
            if file.endswith(".mp3"):
                path = os.path.join(folder, file)
                audio, sr = librosa.load(path, sr=None)
                features = extract_features(audio, sr)
                X.append(features[0])
                y.append(LABELS[label])

X = np.array(X)
y = np.array(y)

joblib.dump((X, y), "training_data.pkl")
print("✅ Dataset prepared successfully")
