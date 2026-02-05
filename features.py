import numpy as np
import librosa

def extract_features(audio, sr):
    # Limit audio length (performance)
    audio = audio[: sr * 5]

    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
        axis=1
    )

    spectral = np.mean(librosa.feature.spectral_flatness(y=audio))

    features = np.hstack([mfcc, spectral])
    return features.reshape(1, -1)
    
