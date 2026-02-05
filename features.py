import numpy as np
import librosa

def extract_features(audio, sr):
    # Shorten audio to first 5 seconds (prevents timeout)
    max_len = sr * 5
    audio = audio[:max_len]

    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
        axis=1
    )

    spectral_flatness = np.mean(
        librosa.feature.spectral_flatness(y=audio)
    )

    features = np.hstack([mfcc, spectral_flatness])
    return features.reshape(1, -1)
    
