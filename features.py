import numpy as np
import librosa

def extract_features(audio, sr):
    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13), axis=1)

    pitch = librosa.yin(audio, fmin=50, fmax=300)
    pitch_std = np.std(pitch[pitch > 0])

    spectral_flatness = np.mean(
        librosa.feature.spectral_flatness(y=audio)
    )

    features = np.hstack([
        mfcc,
        pitch_std,
        spectral_flatness
    ])

    return features.reshape(1, -1)

