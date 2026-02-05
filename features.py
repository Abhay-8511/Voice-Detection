import numpy as np
import librosa

def extract_features(audio, sr):
    # Limit duration to 6 seconds (important for performance)
    max_len = sr * 6
    audio = audio[:max_len]

    # MFCCs
    mfcc = np.mean(
        librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13),
        axis=1
    )

    # Pitch (YIN)
    pitch = librosa.yin(audio, fmin=50, fmax=300)
    pitch = pitch[pitch > 0]

    pitch_mean = np.mean(pitch) if len(pitch) else 0
    pitch_std = np.std(pitch) if len(pitch) else 0

    # Spectral flatness
    flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

    # Energy variation (proxy for breath & human noise)
    energy = librosa.feature.rms(y=audio)[0]
    energy_var = np.var(energy)

    return np.hstack([
        mfcc,
        pitch_mean,
        pitch_std,
        flatness,
        energy_var
    ]).reshape(1, -1)
