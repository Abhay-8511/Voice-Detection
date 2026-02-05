import numpy as np
def extract_features(audio_length, sr=None):
    # Create deterministic proxy features from input size
    np.random.seed(audio_length % 1000)

    features = np.random.rand(1, 14)
    return features
