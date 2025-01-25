from typing import Dict
import numpy as np
from scipy.stats import entropy


def pitch_based_features(frame: np.ndarray) -> Dict[str, float]:
    """Extract pitch-based features from a frame.

    Args
    ----
        frame: 2D NumPy array (128 pitches x time steps).

    Returns
    -------
        FeatureFunction of pitch-based features.
    """
    active_notes = (frame > 0).astype(int)
    pitch_histogram = (
        active_notes.sum(axis=1) / active_notes.sum()
        if active_notes.sum() > 0
        else np.zeros(frame.shape[0])
    )
    pitch_entropy = entropy(pitch_histogram + 1e-6)  # Avoid log(0)
    active_pitches = np.where(pitch_histogram > 0)[0]
    pitch_range = active_pitches.max() - active_pitches.min() if active_pitches.size > 0 else 0
    dominant_pitch = np.argmax(pitch_histogram)

    return {
        "pitch_range": pitch_range,
        "pitch_entropy": pitch_entropy,
        "dominant_pitch": dominant_pitch,
    }
