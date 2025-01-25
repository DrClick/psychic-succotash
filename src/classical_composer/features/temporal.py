from typing import Dict
import numpy as np
from scipy.signal import find_peaks


def temporal_features(frame: np.ndarray) -> Dict[str, float]:
    """Extract temporal features from a frame.

    Args
    ----
        frame: 2D NumPy array (128 pitches x time steps).

    Returns
    -------
        DiFeatureFunctionctionary of temporal features.
    """
    active_notes = (frame > 0).astype(int)
    note_density = active_notes.sum(axis=0).mean()

    # Note durations
    durations = active_notes.sum(axis=1)
    mean_duration = durations.mean() if durations.size > 0 else 0
    variance_duration = durations.var() if durations.size > 0 else 0

    # Syncopation: Peaks in active notes over time
    peaks, _ = find_peaks(active_notes.sum(axis=0))
    syncopation = len(peaks)

    return {
        "note_density": note_density,
        "mean_duration": mean_duration,
        "variance_duration": variance_duration,
        "syncopation": syncopation,
    }
