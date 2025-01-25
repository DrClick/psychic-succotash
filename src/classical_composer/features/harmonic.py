from typing import Dict
import numpy as np


def harmonic_features(frame: np.ndarray) -> Dict[str, float]:
    """Extract harmonic features from a frame.

    Args
    ----
        frame: 2D NumPy array (128 pitches x time steps).

    Returns
    -------
        FeatureFunction of harmonic features.
    """
    active_notes = (frame > 0).astype(int)
    active_pitches = np.where(active_notes.sum(axis=1) > 0)[0]

    # Intervals between active pitches
    harmonic_intervals = np.diff(active_pitches)
    interval_variance = harmonic_intervals.var() if harmonic_intervals.size > 0 else 0

    # TODO: Implement chord identification and consonance/dissonance calculation
    chord_identification = 0.0  # TBD: Return 0 for now (null feature)
    consonance_dissonance = 0.0  # TBD: Return 0 for now (null feature)

    return {
        "interval_variance": interval_variance,
        "chord_identification": chord_identification,
        "consonance_dissonance": consonance_dissonance,
    }
