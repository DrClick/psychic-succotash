from typing import Dict
import numpy as np


def velocity_based_features(frame: np.ndarray) -> Dict[str, float]:
    """Extract velocity-based features from a frame.

    Args
    ----
        frame: 2D NumPy array (128 pitches x time steps).

    Returns
    -------
        FeatureFunction of velocity-based features.
    """
    velocities = frame[frame > 0]
    avg_velocity = velocities.mean() if velocities.size > 0 else 0
    velocity_variance = velocities.var() if velocities.size > 0 else 0
    silent_notes_ratio = 1 - (frame > 0).sum() / frame.size

    return {
        "avg_velocity": avg_velocity,
        "velocity_variance": velocity_variance,
        "silent_notes_ratio": silent_notes_ratio,
    }
