from typing import Dict
import numpy as np


def higher_level_features(frame: np.ndarray) -> Dict[str, float]:
    """Extract higher-level features from a frame.

    Args
    ----
        frame: 2D NumPy array (128 pitches x time steps).

    Returns
    -------
        FeatureFunction of higher-level features.
    """
    frame = frame  # TODO: Remove this line (dummy assignment)
    # TODO: Implement tonality detection and repetitive patterns
    tonality = 0.0  # TBD: Return 0 for now (null feature)
    repetitive_patterns = 0.0  # TBD: Return 0 for now (null feature)
    frame_context = 0.0  # TBD: Return 0 for now (null feature)

    return {
        "tonality": tonality,
        "repetitive_patterns": repetitive_patterns,
        "frame_context": frame_context,
    }
