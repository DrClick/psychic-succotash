"""Provides functionality for extracting features from the midi file/frames.

Includes:
- FeatrureExtractor class to extract features from the midi file/frames
- FeatureFunction type to define the feature function
"""

from typing import Callable, Dict, Sequence
import numpy as np

# Define the feature function type
FeatureFunction = Callable[[np.ndarray], Dict[str, float]]


class FeatureExtractor:
    """Extract features from a MIDI frame using a list of feature functions."""

    def __init__(self, feature_functions: Sequence[FeatureFunction]):
        """
        Initialize the FeatureExtractor with a list of feature functions.

        Args
        ----
            feature_functions: A list of functions that take a frame and return a
            dictionary of features.
        """
        self.feature_functions = feature_functions

    def extract_all_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Extract features from a frame using all registered feature functions.

        Args
        ----
            frame: 2D NumPy array (128 pitches x time steps).

        Returns
        -------
            A dictionary containing all extracted features.
        """
        features = {}
        for feature_function in self.feature_functions:
            features.update(feature_function(frame))
        return features
