"""Provides functionality for extracting features from the midi file/frames.

Includes:
- FeatrureExtractor class to extract features from the midi file/frames
- FeatureFunction type to define the feature function
"""

from typing import Callable, Dict, Sequence
import numpy as np

from classical_composer.features import (
    FeatureFunction,
    frequency_based_features,
    harmonic_features,
    higher_level_features,
    pitch_based_features,
    temporal_features,
    velocity_based_features,
)

class FeatureExtractor:
    """Extract features from a MIDI frame using a list of feature functions."""

    def __init__(self, feature_functions: Sequence[FeatureFunction] = None, 
                 feature_names: Sequence[str] = None):
        """
        Initialize the FeatureExtractor with a list of feature functions.

        Args
        ----
            feature_functions: A list of functions that take a frame and return a
            dictionary of features (default: None).
            
            feature_names: A list of feature names corresponding to the extracted features
            (default: None).

        """
        self.feature_functions = feature_functions
        if feature_functions == None:
            self.feature_functions = [
                pitch_based_features,
                velocity_based_features,
                temporal_features,
                harmonic_features,
                frequency_based_features,
                higher_level_features,
            ]

        self.feature_names = feature_names or [
                "pitch_entropy",
                "dominant_pitch",
                "avg_velocity",
                "spectral_bandwidth"
        ]


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
