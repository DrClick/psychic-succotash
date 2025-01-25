from typing import Dict
import numpy as np
from scipy.fft import fft, fftfreq


def frequency_based_features(frame: np.ndarray) -> Dict[str, float]:
    """Extract frequency-based features from a frame.

    Args
    ----
        frame: 2D NumPy array (128 pitches x time steps).

    Returns
    -------
        FeatureFunction of frequency-based features.
    """
    frame_summed = frame.sum(axis=0)  # Sum velocities over pitches
    fft_result = fft(frame_summed)
    fft_magnitude = np.abs(fft_result)
    freqs = fftfreq(len(fft_result))

    spectral_centroid = (
        (np.sum(freqs * fft_magnitude) / np.sum(fft_magnitude)) if np.sum(fft_magnitude) > 0 else 0
    )
    spectral_bandwidth = (
        np.sqrt(np.sum((freqs - spectral_centroid) ** 2 * fft_magnitude) / np.sum(fft_magnitude))
        if np.sum(fft_magnitude) > 0
        else 0
    )

    return {
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
    }
