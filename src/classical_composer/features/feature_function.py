# Define the feature function type
from typing import Callable, Dict
import numpy as np

FeatureFunction = Callable[[np.ndarray], Dict[str, float]]
