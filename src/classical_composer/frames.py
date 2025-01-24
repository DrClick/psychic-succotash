"""
Provide functionality for manipulating frames in the Classical Composer project.

Includes:
- Extracting fixed frames from the beginning and end of a piano roll.
- Extracting random frames from a piano roll excluding the fixed frames.
"""

import numpy as np


def extract_fixed_frames(piano_roll_shape, frame_size):
    """Extract fixed frames indicies from the beginning and end of a piano roll.

    Args
    ----
        piano_roll_shape: tuple, shape of the piano roll.
        frame_size: int, size of the frame to extract.

    Returns
    -------
        frames: list of lists, each containing the start and end indices of a frame.

    """
    frames = []
    piano_roll_length = piano_roll_shape[1]
    end_frame_start = piano_roll_length - frame_size  # Last 30 seconds

    if piano_roll_length >= frame_size:
        # First frame
        frames.append((0, frame_size))
        # Last frame
        frames.append((end_frame_start, end_frame_start + frame_size))
    else:
        raise ValueError("Piano roll is too short for the required frame size or duration.")

    return frames


# Function to extract random frames excluding the fixed ones
def extract_random_frames(piano_roll_shape, n_frames, frame_size, buffer_size=5, random_seed=42):
    """Extract random frames from a piano roll.

    Uses buffer_size to exclude frames from starting at the the start and end of the piano roll.

    Args
    ----
        piano_roll_shape: tuple, shape of the piano roll.
        n_frames: int, number of frames to extract.
        frame_size: int, size of the frame to extract.
        buffer_size: int, number of frames to exclude from the beginning and end of the file.
        random_seed: int, random seed for reproducibility.


    Returns
    -------
        frames: list of lists, each containing the start and end indices of a frame.

    """
    piano_roll_length = piano_roll_shape[1]

    if piano_roll_length < frame_size:
        raise ValueError("Piano roll is too short for the required frame size or duration.")

    # Define random range excluding the buffer_size from the file ends
    index_range = range(buffer_size, piano_roll_length - buffer_size - frame_size)
    np.random.seed(random_seed)
    random_starts = np.random.choice(index_range, size=n_frames, replace=False)
    frames = [(start, start + frame_size) for start in random_starts]
    return frames
