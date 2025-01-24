import pytest

from classical_composer.frames import extract_random_frames


def test_extract_random_frames():
    """Tests the extract_random_frames function does generate random frame indicies."""
    piano_roll_shape = (128, 1000)  # Example shape
    n_frames = 5
    frame_size = 50
    buffer_size = 5
    random_seed = 42

    frames = extract_random_frames(piano_roll_shape, n_frames, frame_size, buffer_size, random_seed)

    assert len(frames) == n_frames, "Number of frames extracted is incorrect"
    for start, end in frames:
        assert end - start == frame_size, "Frame size is incorrect"
        assert (
            buffer_size <= start < piano_roll_shape[1] - buffer_size - frame_size
        ), "Frame start index is out of range"
        assert (
            buffer_size + frame_size <= end <= piano_roll_shape[1] - buffer_size
        ), "Frame end index is out of range"

    # Check for reproducibility
    frames_again = extract_random_frames(
        piano_roll_shape, n_frames, frame_size, buffer_size, random_seed
    )
    assert frames == frames_again, "Frames are not reproducible with the same random seed"

    return frames


def test_piano_roll_too_short():
    """Tests the extract_random_frames function throws an error when piano roll is too short."""
    piano_roll_shape = (128, 30)  # Example shape too short for the frame size
    n_frames = 5
    frame_size = 50
    buffer_size = 5
    random_seed = 42

    with pytest.raises(
        ValueError, match="Piano roll is too short for the required frame size or duration."
    ):
        extract_random_frames(piano_roll_shape, n_frames, frame_size, buffer_size, random_seed)
