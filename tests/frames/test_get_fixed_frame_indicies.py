import pytest
from classical_composer.frames import get_fixed_frame_indicies


def test_get_fixed_frame_indicies_valid_input():
    piano_roll_shape = (88, 100)
    frame_size = 30
    expected_output = [(0, 30), (70, 100)]
    assert get_fixed_frame_indicies(piano_roll_shape, frame_size) == expected_output


def test_get_fixed_frame_indicies_exact_frame_size():
    piano_roll_shape = (88, 30)
    frame_size = 30
    expected_output = [(0, 30), (0, 30)]
    assert get_fixed_frame_indicies(piano_roll_shape, frame_size) == expected_output


def test_get_fixed_frame_indicies_piano_roll_too_short():
    piano_roll_shape = (88, 20)
    frame_size = 30
    with pytest.raises(
        ValueError, match="Piano roll is too short for the required frame size or duration."
    ):
        get_fixed_frame_indicies(piano_roll_shape, frame_size)
