import os
import numpy as np
import pandas as pd
import pytest
import pretty_midi
from unittest.mock import patch

from classical_composer.data import generate_dataset, save_frames_to_disk


@pytest.fixture
def midi_file(tmp_path):
    # Create a temporary MIDI file for testing
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    note = pretty_midi.Note(velocity=100, pitch=60, start=0, end=1)
    piano.notes.append(note)
    midi.instruments.append(piano)
    midi_path = tmp_path / "test.mid"
    midi.write(str(midi_path))
    return midi_path


@pytest.fixture
def dataframe(midi_file):
    # Create a DataFrame with the path to the temporary MIDI file
    data = {
        "file_index": [0],
        "filepath": [str(midi_file)],
        "dataset": ["PS1"],
        "composer": ["Composer1"],
    }
    df = pd.DataFrame(data)
    df.set_index("file_index", inplace=True)
    return df


@patch("classical_composer.data.get_fixed_frame_indicies")
@patch("classical_composer.data.generate_random_frame_indicies")
def test_generate_dataset(
    mock_generate_random_frame_indicies, mock_get_fixed_frame_indicies, dataframe, tmp_path
):
    # Mock the frame extraction functions
    mock_get_fixed_frame_indicies.return_value = [(0, 100)]
    mock_generate_random_frame_indicies.return_value = [(100, 200)]

    # Call the function to generate the dataset
    dataset = generate_dataset(dataframe, output_folder=tmp_path)

    # Check that the output folder was created
    assert os.path.exists(tmp_path)

    # Check that the dataset DataFrame has the expected structure
    assert isinstance(dataset, pd.DataFrame)
    assert set(dataset.columns) == {
        "file_index",
        "file_path",
        "composer",
        "dataset",
        "frame_start",
        "frame_end",
        "frame_id",
    }

    # Check that the dataset contains the expected data
    assert len(dataset) == 2
    assert dataset.iloc[0]["frame_start"] == 0
    assert dataset.iloc[0]["frame_end"] == 100
    assert dataset.iloc[0]["file_path"] == f"{tmp_path}/test.mid"
    assert dataset.iloc[0]["frame_id"] == "0_0_100"
    assert dataset.iloc[1]["frame_start"] == 100
    assert dataset.iloc[1]["frame_end"] == 200
    assert dataset.iloc[1]["file_path"] == f"{tmp_path}/test.mid"
    assert dataset.iloc[1]["frame_id"] == "0_100_200"

    # Check that the mock functions were called with the expected arguments
    mock_get_fixed_frame_indicies.assert_called_once()
    mock_generate_random_frame_indicies.assert_called_once()


def test_save_frames_to_disk(tmp_path):
    # Create some dummy frames
    frames = [np.random.rand(128, 100) for _ in range(3)]
    frame_idx = ["frame_1", "frame_2", "frame_3"]

    # Call the function to save the frames to disk
    save_frames_to_disk(tmp_path, frame_idx, frames)

    # Check that the output folder was created
    assert os.path.exists(tmp_path)

    # Check that the frames were saved to disk
    for idx in frame_idx:
        assert os.path.exists(os.path.join(tmp_path, f"frames/{idx}.npy"))

    # Check that the saved frames match the original frames
    for idx, frame in zip(frame_idx, frames):
        saved_frame = np.load(os.path.join(tmp_path, f"frames/{idx}.npy"))
        np.testing.assert_array_equal(saved_frame, frame)
