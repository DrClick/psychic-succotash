import os
import pandas as pd
import pytest
import pretty_midi
from unittest.mock import patch
from classical_composer.data import generate_dataset


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


@patch("classical_composer.data.extract_fixed_frames")
@patch("classical_composer.data.extract_random_frames")
def test_generate_dataset(
    mock_extract_random_frames, mock_extract_fixed_frames, dataframe, tmp_path
):
    # Mock the frame extraction functions
    mock_extract_fixed_frames.return_value = [(0, 100)]
    mock_extract_random_frames.return_value = [(100, 200)]

    # Call the function to generate the dataset
    dataset = generate_dataset(dataframe, output_folder=tmp_path)

    # Check that the output folder was created
    assert os.path.exists(tmp_path)

    # Check that the dataset DataFrame has the expected structure
    assert isinstance(dataset, pd.DataFrame)
    assert set(dataset.columns) == {"file_index", "composer", "dataset", "frame_start", "frame_end"}

    # Check that the dataset contains the expected data
    assert len(dataset) == 2
    assert dataset.iloc[0]["frame_start"] == 0
    assert dataset.iloc[0]["frame_end"] == 100
    assert dataset.iloc[1]["frame_start"] == 100
    assert dataset.iloc[1]["frame_end"] == 200

    # Check that the mock functions were called with the expected arguments
    mock_extract_fixed_frames.assert_called_once()
    mock_extract_random_frames.assert_called_once()
