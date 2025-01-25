"""
Provides functionality for managing and processing data in the Classical Composer project.

Includes:
- Functions to generate datasets.
- Utilities to preprocess MIDI files for analysis.
- Data loading and transformation helpers.
"""

import os
import shutil
from typing import List, Tuple, cast
import warnings
import zipfile
import pretty_midi
import pandas as pd
from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split


from classical_composer.frames import get_fixed_frame_indicies, generate_random_frame_indicies


def extract_data(
    source_data_file: str = "data/Challenge_DataSet.zip",
    output_folder: str = "data/Challenge_DataSet",
) -> None:
    """Intakes the original training data and extracts it.

    Assumes the following structure:
    PS1
        - Composer1
            - File1.mid
            - File2.mid
            - ...
        - ComposerN
            - File1.mid
            - File2.mid
            - ...
    PS2
        - File1.mid
        - File2.mid
        - ...

    This will updated the folder structure to of the PS2 folder to be:

    PS2
        -Unknown
            - File1.mid
            - File2.mid
            - ...


    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Unzip the data folder if it exists as a zip file
    if os.path.exists(source_data_file):
        with zipfile.ZipFile(source_data_file, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(output_folder))

    # Create unknown folder in PS2 if it doesn't exist
    ps2_folder = os.path.join(output_folder, "PS2")
    unknown_folder = os.path.join(ps2_folder, "unknown")
    if not os.path.exists(unknown_folder):
        os.makedirs(unknown_folder)

    # Move all .mid files from PS2 root to unknown folder
    for file in os.listdir(ps2_folder):
        if file.endswith(".mid"):
            src = os.path.join(ps2_folder, file)
            dst = os.path.join(unknown_folder, file)
            shutil.move(src, dst)

    # List all files in a tree view in the console
    for root, dirs, files in os.walk(output_folder):
        level = root.replace(output_folder, "").count(os.sep)
        indent = " " * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


def create_file_dataframe(data_folder: str = "data/Challenge_DataSet") -> pd.DataFrame:
    """Create a DataFrame containing information about the MIDI files in the dataset.

    Args
    ----
        data_folder: Path to the folder containing the MIDI files.

    Returns
    -------
        df: DataFrame with columns 'file_index', 'file_path', 'dataset' and 'composer'

    """
    data = []
    file_index = 0
    for dataset in ["PS1", "PS2"]:
        dataset_folder = os.path.join(data_folder, dataset)
        for composer in os.listdir(dataset_folder):
            composer_folder = os.path.join(dataset_folder, composer)
            if os.path.isdir(composer_folder):
                for filename in os.listdir(composer_folder):
                    if filename.endswith(".mid"):
                        filepath = os.path.join(composer_folder, filename)
                        data.append([file_index, filepath, dataset, composer])
                        file_index += 1

    df = pd.DataFrame(data, columns=["file_index", "filepath", "dataset", "composer"])
    # Set file_index as the index
    df.set_index("file_index", inplace=True)
    return df


def generate_dataset(
    df: pd.DataFrame,
    output_folder: str = "data/processed",
    frame_length_seconds: int = 30,
    sample_freq: int = 100,
) -> pd.DataFrame:
    """Generate a dataset from a DataFrame of MIDI files.

    Args
    ----
        df: DataFrame containing information about the MIDI files.
        output_folder: Folder to save the processed data.
        frame_length_seconds: Duration of each frame in seconds.
        sample_freq: The length of each sample from the piano roll (1/n seconds).

    Returns
    -------
        dataset: DataFrame containing the processed data.

    """
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Initialize an empty list to store the dataset
    dataset = []

    for file_index, row in tqdm(df.iterrows(), desc="Processing MIDI files", total=len(df)):
        # Load the MIDI file
        midi_data = pretty_midi.PrettyMIDI(row["filepath"])

        # Extract the piano roll
        piano_roll = midi_data.get_piano_roll(fs=sample_freq)

        # Extract fixed frames from the beginning and end of the piano roll
        frame_size = frame_length_seconds * sample_freq
        fixed_frames = get_fixed_frame_indicies(piano_roll.shape, frame_size=frame_size)

        # Extract random frames from the piano roll
        random_frames = generate_random_frame_indicies(
            piano_roll.shape,
            n_frames=20,
            frame_size=frame_size,
            buffer_size=5,
            random_seed=42,
        )

        # Extract features from each frame
        for start, end in fixed_frames + random_frames:
            # Append the features to the dataset
            dataset.append(
                {
                    "file_index": file_index,
                    "file_path": row["filepath"],
                    "composer": row["composer"],
                    "dataset": row["dataset"],
                    "frame_start": start,
                    "frame_end": end,
                    "frame_id": f"{file_index}_{start}_{end}",
                }
            )

    # Convert the dataset to a DataFrame
    dataset = pd.DataFrame(dataset)

    return dataset


def generate_piano_roll(file_path: str, fs: int = 100) -> NDArray[np.float64]:
    """Generate a piano roll from a MIDI file.

    Args
    ----
        file_path: str, path to the MIDI file.
        fs: int, sampling frequency of the piano roll.

    Returns
    -------
        piano_roll: np.ndarray, binary matrix of shape (128, n_time_steps determined by
        (total time of file in seconds/fs)).

    """
    with warnings.catch_warnings():
        # this can be a bit noisy, ignore warnings for now
        warnings.simplefilter("ignore")  # Ignore all warnings
        midi_data = pretty_midi.PrettyMIDI(file_path)
        piano_roll = midi_data.get_piano_roll(fs=fs)

    return cast(NDArray[np.float64], piano_roll)


def save_frames_to_disk(output_folder: str, frame_idx: List[str], frames: List[np.ndarray]) -> None:
    """Save a list of piano roll frames to disk.

    Args
    ----
        output_folder: str, path to the output folder.
        frame_idx: list of str, frame identifiers.
        frames: list of np.ndarray, piano roll frames.

    """
    frames_folder = f"{output_folder}/frames"
    os.makedirs(frames_folder, exist_ok=True)
    for idx, frame in zip(frame_idx, frames):
        np.save(os.path.join(frames_folder, f"{idx}.npy"), frame)


def stratified_group_split(
    df: pd.DataFrame,
    group_col: str,
    stratify_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataset into train and test sets.

    Ensuring all rows with the same group_col value are in one set or the other,
    stratified by stratify_col.

    Args
    ----
        df: DataFrame containing the dataset.
        group_col: Column name that identifies groups (e.g., file_idx).
        stratify_col: Column name for stratification (e.g., composer).
        test_size: Proportion of the dataset to include in the test split.
        random_state: Random seed for reproducibility.

    Returns
    -------
        train_df: Training subset of the DataFrame.
        test_df: Testing subset of the DataFrame.
    """
    # Group by `group_col` and retain a single representative row for each group
    group_df = df.groupby(group_col).first().reset_index()

    # Perform stratified split on the grouped data
    train_groups, test_groups = train_test_split(
        group_df[group_col],  # Split by group_col values
        test_size=test_size,
        random_state=random_state,
        stratify=group_df[stratify_col],  # Stratify by the stratify_col
    )

    # Map the split back to the original dataset
    train_df = df[df[group_col].isin(train_groups)]
    test_df = df[df[group_col].isin(test_groups)]

    return train_df, test_df
