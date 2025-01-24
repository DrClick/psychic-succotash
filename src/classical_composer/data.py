"""
Provides functionality for managing and processing data in the Classical Composer project.

Includes:
- Functions to generate datasets.
- Utilities to preprocess MIDI files for analysis.
- Data loading and transformation helpers.
"""

import os
import shutil
import zipfile
import pretty_midi
import pandas as pd

from sklearn.model_selection import train_test_split


def extract_data(
    source_data_file="data/Challenge_DataSet.zip", output_folder="data/Challenge_DataSet"
):
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


def create_file_dataframe(data_folder="data/Challenge_DataSet"):
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


# def generate_dataset(df, output_folder="data/processed", frame_length_seconds=30, frame_step=100):
#     pass


def stratified_group_split(df, group_col, stratify_col, test_size=0.2, random_state=42):
    """Split a dataset into train and test sets.

    Ensuring all rows with the same group_col value are in one set or the other, stratified by
    stratify_col.

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


# Function to generate piano roll
def generate_piano_roll(file_path, fs=100):
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
    midi_data = pretty_midi.PrettyMIDI(file_path)
    piano_roll = midi_data.get_piano_roll(fs=fs)
    return piano_roll
