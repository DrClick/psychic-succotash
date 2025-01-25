# This is needed to keep the code functioning with imports correctly and settings
import os
import sys

# Add the root directory (parent of scripts) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------
import numpy as np
import pandas as pd
from tqdm import tqdm

from classical_composer.data import (
    create_file_dataframe,
    extract_data,
    generate_dataset,
    generate_piano_roll,
    save_frames_to_disk,
)
from classical_composer.features.feature_extractor import FeatureExtractor
from classical_composer.features import (
    frequency_based_features,
    harmonic_features,
    higher_level_features,
    pitch_based_features,
    temporal_features,
    velocity_based_features,
)
from classical_composer.misc.logging import logger
from classical_composer.frames import extract_frames
from classical_composer.misc import load_pipeline_status, save_pipeline_status

from settings import settings

"""
This script generates the dataset for the classical composer challenge. The steps are as follows:

1) Unzip the original datasaet
2) move all .mid files from PS2 root to unknown folder
3) Create a dataframe containing information about MIDI files, including file paths, composers, and
file indices.
4) Extract 30 second frames from each MIDI file
5) Extract features from each frame and save them to disk
   Save the dataset to disk for later use
"""

# Load the current status
status = load_pipeline_status()

# ------------------------------------------------------------
# 1-2) Unzip the original datasaet
if not status["extract_data"]:
    logger.info("Step 1,2) Extracting data from the original dataset")
    extract_data(source_data_file=settings.source_data_file, output_folder=settings.data_folder)
    logger.info("Data extraction complete")
    status["extract_data"] = True
    save_pipeline_status(status)
else:
    logger.info("Step 1,2) Data has already been extracted, skipping this step")

# ------------------------------------------------------------
# 3) Create a dataframe containing information about MIDI files, including file paths, composers,
# and file indices.
if not status["create_dataframe"]:
    logger.info("Step 3) Creating a dataframe containing information about the MIDI files")
    file_df = create_file_dataframe(data_folder=settings.data_folder)
    file_df.to_csv(os.path.join(settings.data_folder, "file_dataframe.csv"))
    status["create_dataframe"] = True
    save_pipeline_status(status)
else:
    logger.info("Step 3) Dataframe has already been created, loading from disk")
    file_df = pd.read_csv(os.path.join(settings.data_folder, "file_dataframe.csv"))

num_train = len(file_df[file_df["dataset"] == "PS1"])
num_test = len(file_df[file_df["dataset"] == "PS2"])
logger.info(f"\tThere are {num_train} files in the training set")
logger.info(f"\tThere are {num_test} files in the test set")
logger.info(f"\n\nDataFrame -------------------------------------\n{file_df.head()}\n")

# ------------------------------------------------------------
# Step 4a: Generate Dataset
if not status["generate_dataset"]:
    logger.info("Step 4a) Creating frame based dataset from MIDI files")
    dataset = generate_dataset(file_df, output_folder=settings.data_folder)
    dataset.to_csv(os.path.join(settings.data_folder, "dataset.csv"), index=False)
    status["generate_dataset"] = True
    save_pipeline_status(status)
else:
    logger.info("Step 4a) Dataset has already been created, loading from disk")
    dataset = pd.read_csv(os.path.join(settings.data_folder, "dataset.csv"))

# ------------------------------------------------------------
# Step 4b: Extract 30 second frames from each MIDI file
if not status["extract_frames"]:
    logger.info("Step 4b) Extracting frames from MIDI files")
    midi_files = dataset[["file_index", "file_path"]].drop_duplicates()
    with tqdm(total=len(midi_files), desc="Extracting frames from MIDI files") as pbar:
        for row in midi_files.iterrows():
            file_index = row[1]["file_index"]
            file_path = row[1]["file_path"]
            pbar.set_postfix(file=file_path)
            # Get the piano roll for the current MIDI file
            piano_roll = generate_piano_roll(file_path, fs=100)
            frame_indices = dataset[dataset["file_index"] == file_index][
                ["frame_start", "frame_end"]
            ].values
            frame_indices = [(int(start), int(end)) for start, end in frame_indices]
            frames = extract_frames(piano_roll, frame_indices)
            frame_ids = dataset[dataset["file_index"] == file_index]["frame_id"].values
            save_frames_to_disk(settings.data_folder, frame_ids, frames)
            pbar.update(1)

    status["extract_frames"] = True
    save_pipeline_status(status)
else:
    logger.info("Step 4b) Frames have already been extracted, skipping this step")

# ------------------------------------------------------------
# 5) Extract features from each frame
if not status["extract_features"]:
    logger.info("Step 5) Extracting features from each frame")
    # Load the dataset
    dataset = pd.read_csv(os.path.join(settings.data_folder, "dataset.csv"))
    features_dataset = dataset.copy()

    # Create the featureExtractor object
    # Register the feature functions
    feature_functions = [
        pitch_based_features,
        velocity_based_features,
        temporal_features,
        harmonic_features,
        frequency_based_features,
        higher_level_features,
    ]
    featureExtractor = FeatureExtractor(feature_functions)

    # Apply the feature extraction to the dataset
    # Define a function to extract features for each row in the dataset
    def _extract_features_for_row(frame_id):
        frame_path = os.path.join(settings.data_folder, f"frames/{frame_id}.npy")
        frame = np.load(frame_path)
        result = featureExtractor.extract_all_features(frame)
        result["frame_id"] = frame_id
        return result

    # Apply and expand the features
    features_df = pd.DataFrame.from_records(
        features_dataset["frame_id"].apply(_extract_features_for_row)
    )

    # Concatenate the original dataset with the new features
    features_dataset = pd.concat([features_dataset, features_df.drop(columns=["frame_id"])], axis=1)

    # Save the dataset with features to disk
    features_dataset.to_csv(os.path.join(settings.data_folder, "features_dataset.csv"), index=False)

    status["extract_features"] = True
    save_pipeline_status(status)
else:
    logger.info("Step 5) Features have already been extracted, skipping this step")

logger.info("Pipeline complete")
