# This is needed to keep the code functioning with imports correctly and settings
import os
import sys

# Add the root directory (parent of scripts) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------
import json
import pandas as pd
from tqdm import tqdm

from classical_composer.data import (
    create_file_dataframe,
    extract_data,
    generate_dataset,
    generate_piano_roll,
    save_frames_to_disk,
)
from classical_composer.misc.logging import logger
from classical_composer.frames import extract_frames
from settings import settings

"""
This script generates the dataset for the classical composer challenge. The steps are as follows:

1) Unzip the original datasaet
2) move all .mid files from PS2 root to unknown folder
3) Create a dataframe containing information about MIDI files, including file paths, composers, and
file indices.
4) Extract 30 second frames from each MIDI file
5) Extract features from each frame
6) Split the dataframe into train valuation and test sets
7) Save the datasets to disk for later use
"""

# Load or initialize the status file for the pipeline
STATUS_FILE = os.path.join(settings.data_folder, "pipeline_status.json")


def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    return {
        "extract_data": False,
        "create_dataframe": False,
        "generate_dataset": False,
        "extract_frames": False,
        "extract_features": False,
    }


def save_status(status):
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=4)


# Load the current status
status = load_status()

# ------------------------------------------------------------
# 1-2) Unzip the original datasaet
if not status["extract_data"]:
    logger.info("Step 1,2) Extracting data from the original dataset")
    extract_data(source_data_file=settings.source_data_file, output_folder=settings.data_folder)
    logger.info("Data extraction complete")
    status["extract_data"] = True
    save_status(status)
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
    save_status(status)
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
    save_status(status)
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
    save_status(status)
else:
    logger.info("Step 4b) Frames have already been extracted, skipping this step")

# ------------------------------------------------------------
# 5) Extract features from each frame
