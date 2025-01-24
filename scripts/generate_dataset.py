# This is needed to keep the code functioning with imports correctly and settings
import os
import sys

# Add the root directory (parent of scripts) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------

from classical_composer.data import create_file_dataframe
from classical_composer.misc.logging import logger
from settings import settings

"""
This script generates the dataset for the classical composer challenge. The steps are as follows:

1) Unzip the original datasaet
2) move all .mid files from PS2 root to unknown folder
3) Create a dataframe containing information about MIDI files, including file paths, composers, and
file indices.
5) Extract 30 second frames from each MIDI file
6) Extract featuers from each frame
7) Split the dataframe into train valuation and test sets
8) Save the datasets to disk for later use
"""

# 1) Unzip the original datasaet
# logger.info("Extracting data from the original dataset")
# extract_data(source_data_file=settings.source_data_file, output_folder=settings.data_folder)
# logger.info("Data extraction complete")

# 2) Create a dataframe containing information about MIDI files, including file paths, composers,
# and file indices.
file_df = create_file_dataframe(data_folder=settings.data_folder)
num_train = len(file_df[file_df["dataset"] == "PS1"])
num_test = len(file_df[file_df["dataset"] == "PS2"])

logger.info(f"There are {num_train} files in the training set")
logger.info(f"There are {num_test} files in the test set")
logger.info(f"DF head-------------------------------------\n{file_df.head()}\n")
