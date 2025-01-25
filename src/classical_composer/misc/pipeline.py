"""Managing and processing data in the Classical Composer project.

Includes:
- Functions to get and set pipeline status.
"""

import os
import json

from settings import settings

STATUS_FILE = os.path.join(settings.data_folder, "pipeline_status.json")


def load_pipeline_status() -> dict[str, bool]:
    """Load the pipeline status from disk."""
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            return {k: bool(v) for k, v in json.load(f).items()}
    return {
        "extract_data": False,
        "create_dataframe": False,
        "generate_dataset": False,
        "extract_frames": False,
        "extract_features": False,
        "train_kmeans": False,
    }


def save_pipeline_status(status: dict) -> None:
    """Save the pipeline status to disk.

    Args
    ----
        status: dict, pipeline status.

    """
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f, indent=4)
