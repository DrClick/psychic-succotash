"""
Classical Composer project.

This project is a pipeline for analyzing classical music MIDI files.

"""

import subprocess

from settings import settings
from classical_composer.misc.logging import logger


class App:
    """Classical Composer app."""

    def __init__(self):
        logger.info("Initializing classical composer")

    def run(self):
        """Run the classical composer pipeline."""

        if settings.get("start_server", False):
            logger.info("Starting server")
            subprocess.run(["hatch", "run", "python", "src/classical_composer/api/server.py"])
        elif settings.get("generate_dataset", False):
            logger.info("Generating dataset")
            subprocess.run(["hatch", "run", "python", "scripts/generate_dataset.py"])
        elif settings.get("train_kmeans", False):
            logger.info("Training KMeans")
            subprocess.run(["hatch", "run", "python", "scripts/train_kmeans.py"])
        elif settings.get("train_cnn", False):
            logger.info("Training CNN")
            subprocess.run(["hatch", "run", "python", "scripts/train_cnn.py"])
        else:
            print("Please choose only one of the following modes:")
            print("1. Start server --start-server")
            print("2. Generate dataset --generate-dataset")
            print("3. Train KMeans --train-kmeans")
            print("4. Train CNN --train-cnn")


def create_app():
    """Create the app."""
    return App()
