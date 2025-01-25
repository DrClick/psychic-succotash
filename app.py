"""
Classical Composer project.

This project is a pipeline for analyzing classical music MIDI files.

"""

from classical_composer.misc.logging import logger


class App:
    """Classical Composer app."""

    def __init__(self):
        logger.info("Initializing classical composer")

    def run(self):
        """Run the classical composer pipeline."""
        logger.info("Running classical composer pipeline")

        print("\n\n\n\nPlease run > hatch run python scripts/generate_dataset.py")
        print("then > hatch run python scripts/train_kmeans.py")


def create_app():
    """Create the app."""
    return App()
