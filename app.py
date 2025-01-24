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


def create_app():
    """Create the app."""
    return App()
