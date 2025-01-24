"""
Provides functionality for unified logging.

Includes:
- logger - the default logger to use in the project

"""

import logging

from classical_composer.misc.command_line import generate_intro


class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored log levels."""

    # ANSI escape codes for colors
    COLORS = {
        "DEBUG": "\033[0;34m",  # Blue
        "INFO": "\033[0;32m",  # Green
        "WARNING": "\033[0;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
        "CRITICAL": "\033[1;41m",  # White on Red background
    }
    RESET = "\033[0m"  # Reset color to default

    def format(self, record):
        """Apply color to the log level name."""
        log_color = self.COLORS.get(record.levelname, self.RESET)  # Default to no color
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"  # Colorize level name
        return super().format(record)


def _classical_composer_logger(name="classical_composer_logger"):
    """Set up a logger with colored output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all logs

    # Create console handler with the custom formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Define the log format
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = ColoredFormatter(log_format)

    # Apply the formatter to the handler
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


print(generate_intro())
logger = _classical_composer_logger()
