"""Entry point for the module.

Create and runs the app.
"""

from app import create_app
from classical_composer.misc import logger

if __name__ == "__main__":
    # Create and run the app
    logger.info("Initializing classical composer")
    app = create_app()
    app.run()
