from app import create_app

from settings import settings
from classical_composer.misc import logger

if __name__ == "__main__":
    # Print final configuration
    logger.info(f"Current Configuration:")

    # Create and run the app
    app = create_app()
    app.run()






