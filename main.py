"""Entry point for the module.

Create and runs the app.
"""

import argparse
from app import create_app
from classical_composer.misc import logger
from settings import settings


def main():
    logger.info("Initializing classical composer")

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-server", action="store_true", help="Start the server")
    parser.add_argument("--generate-dataset", action="store_true", help="Generate dataset")
    parser.add_argument("--train-kmeans", action="store_true", help="Train KMeans")
    parser.add_argument("--train-cnn", action="store_true", help="Train CNN")
    args = parser.parse_args()

    # Dynamically update settings based on CLI args
    settings.set("start_server", args.start_server)
    settings.set("generate_dataset", args.generate_dataset)
    settings.set("train_kmeans", args.train_kmeans)
    settings.set("train_cnn", args.train_cnn)

    app = create_app()
    app.run()


if __name__ == "__main__":
    # Create and run the app
    main()
