# This is needed to keep the code functioning with imports correctly and settings
import os
import sys

# Add the root directory (parent of scripts) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# ------------------------------------------------------------

import pandas as pd
import pickle

from classical_composer.misc import logger
from classical_composer.misc import load_pipeline_status, save_pipeline_status
from classical_composer.data import stratified_group_split
from classical_composer.kmeans import evaluate_and_test_kmeans

from settings import settings


# Load the current status
status = load_pipeline_status()
DATA_DIR = settings.data_folder

if not status["generate_dataset"]:
    logger.critical(
        """Unable to proceeed, the dataset has not been generated.
        Please run the generate_dataset.py script first"""
    )
    sys.exit()

if status["train_kmeans"]:
    logger.info("The KNN model has already been trained, skipping this step")
else:
    logger.info("Starting to train the KNN model")

    logger.info("loading the dataset...")
    features_dataset = pd.read_csv(f"{DATA_DIR}/features_dataset.csv")
    logger.info(f"\n\nDataFrame -------------------------------------\n{features_dataset.head()}\n")

    # Do the train test split ensuring that the frames from a single file are either in the train or
    # validation set. Stratify by composer
    logger.info("splitting the dataset into train, validation and test sets...")
    train_df = features_dataset[features_dataset["dataset"] == "PS1"]  # Filter for training data
    train_df, validation_df = stratified_group_split(
        train_df, "file_index", "composer", test_size=0.2
    )
    test_df = features_dataset[features_dataset["dataset"] == "PS2"]  # Filter for test data
    # Print the sizes of the splits
    print(
        f"Train size: {len(train_df)}, Validation size: {len(validation_df)}",
        f"Test size: {len(test_df)}",
    )

    # These features were selected manually, based on the results of the feature analysis.
    # Grid search should be used to find the best features
    features = [
        # 'pitch_range',
        "pitch_entropy",
        "dominant_pitch",
        "avg_velocity",
        # 'velocity_variance',
        # 'silent_notes_ratio',
        # 'note_density',
        # 'mean_duration',
        # 'variance_duration',
        # 'syncopation',
        # 'interval_variance',
        # 'chord_identification',
        # 'consonance_dissonance',
        # 'spectral_centroid',
        "spectral_bandwidth",
        # 'tonality',
        # 'repetitive_patterns'
    ]
    n_clusters = 4
    threshold = None  # Will use the default method for mean distance times 1.5

    logger.info("Training the KMeans model...")
    kmeans_model, cluster_mapping, validation_results, test_results, threshold_used = (
        evaluate_and_test_kmeans(train_df, validation_df, test_df, features, n_clusters, threshold)
    )

    # Analyze "unknown" classification
    print(f"Samples with a distance greater than {threshold_used} where classified as `Unknown`")
    unknown_samples = test_results[test_results["is_unknown"]]
    print(f"Number of unknown samples: {len(unknown_samples)}")

    # Save the model and results
    logger.info("Saving the model and results...")
    run_date = pd.Timestamp.now()
    model_run = {
        "run_date": run_date,
        "model": kmeans_model,
        "cluster_mapping": cluster_mapping,
        "validation_results": validation_results,
        "test_results": test_results,
        "threshold": threshold_used,
    }
    with open(f"{DATA_DIR}/model_run_{run_date}.pkl", "wb") as f:
        pickle.dump(model_run, f)

    status["train_kmeans"] = True
    save_pipeline_status(status)
