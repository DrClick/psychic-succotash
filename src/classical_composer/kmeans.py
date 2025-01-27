from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import classification_report, silhouette_score
import pandas as pd
from settings import settings
from classical_composer.misc import logger
from typing import Optional


def normalize_features(
    df: pd.DataFrame, feature_columns: List[str]
) -> Tuple[np.ndarray, StandardScaler]:
    """Normalize features using StandardScaler.

    Args
    ----
        df: DataFrame containing features.
        feature_columns: List of feature columns to normalize.

    Returns
    -------
        normalized: Normalized feature matrix.
        scaler: Fitted StandardScaler object.
    """

    logger.info(f"Normalizing features: {feature_columns}")
    scaler = StandardScaler()
    normalized = scaler.fit_transform(df[feature_columns])
    return normalized, scaler


def train_kmeans_with_known_composers(
    X_train: np.ndarray, y_train: pd.Series, n_clusters: int
) -> Tuple[KMeans, dict, float]:
    """Train KMeans model and map clusters to composers.

    Args
    ----
        X_train: Training feature matrix.
        y_train: Training labels (actual composers).
        n_clusters: Number of clusters.

    Returns
    -------
        kmeans: Trained KMeans model.
        cluster_mapping: Mapping of clusters to composer labels.
        silhouette: Silhouette score for the clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_train)

    # Calculate Silhouette Score
    silhouette = silhouette_score(X_train, kmeans.labels_)

    # Map clusters to composers
    cluster_mapping = {}
    for cluster in range(n_clusters):
        labels_in_cluster = y_train.iloc[kmeans.labels_ == cluster]
        if len(labels_in_cluster) > 0:
            dominant_composer = labels_in_cluster.mode()[0]
            cluster_mapping[cluster] = dominant_composer

    return kmeans, cluster_mapping, silhouette


# 3. Classify Validation/Test Set
def classify_with_kmeans(
    kmeans: KMeans, cluster_mapping: dict, X: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Classify samples based on the closest cluster centroid.

    Args
    ----
        kmeans: Trained KMeans model.
        cluster_mapping: Mapping of clusters to composer labels.
        X: Feature matrix to classify.

    Returns
    -------
        predictions: List of predicted composers based on closest cluster.
        distances: Distance logits for each sample to all cluster centroids.
    """
    distances = kmeans.transform(X)  # Distance to each centroid
    closest_clusters = np.argmin(distances, axis=1)  # Index of closest cluster

    # Map clusters to composers
    predictions = [cluster_mapping.get(cluster, "unknown") for cluster in closest_clusters]

    return predictions, distances


# 4. Visualize Clusters
def visualize_clusters(
    kmeans: KMeans, X: np.ndarray, y: List[str], title: str, pca: PCA = None, to_disk: bool = False
) -> PCA:
    """Visualize clusters using PCA.

    Args
    ----
        kmeans: Trained KMeans model.
        X: Feature matrix.
        y: True labels.
        title: Plot title.
        pca: PCA model (optional).
        to_disk: Save plot to disk (default=False).

    Returns
    -------
        pca: Trained PCA model.
    """

    if pca is None:
        pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X)
    cluster_centers = pca.transform(kmeans.cluster_centers_)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=reduced_features[:, 0], y=reduced_features[:, 1], hue=y, palette="tab10", legend="full"
    )
    plt.scatter(
        cluster_centers[:, 0],
        cluster_centers[:, 1],
        c="black",
        s=200,
        label="Centroids",
        marker="X",
    )
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()

    if to_disk:
        plt.savefig(f"{settings.data_folder}/{title}.png")
        plt.close()
    else:
        plt.show()

    return pca


# 5. Calculate a threshold based on the mean distance the centroids in the training set
def calculate_threshold(kmeans: KMeans, X_train: np.ndarray, std_multiplier: float = 1.0) -> float:
    """
    Calculate a threshold based on the mean distance to centroids in the training set.

    Args
    ----
        kmeans: Trained KMeans model.
        X_train: Training feature matrix.
        std_multiplier: Multiplier for the standard deviation (default=1.0).

    Returns
    -------
        threshold: Calculated threshold for "unknown" classification.
    """

    # Compute distances of training samples to centroids
    distances = kmeans.transform(X_train)
    closest_distances = distances.min(axis=1)  # Distance to the closest centroid

    # Calculate mean and standard deviation of distances
    mean_distance = closest_distances.mean()
    std_distance = closest_distances.std()

    # Set threshold
    threshold = mean_distance + std_multiplier * std_distance
    return float(threshold)


# 6. Evaluate and Test K-Means
def evaluate_and_test_kmeans(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    n_clusters: int,
    threshold: Optional[float] = None,
) -> Tuple[KMeans, dict, pd.DataFrame, pd.DataFrame, float]:
    """Evaluate and test the K-Means clustering model.

    Args
    ----
        train_df: Training dataset with features and labels.
        validation_df: Validation dataset with features and labels.
        test_df: Test dataset with features and labels.
        feature_columns: List of feature columns to use for clustering.
        n_clusters: Number of clusters for KMeans.
        threshold: Distance threshold for identifying unknown samples (optional).

    Returns
    -------
        kmeans: Trained KMeans model.
        cluster_mapping: Mapping of clusters to composer labels.
        validation_df: Updated validation dataset with predictions.
        test_df: Updated test dataset with predictions and "unknown" classification.
        threshold: Calculated or provided threshold for unknown classification.
    """
    # Normalize Features
    X_train, scaler = normalize_features(train_df, feature_columns)
    y_train = train_df["composer"]

    X_val = scaler.transform(validation_df[feature_columns])
    y_val = validation_df["composer"]

    X_test = scaler.transform(test_df[feature_columns])

    # Train K-Means and Calculate Silhouette Score
    kmeans, cluster_mapping, silhouette = train_kmeans_with_known_composers(
        X_train, y_train, n_clusters
    )
    print(f"Silhouette Score for {n_clusters} clusters: {silhouette:.4f}")

    # Visualize Known Composers (Training Set)
    pca_model = visualize_clusters(
        kmeans, X_train, y_train, title="K-Means Clusters (Training Set)", pca=None, to_disk=True
    )

    # Evaluate on Validation Set
    val_predictions, val_distances = classify_with_kmeans(kmeans, cluster_mapping, X_val)
    validation_df["predicted_composer"] = val_predictions

    # Visualize Validation Clusters
    visualize_clusters(
        kmeans,
        X_val,
        val_predictions,
        title="Validation Set Clustering",
        pca=pca_model,
        to_disk=True,
    )

    print("\n\n------------------------------------------------------------")
    print("Validation Results:")
    # Define all possible composers
    all_composers = ["Bach", "Beethoven", "Brahms", "Schubert"]
    # Create the Validation Results table
    validation_results = (
        validation_df.groupby(["composer", "predicted_composer"]).size().unstack(fill_value=0)
    )
    # Reindex to ensure all composers are included
    validation_results = validation_results.reindex(
        index=all_composers, columns=all_composers, fill_value=0
    )
    print(validation_results)

    # Validation Metrics
    print("\n\nValidation Classification Report:")
    print(classification_report(y_val, val_predictions, zero_division=0))

    # Test the Unknown Set (Post-Processing)
    test_predictions, test_distances = classify_with_kmeans(kmeans, cluster_mapping, X_test)
    test_df.loc[:, "predicted_composer"] = test_predictions

    if threshold is None:
        threshold = calculate_threshold(kmeans, X_train, std_multiplier=1.5)

    closest_distances = test_distances.min(axis=1)
    test_df.loc[:, "is_unknown"] = closest_distances > threshold

    print("\n\n------------------------------------------------------------")
    print("\nTest Results:")
    print(test_df["predicted_composer"].value_counts())

    if threshold is not None:
        print("Unknown Classification Results:")
        print(test_df["is_unknown"].value_counts())

    # Visualize Test Clusters
    visualize_clusters(
        kmeans, X_test, test_predictions, title="Test Set Clustering", pca=pca_model, to_disk=True
    )

    return kmeans, scaler, cluster_mapping, validation_df, test_df, threshold
