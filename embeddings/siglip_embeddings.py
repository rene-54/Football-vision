import os

# Limit number of CPU threads used by numerical libraries (prevents overuse of resources)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def _extract_color_histogram(crop_bgr, bins=32):
    h, w = crop_bgr.shape[:2]

    # Focus only on the upper body (jersey area) to avoid noise from shorts/field
    torso = crop_bgr[:int(h * 0.66), :]

    # Convert from BGR (OpenCV default) to HSV (better for color separation)
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # Compute histograms for each HSV channel
    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()

    # Combine all channel histograms into one feature vector
    hist = np.concatenate([hist_h, hist_s, hist_v])

    # Normalize histogram so it represents proportions instead of raw counts
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist


def _get_features(crops):
    features = []
    for crop in crops:
        # Skip invalid or empty images
        if crop is None or crop.size == 0:
            continue
        features.append(_extract_color_histogram(crop))
    return np.array(features)


def fit_team_classifier(crops):
    # Convert all image crops into feature vectors
    features = _get_features(crops)

    # Standardize features (mean=0, std=1) for better clustering performance
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Use KMeans to group players into 2 clusters (assumes 2 teams)
    clustering_model = KMeans(n_clusters=2, random_state=42, n_init=10)
    clustering_model.fit(features_scaled)
    return scaler, clustering_model


def predict_team(crops, scaler, clustering_model):
    # Handle edge case where no crops are provided
    if len(crops) == 0:
        return np.array([], dtype=int)

    # Extract and scale features using the previously fitted scaler
    features = _get_features(crops)
    features_scaled = scaler.transform(features)

    # Predict cluster (team) for each crop
    return clustering_model.predict(features_scaled).astype(int)