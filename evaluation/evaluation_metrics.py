import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np


def load_and_split(csv_path: str, target_col: str, test_size: float = 0.2, val_size: float = 0.25):
    """
    Load a CSV, drop NaN rows, and split into train/val/test sets.

    Args:
        csv_path: path to the CSV file
        target_col: name of the label column
        test_size: fraction of total data held out for test (default 0.2)
        val_size: fraction of training data held out for validation (default 0.25 → 20% of total)

    Returns:
        x_train, x_val, x_test, y_train, y_val, y_test
    """

    # Load dataset from CSV file
    data = pd.read_csv(csv_path)

    # Remove rows with missing values to ensure clean training data
    data.dropna(inplace=True)

    # Separate features (x) and target variable (y)
    x = data.drop(columns=[target_col])
    y = data[target_col]

    # First split: separate out test set from the full dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=42
    )

    # Second split: take a portion of training data to use as validation set
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=val_size, random_state=42
    )

    # Final output: train, validation, and test splits
    return x_train, x_val, x_test, y_train, y_val, y_test


def compute_regression_metrics(y_true, y_pred) -> dict:
    """
    Compute standard regression metrics.

    Returns:
        dict with MAE, RMSE, and R2 score
    """

    return {
        # Mean Absolute Error → average absolute difference between predictions and actual values
        "MAE": mean_absolute_error(y_true, y_pred),

        # Root Mean Squared Error → penalizes larger errors more heavily
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),

        # R² Score → measures how well the model explains variance in the data (1 = perfect fit)
        "R2": r2_score(y_true, y_pred)
    }