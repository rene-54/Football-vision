import os
from ultralytics import YOLO
from roboflow import Roboflow


def get_roboflow_model(model_id: str, api_key: str):
    """
    Loads a trained model from Roboflow using a project/version identifier.

    This is used to access hosted models without manually downloading weights.
    """

    # Initialize Roboflow API connection
    rf = Roboflow(api_key=api_key)

    # Split "project/version" format into components
    project_name, version = model_id.split("/")

    # Access project and retrieve the requested model version
    project = rf.workspace().project(project_name)
    return project.version(int(version)).model


def load_yolo_model(model_path: str) -> YOLO:
    """
    Loads a local YOLO model from disk.

    Ensures the model file exists before initialization.
    """

    # Validate that the model weights file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")

    # Load YOLO model from .pt weights file
    return YOLO(model_path)