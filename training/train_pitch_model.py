import os
from ultralytics import YOLO
from modules.data_setup import download_player_detection_dataset


def train_pitch_keypoint_model(home_dir: str) -> None:
    """
    Fine-tunes a YOLOv8 pose model for soccer pitch keypoint detection.

    The function downloads the dataset from Roboflow and trains a
    keypoint detection model to identify pitch structure.
    """

    # Ensure dataset directory exists
    datasets_path = os.path.join(home_dir, "datasets")
    os.makedirs(datasets_path, exist_ok=True)

    # Retrieve API key for dataset download
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY environment variable is not set.")

    # Download football pitch keypoint dataset from Roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
    version = project.version(15)
    dataset = version.download("yolov8", location=datasets_path)

    # Load pretrained YOLOv8 pose model
    model = YOLO("yolov8x-pose.pt")

    # Train model on pitch keypoint dataset
    model.train(
        task='pose',
        data=os.path.join(dataset.location, "data.yaml"),
        batch=48,
        epochs=500,
        imgsz=640,
        mosaic=0.0,
        plots=True
    )


if __name__ == "__main__":
    # Run training from project root directory
    home_dir = os.getcwd()
    train_pitch_keypoint_model(home_dir)