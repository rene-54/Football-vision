import os
from ultralytics import YOLO


def train_pose_model(home_dir: str) -> None:
    """
    Fine-tunes a YOLOv8 pose model for player pose estimation.

    The pipeline downloads a labeled dataset from Roboflow and trains a
    pretrained pose model on football player data.
    """

    # Ensure dataset directory exists
    datasets_path = os.path.join(home_dir, "datasets")
    os.makedirs(datasets_path, exist_ok=True)

    # Load API key required for Roboflow dataset access
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY environment variable is not set.")

    # Download football player dataset from Roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(20)
    dataset = version.download("yolov8", location=datasets_path)

    # Load pretrained YOLOv8 pose model
    model = YOLO("yolov8x-pose.pt")

    # Train model on the downloaded dataset
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
    # Execute training from project root
    home_dir = os.getcwd()
    train_pose_model(home_dir)