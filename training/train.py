import os
from ultralytics import YOLO


def train_player_detector(home_dir: str) -> None:
    """
    Fine-tunes a YOLOv8 object detection model for football players.

    The model is trained on a custom dataset defined in data.yaml.
    """

    # Load pretrained YOLOv8 detection model
    model = YOLO("yolov8x.pt")

    # Train model on player detection dataset
    results = model.train(
        data=os.path.join(home_dir, "data.yaml"),
        batch=6,
        epochs=50,
        imgsz=1280,
        plots=True
    )

    return results


if __name__ == "__main__":
    # Run training from project root directory
    home_dir = os.getcwd()
    train_player_detector(home_dir)