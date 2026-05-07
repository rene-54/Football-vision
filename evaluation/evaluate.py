import os
from ultralytics import YOLO


def evaluate_model(home_dir: str, weights_path: str = None) -> None:
    """
    Evaluate a trained YOLO model on the validation split.

    Args:
        home_dir: project root directory containing data.yaml
        weights_path: path to model weights; defaults to best.pt from last training run
    """

    # If no weights are provided, use the default path from the last training run
    if weights_path is None:
        weights_path = os.path.join(
            home_dir, "runs", "detect", "train", "weights", "best.pt"
        )

    # Ensure the model weights exist before trying to load them
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found at: {weights_path}")

    # Load the trained YOLO model using the specified weights
    model = YOLO(weights_path)

    # Run validation on the dataset defined in data.yaml
    # imgsz=1280 ensures higher resolution evaluation for better accuracy
    results = model.val(
        data=os.path.join(home_dir, "data.yaml"),
        imgsz=1280
    )

    return results  # returns metrics like mAP, precision, recall


if __name__ == "__main__":
    # Use current working directory as project root
    home_dir = os.getcwd()

    # Run evaluation using default weights
    evaluate_model(home_dir)