import os
from roboflow import Roboflow


def download_player_detection_dataset(home_dir: str) -> None:
    """
    Downloads a football player detection dataset from Roboflow
    in YOLOv8 format and stores it in the project datasets folder.

    Requires ROBOFLOW_API_KEY environment variable.
    """

    # Create datasets directory and switch working directory to it
    datasets_path = os.path.join(home_dir, "datasets")
    os.makedirs(datasets_path, exist_ok=True)
    os.chdir(datasets_path)

    # Ensure API key is available for authentication
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY environment variable is not set.")

    # Connect to Roboflow and download dataset version
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(20)
    version.download("yolov8")

    # Return to original project directory
    os.chdir(home_dir)