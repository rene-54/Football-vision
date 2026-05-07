from tqdm import tqdm
import supervision as sv
from ultralytics import YOLO

STRIDE = 30
PLAYER_ID = 2


def extract_crops(source_video_path: str, model: YOLO) -> list:
    """
    Extract player image crops from a video using a YOLO detection model.

    The process samples video frames at a fixed stride, runs object detection,
    filters detections to keep only players, and crops those regions from frames.
    """

    crops = []

    # Generate video frames at a reduced sampling rate for efficiency
    frame_generator = sv.get_video_frames_generator(
        source_video_path, stride=STRIDE
    )

    for frame in tqdm(frame_generator, desc="Extracting player crops"):

        result = model(frame, conf=0.3)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Clean detections and keep only relevant class (players)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == PLAYER_ID]

        # Crop detected player regions from the frame
        crops += [
            sv.crop_image(frame, xyxy)
            for xyxy in detections.xyxy
        ]

    return crops