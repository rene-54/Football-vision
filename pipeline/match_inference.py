"""
Standalone script for full football match analysis pipeline.

The script:
- extracts player crops from video
- trains a lightweight team classifier
- runs frame-by-frame detection + tracking
- assigns teams, goalkeepers, referees, and ball
- exports an annotated output video
"""

from tqdm import tqdm
import supervision as sv
import numpy as np
import torch

from pipeline.inference import load_yolo_model
from embeddings.siglip_embeddings import fit_team_classifier, predict_team
from modules.crop_players import extract_crops
from modules.goalkeepers import resolve_goalkeepers_team_id

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


def run_match_inference(
    source_video_path: str,
    output_video_path: str,
    model_path: str = "models/object_detection_model.pt"
) -> None:
    """
    Runs the full inference pipeline on a football match video:
    detection, tracking, team classification, and visualization.
    """

    model = load_yolo_model(model_path)

    # --- Train team classifier from extracted player appearances ---
    print("Extracting crops for team classifier...")
    crops = extract_crops(source_video_path, model)

    if len(crops) == 0:
        raise RuntimeError("No player crops extracted. Check input video or model.")

    print(f"Fitting team classifier on {len(crops)} crops...")
    reducer, clustering_model = fit_team_classifier(crops)

    # --- Visualization components ---
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )
    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=21,
        outline_thickness=1
    )

    # --- Tracking setup ---
    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(source_video_path)

    print("Running inference loop...")

    # Process video frame-by-frame and write annotated output
    with sv.VideoSink(output_video_path, video_info=video_info) as sink:
        for frame in tqdm(
            sv.get_video_frames_generator(source_video_path),
            total=video_info.total_frames,
            desc="Processing frames"
        ):

            # Run object detection
            result = model(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Handle ball separately (no tracking needed)
            ball = detections[detections.class_id == BALL_ID]
            ball.xyxy = sv.pad_boxes(xyxy=ball.xyxy, px=10)

            # Track remaining objects (players, refs, goalkeepers)
            others = detections[detections.class_id != BALL_ID]
            others = others.with_nms(threshold=0.5, class_agnostic=True)
            others = tracker.update_with_detections(detections=others)

            players = others[others.class_id == PLAYER_ID]
            goalkeepers = others[others.class_id == GOALKEEPER_ID]
            referees = others[others.class_id == REFEREE_ID]

            # Assign team IDs based on appearance model
            if len(players) > 0:
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                players.class_id = predict_team(
                    player_crops, reducer, clustering_model
                )

            # Assign goalkeepers based on proximity to team clusters
            if len(goalkeepers) > 0 and len(players) > 0:
                goalkeepers.class_id = resolve_goalkeepers_team_id(
                    players, goalkeepers
                )

            # Assign referees to neutral class
            if len(referees) > 0:
                referees.class_id = np.full(len(referees), 2, dtype=int)

            # Merge all tracked detections for visualization
            merged = sv.Detections.merge([players, goalkeepers, referees])
            merged.class_id = merged.class_id.astype(int)

            labels = [f"#{tid}" for tid in merged.tracker_id]

            # Apply visual annotations
            annotated = frame.copy()
            annotated = ellipse_annotator.annotate(annotated, merged)
            annotated = label_annotator.annotate(annotated, merged, labels=labels)
            annotated = triangle_annotator.annotate(annotated, ball)

            # Write frame to output video
            sink.write_frame(annotated)

    print(f"Done. Output saved to: {output_video_path}")