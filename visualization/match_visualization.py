"""
Renders a top-down pitch visualization from a broadcast football video.

The pipeline:
- Detects players, ball, referees in camera space
- Tracks and assigns team identities
- Estimates camera-to-pitch homography using keypoints
- Projects all detections onto a top-down pitch view
- Writes an annotated pitch-map video
"""

import os
from tqdm import tqdm
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.configs.soccer import SoccerPitchConfiguration

from pipeline.inference import get_roboflow_model
from modules.homography import ViewTransformer, pad_keypoints
from modules.crop_players import extract_crops
from modules.goalkeepers import resolve_goalkeepers_team_id
from embeddings.siglip_embeddings import fit_team_classifier, predict_team

BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

CONFIG = SoccerPitchConfiguration()


def run_pitch_map_visualization(
    source_video_path: str,
    output_video_path: str,
    object_model_path: str = "models/object_detection_model.pt",
    roboflow_api_key: str = None
) -> None:
    """
    Generates a top-down tactical pitch map from a football broadcast video.

    Combines object detection, tracking, team classification, and homography
    to project real-world player positions onto a 2D pitch layout.
    """

    # Ensure API key is available for keypoint model
    if roboflow_api_key is None:
        roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
    if not roboflow_api_key:
        raise EnvironmentError("ROBOFLOW_API_KEY is not set.")

    # Load models
    object_model = YOLO(object_model_path)
    keypoint_model = get_roboflow_model(
        model_id="football-field-detection-f07vi/15",
        api_key=roboflow_api_key
    )

    # --- Train team classifier from appearance features ---
    print("Extracting crops for team classifier...")
    crops = extract_crops(source_video_path, object_model)
    print(f"Fitting classifier on {len(crops)} crops...")
    reducer, clustering_model = fit_team_classifier(crops)

    # Initialize tracking
    tracker = sv.ByteTrack()
    tracker.reset()

    video_info = sv.VideoInfo.from_video_path(source_video_path)

    # Create base pitch canvas
    pitch_frame = draw_pitch(config=CONFIG)
    pitch_h, pitch_w = pitch_frame.shape[:2]

    pitch_video_info = sv.VideoInfo(
        width=pitch_w, height=pitch_h, fps=video_info.fps
    )

    print("Rendering pitch map...")

    # Process video frame-by-frame and render pitch projection
    with sv.VideoSink(target_path=output_video_path, video_info=pitch_video_info) as sink:
        for frame in tqdm(
            sv.get_video_frames_generator(source_video_path),
            total=video_info.total_frames,
            desc="Pitch map frames"
        ):

            # --- Object detection ---
            result = object_model(frame, conf=0.3)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Handle ball separately (no tracking needed)
            ball = detections[detections.class_id == BALL_ID]
            ball.xyxy = sv.pad_boxes(xyxy=ball.xyxy, px=10)

            # Track remaining objects
            others = detections[detections.class_id != BALL_ID]
            others = others.with_nms(threshold=0.5, class_agnostic=True)
            others = tracker.update_with_detections(detections=others)

            players = others[others.class_id == PLAYER_ID]
            goalkeepers = others[others.class_id == GOALKEEPER_ID]
            referees = others[others.class_id == REFEREE_ID]

            # Assign teams based on appearance clustering
            if len(players) > 0:
                player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                players.class_id = predict_team(
                    player_crops, reducer, clustering_model
                )

            # Assign goalkeepers to nearest team cluster
            if len(goalkeepers) > 0 and len(players) > 0:
                goalkeepers.class_id = resolve_goalkeepers_team_id(
                    players, goalkeepers
                )

            # --- Estimate homography using pitch keypoints ---
            kp_result = keypoint_model.infer(frame, confidence=0.3)[0]

            # Skip frame if keypoints are not reliable
            if len(kp_result.predictions) == 0:
                sink.write_frame(draw_pitch(config=CONFIG))
                continue

            full_xy, full_confidence = pad_keypoints(kp_result)
            mask = full_confidence > 0.5

            if mask.sum() < 4:
                sink.write_frame(draw_pitch(config=CONFIG))
                continue

            # Compute transformation between camera and pitch space
            frame_ref_pts = full_xy[mask]
            pitch_ref_pts = np.array(CONFIG.vertices)[mask]

            transformer = ViewTransformer(
                source=frame_ref_pts,
                target=pitch_ref_pts
            )

            # --- Render pitch visualization ---
            pitch = draw_pitch(config=CONFIG)

            # Project ball onto pitch
            if len(ball) > 0:
                frame_ball_xy = ball.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_ball_xy = transformer.transform_points(frame_ball_xy)
                pitch = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_ball_xy,
                    face_color=sv.Color.WHITE,
                    edge_color=sv.Color.BLACK,
                    radius=10,
                    pitch=pitch
                )

            # Project players onto pitch (split by team)
            if len(players) > 0:
                players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_players_xy = transformer.transform_points(players_xy)

                pitch = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players.class_id == 0],
                    face_color=sv.Color.from_hex("#00BFFF"),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch
                )
                pitch = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[players.class_id == 1],
                    face_color=sv.Color.from_hex("#FF1493"),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch
                )

            # Project referees onto pitch
            if len(referees) > 0:
                referees_xy = referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
                pitch_referees_xy = transformer.transform_points(referees_xy)

                pitch = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_referees_xy,
                    face_color=sv.Color.from_hex("#FFD700"),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=pitch
                )

            # Write final pitch frame
            sink.write_frame(pitch)

    print(f"Pitch map saved to: {output_video_path}")