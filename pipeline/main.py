import sys
import os

# Set project root so all relative imports work consistently
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Limit CPU threading to avoid performance issues / oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JOBLIB_START_METHOD"] = "spawn"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import multiprocessing

if __name__ == "__main__":
    # Ensure safe multiprocessing behavior (especially for Windows compatibility)
    multiprocessing.set_start_method("spawn", force=True)

    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    from modules.crop_players import extract_crops
    from modules.goalkeepers import resolve_goalkeepers_team_id
    from embeddings.siglip_embeddings import fit_team_classifier, predict_team

    # Class IDs for object detection model
    BALL_ID = 0
    GOALKEEPER_ID = 1
    PLAYER_ID = 2
    REFEREE_ID = 3

    def main(video_name: str) -> None:
        """
        Full inference pipeline:
        - Load video
        - Run detection + tracking
        - Classify teams
        - Annotate and export output video
        """

        SOURCE = f"data/{video_name}.mp4"
        OUTPUT = f"data/{video_name}_output.mp4"

        if not os.path.exists(SOURCE):
            raise FileNotFoundError(f"Video not found: {SOURCE}")

        # Load trained object detection model
        model = YOLO("models/object_detection_model.pt")

        print("Extracting player crops for team classifier fitting...")
        crops = extract_crops(SOURCE, model)

        if len(crops) == 0:
            raise RuntimeError("No player crops extracted — check video/model input.")

        print(f"Fitting team classifier on {len(crops)} crops...")

        # Train simple clustering model to separate teams based on appearance
        scaler, clustering_model = fit_team_classifier(crops)

        # Visual annotators for different object types
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

        # Initialize tracker for consistent object IDs across frames
        tracker = sv.ByteTrack(minimum_matching_threshold=0.8)
        tracker.reset()

        video_info = sv.VideoInfo.from_video_path(SOURCE)

        print("Running inference loop...")

        # Process video frame-by-frame and write annotated output
        with sv.VideoSink(OUTPUT, video_info=video_info) as sink:
            for frame in sv.get_video_frames_generator(SOURCE):

                # Run object detection on current frame
                result = model(frame, conf=0.3)[0]
                detections = sv.Detections.from_ultralytics(result)

                # Extract and highlight ball detection
                ball = detections[detections.class_id == BALL_ID]
                ball.xyxy = sv.pad_boxes(xyxy=ball.xyxy, px=10)

                # Process all non-ball detections (players, refs, goalkeepers)
                others = detections[detections.class_id != BALL_ID]
                others = others.with_nms(threshold=0.5, class_agnostic=True)
                others = tracker.update_with_detections(detections=others)

                # Separate tracked objects by class
                players = others[others.class_id == PLAYER_ID]
                goalkeepers = others[others.class_id == GOALKEEPER_ID]
                referees = others[others.class_id == REFEREE_ID]

                # Predict team identity for each player based on appearance
                if len(players) > 0:
                    player_crops = [sv.crop_image(frame, xyxy) for xyxy in players.xyxy]
                    players.class_id = predict_team(player_crops, scaler, clustering_model)

                # Assign goalkeeper to nearest team based on spatial proximity
                if len(goalkeepers) > 0 and len(players) > 0:
                    goalkeepers.class_id = resolve_goalkeepers_team_id(players, goalkeepers)

                # Assign all referees to a fixed class
                if len(referees) > 0:
                    referees.class_id = np.full(len(referees), 2, dtype=int)

                # Merge all detections for visualization
                merged = sv.Detections.merge([players, goalkeepers, referees])
                merged.class_id = merged.class_id.astype(int)

                # Create tracking labels
                labels = [f"#{tid}" for tid in merged.tracker_id]

                # Apply visual annotations
                annotated = frame.copy()
                annotated = ellipse_annotator.annotate(annotated, merged)
                annotated = label_annotator.annotate(annotated, merged, labels=labels)
                annotated = triangle_annotator.annotate(annotated, ball)

                # Save frame to output video
                sink.write_frame(annotated)

        print(f"Done. Output saved to: {OUTPUT}")

    main("Football_A")