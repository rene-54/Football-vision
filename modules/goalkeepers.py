import supervision as sv
import numpy as np


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Assign each goalkeeper to the nearest team based on player positions.

    The function estimates each team’s position using player locations,
    then assigns goalkeepers to the closest team centroid.
    """

    # Extract bottom-center coordinates for spatial comparison
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)

    # Split players into their respective teams
    team_0_players = players_xy[players.class_id == 0]
    team_1_players = players_xy[players.class_id == 1]

    # Estimate team centers (fallback to origin if no players detected)
    team_0_centroid = (
        team_0_players.mean(axis=0) if len(team_0_players) > 0
        else np.array([0.0, 0.0])
    )
    team_1_centroid = (
        team_1_players.mean(axis=0) if len(team_1_players) > 0
        else np.array([0.0, 0.0])
    )

    # Assign each goalkeeper to the closest team centroid
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id, dtype=int)