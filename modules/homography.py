import cv2
import numpy as np
import supervision as sv


class ViewTransformer:
    """
    Handles coordinate transformation between two perspectives using homography.

    Typically used to map points from a camera view (pixel space) to a
    top-down or field coordinate system.
    """

    def __init__(self, source: np.ndarray, target: np.ndarray):
        """
        Builds a homography matrix from corresponding points in two spaces.

        Args:
            source: reference points in the original image space
            target: matching points in the destination coordinate space
        """

        # Ensure correct numeric type for OpenCV
        source = source.astype(np.float32)
        target = target.astype(np.float32)

        # Compute transformation matrix between the two coordinate systems
        self.m, _ = cv2.findHomography(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Applies homography transformation to a set of points.

        Args:
            points: input coordinates in source space

        Returns:
            transformed coordinates in target space
        """

        # Handle empty input safely
        if len(points) == 0:
            return points

        # Convert format required by OpenCV perspective transform
        points = points.reshape(-1, 1, 2).astype(np.float32)

        # Apply homography transformation
        points = cv2.perspectiveTransform(points, self.m)

        return points.reshape(-1, 2).astype(np.float32)


def pad_keypoints(result, num_keypoints: int = 32):
    """
    Converts sparse keypoint detections into a fixed-size representation.

    Missing keypoints are filled with zeros to maintain consistent shape.
    """

    prediction = result.predictions[0]

    # Initialize full keypoint arrays (fixed size output)
    full_xy = np.zeros((num_keypoints, 2), dtype=np.float32)
    full_confidence = np.zeros(num_keypoints, dtype=np.float32)

    # Fill in detected keypoints at their respective indices
    for kp in prediction.keypoints:
        idx = kp.class_id
        if 0 <= idx < num_keypoints:
            full_xy[idx] = [kp.x, kp.y]
            full_confidence[idx] = kp.confidence

    return full_xy, full_confidence