"""Pose estimator — MediaPipe Pose for 33-keypoint body pose estimation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from padelvision.types import Keypoint, PlayerDetection, PoseLandmarks

logger = logging.getLogger(__name__)

MODEL_COMPLEXITY_MAP = {
    "lite": 0,
    "full": 1,
    "heavy": 2,
}

MEDIAPIPE_KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


class PoseEstimator:
    """Estimates human pose using MediaPipe Pose.

    Supports three model complexities (lite/full/heavy) and processes
    detected player bounding boxes to produce 33 keypoints per person.
    """

    def __init__(
        self,
        model_complexity: str = "full",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_dir: str | Path = "models/mediapipe",
    ) -> None:
        if model_complexity not in MODEL_COMPLEXITY_MAP:
            raise ValueError(
                f"Invalid model complexity '{model_complexity}'. Choose from: {list(MODEL_COMPLEXITY_MAP.keys())}"
            )

        self._model_complexity_name = model_complexity
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._model_dir = Path(model_dir)
        self._landmarker = None
        self._initialized = False

    def _initialize(self) -> None:
        if self._initialized:
            return

        import mediapipe as mp

        model_suffix = MODEL_COMPLEXITY_MAP[self._model_complexity_name]
        model_files = {0: "pose_landmarker_lite.task", 1: "pose_landmarker_full.task", 2: "pose_landmarker_heavy.task"}
        model_file = model_files[model_suffix]
        model_path = self._model_dir / model_file

        if not model_path.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {model_path}")

        # MediaPipe API uses PascalCase — these are factory classes, not local variables
        BaseOptions = mp.tasks.BaseOptions  # noqa: N806
        PoseLandmarker = mp.tasks.vision.PoseLandmarker  # noqa: N806
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions  # noqa: N806
        VisionRunningMode = mp.tasks.vision.RunningMode  # noqa: N806

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=VisionRunningMode.IMAGE,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )

        self._landmarker = PoseLandmarker.create_from_options(options)
        self._initialized = True

        logger.info(
            f"Initialized MediaPipe Pose ({self._model_complexity_name}) "
            f"detection_conf={self._min_detection_confidence} "
            f"tracking_conf={self._min_tracking_confidence}"
        )

    def estimate(
        self,
        image: np.ndarray,
        detections: list[PlayerDetection],
        frame_idx: int = 0,
    ) -> list[PoseLandmarks]:
        """Estimate poses for detected players in a single frame.

        Crops each player's bounding box before inference for better accuracy.
        """
        self._initialize()

        if not detections:
            return []

        results: list[PoseLandmarks] = []

        for detection in detections:
            pose = self._estimate_single(image, detection, frame_idx)
            if pose is not None:
                results.append(pose)

        return results

    def _estimate_single(
        self,
        image: np.ndarray,
        detection: PlayerDetection,
        frame_idx: int,
    ) -> PoseLandmarks | None:
        import mediapipe as mp

        h, w = image.shape[:2]
        bbox = detection.bbox

        x1 = max(0, int(bbox.x1) - 10)
        y1 = max(0, int(bbox.y1) - 10)
        x2 = min(w, int(bbox.x2) + 10)
        y2 = min(h, int(bbox.y2) + 10)

        crop_h = y2 - y1
        crop_w = x2 - x1
        if crop_h < 10 or crop_w < 10:
            return None

        crop = image[y1:y2, x1:x2]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

        if self._landmarker is None:
            return None
        try:
            result = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.warning(f"MediaPipe detection failed for track {detection.track_id}: {e}")
            return None

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        keypoints: list[Keypoint] = []

        for lm in landmarks:
            abs_x = x1 + lm.x * crop_w if crop_w > 0 else x1
            abs_y = y1 + lm.y * crop_h if crop_h > 0 else y1
            keypoints.append(
                Keypoint(
                    x=float(abs_x),
                    y=float(abs_y),
                    visibility=float(lm.visibility) if hasattr(lm, "visibility") else 1.0,
                )
            )

        return PoseLandmarks(keypoints=keypoints, frame_idx=frame_idx)

    def estimate_batch(
        self,
        frames: Sequence[np.ndarray],
        detections_per_frame: list[list[PlayerDetection]],
    ) -> list[list[PoseLandmarks]]:
        """Estimate poses for all detected players across a batch of frames."""
        self._initialize()

        if len(frames) != len(detections_per_frame):
            raise ValueError(
                f"Number of frames ({len(frames)}) doesn't match "
                f"number of detection lists ({len(detections_per_frame)})"
            )

        all_poses: list[list[PoseLandmarks]] = []

        for frame_idx, (frame, detections) in enumerate(zip(frames, detections_per_frame)):
            poses = self.estimate(frame, detections, frame_idx=frame_idx)
            all_poses.append(poses)

        return all_poses

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
            self._initialized = False

    def __enter__(self) -> PoseEstimator:
        self._initialize()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
