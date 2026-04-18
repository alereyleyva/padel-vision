"""Tests for PoseEstimator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from padelvision.core.pose_estimator import PoseEstimator


class TestPoseEstimatorInit:
    def test_valid_complexities(self) -> None:
        for complexity in ["lite", "full", "heavy"]:
            est = PoseEstimator(model_complexity=complexity)
            assert est._model_complexity_name == complexity

    def test_invalid_complexity(self) -> None:
        with pytest.raises(ValueError, match="Invalid model complexity"):
            PoseEstimator(model_complexity="invalid")

    def test_custom_confidence(self) -> None:
        est = PoseEstimator(min_detection_confidence=0.7, min_tracking_confidence=0.6)
        assert est._min_detection_confidence == 0.7
        assert est._min_tracking_confidence == 0.6


class TestPoseEstimatorEstimate:
    def test_estimate_with_no_detections(self, sample_frame: np.ndarray) -> None:
        est = PoseEstimator.__new__(PoseEstimator)
        est._landmarker = MagicMock()
        est._initialized = True
        est._model_complexity_name = "full"
        est._min_detection_confidence = 0.5
        est._min_tracking_confidence = 0.5
        est._model_dir = Path("models/mediapipe")

        result = est.estimate(sample_frame, detections=[], frame_idx=0)
        assert result == []

    def test_estimate_batch_length_mismatch(self, sample_frames: list[np.ndarray]) -> None:
        est = PoseEstimator.__new__(PoseEstimator)
        est._initialized = True

        with pytest.raises(ValueError, match="Number of frames"):
            est.estimate_batch(sample_frames, [[]])
