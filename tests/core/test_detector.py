"""Tests for PlayerDetector."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch

from padelvision.core.detector import PlayerDetector
from padelvision.types import BBox, BoundingBox, PlayerDetection


class TestPlayerDetectorInit:
    def test_resolve_device_auto_mps(self) -> None:
        import torch

        expected = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        result = PlayerDetector._resolve_device("auto")
        assert result == expected

    def test_resolve_device_cpu(self) -> None:
        assert PlayerDetector._resolve_device("cpu") == "cpu"

    def test_resolve_device_mps(self) -> None:
        assert PlayerDetector._resolve_device("mps") == "mps"


class TestCourtFiltering:
    def test_is_in_court_inside(self) -> None:
        court_roi = BoundingBox(x=100, y=50, width=1100, height=620)
        detection = PlayerDetection(
            track_id=1,
            bbox=BBox(x1=200, y1=100, x2=400, y2=500),
            confidence=0.9,
        )
        assert PlayerDetector._is_in_court(detection, court_roi)

    def test_is_in_court_outside(self) -> None:
        court_roi = BoundingBox(x=100, y=50, width=300, height=200)
        detection = PlayerDetection(
            track_id=1,
            bbox=BBox(x1=800, y1=600, x2=1000, y2=800),
            confidence=0.9,
        )
        assert not PlayerDetector._is_in_court(detection, court_roi)


class TestBatchTracking:
    def test_detect_frames_uses_tracker_ids(self) -> None:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        fake_box = SimpleNamespace(
            id=torch.tensor([7]),
            xyxy=torch.tensor([[1.0, 2.0, 10.0, 20.0]]),
            conf=torch.tensor([0.95]),
        )
        fake_result = SimpleNamespace(boxes=[fake_box])

        class FakeModel:
            def __init__(self) -> None:
                self.track_called = False

            def track(self, **kwargs):
                self.track_called = True
                return [fake_result]

        detector = PlayerDetector.__new__(PlayerDetector)
        detector._model = cast(Any, FakeModel())
        detector._device = "cpu"
        detector._conf = 0.45
        detector._iou = 0.5

        detections = detector.detect_frames([frame])

        assert detector._model.track_called is True
        assert len(detections) == 1
        assert detections[0][0].track_id == 7


class TestTeamAssignment:
    def test_assign_teams_two_players(self) -> None:
        tracks = {
            1: [PlayerDetection(track_id=1, bbox=BBox(x1=100, y1=200, x2=300, y2=500), confidence=0.9)],
            2: [PlayerDetection(track_id=2, bbox=BBox(x1=500, y1=200, x2=700, y2=500), confidence=0.88)],
        }
        result = PlayerDetector._assign_teams(tracks)
        assert result[1][0].team == 0  # left
        assert result[2][0].team == 1  # right

    def test_assign_teams_four_players(self) -> None:
        tracks = {
            1: [PlayerDetection(track_id=1, bbox=BBox(x1=100, y1=200, x2=300, y2=500), confidence=0.9)],
            2: [PlayerDetection(track_id=2, bbox=BBox(x1=500, y1=200, x2=700, y2=500), confidence=0.88)],
            3: [PlayerDetection(track_id=3, bbox=BBox(x1=150, y1=220, x2=320, y2=520), confidence=0.75)],
            4: [PlayerDetection(track_id=4, bbox=BBox(x1=550, y1=210, x2=720, y2=510), confidence=0.82)],
        }
        result = PlayerDetector._assign_teams(tracks)
        teams = {tid: result[tid][0].team for tid in result}
        assert teams[1] in {0, 1}
        assert teams[2] in {0, 1}

    def test_assign_teams_single_player(self) -> None:
        tracks = {
            1: [PlayerDetection(track_id=1, bbox=BBox(x1=100, y1=200, x2=300, y2=500), confidence=0.9)],
        }
        result = PlayerDetector._assign_teams(tracks)
        assert result[1][0].team == 0

    def test_assign_teams_empty(self) -> None:
        result = PlayerDetector._assign_teams({})
        assert result == {}
