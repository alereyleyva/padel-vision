"""Tests for shared types module."""

from __future__ import annotations

from pathlib import Path

import pytest

from padelvision.types import (
    BBox,
    BoundingBox,
    Keypoint,
    PlayerDetection,
    PlayerScore,
    PoseLandmarks,
    VideoMetadata,
)


class TestBBox:
    def test_properties(self, sample_bbox: BBox) -> None:
        assert sample_bbox.width == 300.0
        assert sample_bbox.height == 400.0
        assert sample_bbox.center_x == 250.0
        assert sample_bbox.center_y == 400.0
        assert sample_bbox.area == 120000.0

    def test_frozen(self) -> None:
        bbox = BBox(x1=0, y1=0, x2=100, y2=100)
        with pytest.raises(AttributeError):
            bbox.x1 = 50  # type: ignore[misc]

    def test_negative_area(self) -> None:
        bbox = BBox(x1=100, y1=100, x2=50, y2=50)
        assert bbox.area == 0.0


class TestBoundingBox:
    def test_as_bbox(self) -> None:
        roi = BoundingBox(x=10.0, y=20.0, width=300.0, height=400.0)
        bbox = roi.as_bbox
        assert bbox.x1 == 10.0
        assert bbox.y1 == 20.0
        assert bbox.x2 == 310.0
        assert bbox.y2 == 420.0


class TestVideoMetadata:
    def test_valid_duration(self) -> None:
        meta = VideoMetadata(
            source=Path("/test.mp4"),
            fps=25.0,
            total_frames=250,
            width=1280,
            height=720,
            duration_sec=10.0,
            codec="avc1",
        )
        assert meta.is_valid_duration

    def test_invalid_duration_too_short(self) -> None:
        meta = VideoMetadata(
            source=Path("/test.mp4"),
            fps=25.0,
            total_frames=50,
            width=1280,
            height=720,
            duration_sec=2.0,
            codec="avc1",
        )
        assert not meta.is_valid_duration

    def test_invalid_duration_too_long(self) -> None:
        meta = VideoMetadata(
            source=Path("/test.mp4"),
            fps=25.0,
            total_frames=5000,
            width=1280,
            height=720,
            duration_sec=200.0,
            codec="avc1",
        )
        assert not meta.is_valid_duration


class TestKeypoint:
    def test_visible(self) -> None:
        kp = Keypoint(x=100.0, y=200.0, visibility=0.95)
        assert kp.is_visible

    def test_not_visible(self) -> None:
        kp = Keypoint(x=100.0, y=200.0, visibility=0.3)
        assert not kp.is_visible


class TestPoseLandmarks:
    def test_get_keypoint_by_name(self) -> None:
        keypoints = [Keypoint(x=float(i), y=float(i * 2), visibility=0.9) for i in range(33)]
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        nose = pose.get("nose")
        assert nose is not None
        assert nose.x == 0.0

    def test_get_nonexistent_keypoint(self) -> None:
        pose = PoseLandmarks(keypoints=[], frame_idx=0)
        result = pose.get("nonexistent")
        assert result is None


class TestPlayerDetection:
    def test_detection_creation(self, sample_bbox: BBox) -> None:
        det = PlayerDetection(track_id=1, bbox=sample_bbox, confidence=0.9, team=0)
        assert det.track_id == 1
        assert det.team == 0
        assert det.confidence > 0.8

    def test_detection_no_team(self, sample_bbox: BBox) -> None:
        det = PlayerDetection(track_id=5, bbox=sample_bbox, confidence=0.5)
        assert det.team is None


class TestBallPosition:
    def test_detected_position(self) -> None:
        from padelvision.types import BallPosition

        bp = BallPosition(frame_idx=10, x=320.5, y=240.0, confidence=0.85)
        assert bp.is_detected
        assert not bp.interpolated

    def test_interpolated_position(self) -> None:
        from padelvision.types import BallPosition

        bp = BallPosition(frame_idx=10, x=None, y=None, confidence=0.0, interpolated=True)
        assert not bp.is_detected
        assert bp.interpolated


class TestPlayerScore:
    def test_breakdown(self) -> None:
        score = PlayerScore(
            global_score=4.2,
            consistency=4.5,
            technique=4.0,
            mobility=3.8,
            power=4.6,
            positioning=4.1,
        )
        bd = score.breakdown
        assert len(bd) == 5
        assert bd["consistency"] == 4.5
        assert bd["power"] == 4.6
