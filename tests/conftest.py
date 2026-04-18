"""Shared test fixtures for PadelVision tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from padelvision.types import (
    BBox,
    BoundingBox,
    Keypoint,
    PlayerDetection,
    PoseLandmarks,
    VideoMetadata,
)

SAMPLES_DIR = Path(__file__).parent.parent / "data" / "raw"


@pytest.fixture
def sample_video_path() -> Path | None:
    """Return path to a sample padel video, or skip if not available."""
    candidates = [
        SAMPLES_DIR / "padel_clip.mp4",
        SAMPLES_DIR / "padel_pro_clip.mp4",
        SAMPLES_DIR / "padel_amateur_clip.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a synthetic 1280x720 BGR frame for testing."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def sample_bbox() -> BBox:
    return BBox(x1=100.0, y1=200.0, x2=400.0, y2=600.0)


@pytest.fixture
def sample_detection(sample_bbox: BBox) -> PlayerDetection:
    return PlayerDetection(track_id=1, bbox=sample_bbox, confidence=0.85, team=0)


@pytest.fixture
def sample_detections(sample_bbox: BBox) -> list[PlayerDetection]:
    return [
        PlayerDetection(track_id=1, bbox=BBox(x1=100, y1=200, x2=300, y2=500), confidence=0.9, team=0),
        PlayerDetection(track_id=2, bbox=BBox(x1=500, y1=200, x2=700, y2=500), confidence=0.88, team=1),
        PlayerDetection(track_id=3, bbox=BBox(x1=150, y1=220, x2=320, y2=520), confidence=0.75, team=0),
        PlayerDetection(track_id=4, bbox=BBox(x1=550, y1=210, x2=720, y2=510), confidence=0.82, team=1),
    ]


@pytest.fixture
def sample_keypoints() -> list[Keypoint]:
    """Generate 33 synthetic keypoints mimicking MediaPipe Pose output."""
    rng = np.random.default_rng(42)
    keypoints = []
    names = PoseLandmarks.MEDIAPIPE_KEYPOINT_NAMES
    for i, _name in enumerate(names):
        x = 100.0 + rng.normal(0, 20)
        y = 200.0 + i * 10 + rng.normal(0, 15)
        visibility = max(0.0, min(1.0, rng.normal(0.8, 0.15)))
        keypoints.append(Keypoint(x=x, y=y, visibility=visibility))
    return keypoints


@pytest.fixture
def sample_pose(sample_keypoints: list[Keypoint]) -> PoseLandmarks:
    return PoseLandmarks(keypoints=sample_keypoints, frame_idx=0)


@pytest.fixture
def sample_video_metadata() -> VideoMetadata:
    return VideoMetadata(
        source=Path("/fake/video.mp4"),
        fps=25.0,
        total_frames=2500,
        width=1920,
        height=1080,
        duration_sec=100.0,
        codec="avc1",
    )


@pytest.fixture
def sample_court_roi() -> BoundingBox:
    return BoundingBox(x=100.0, y=50.0, width=1100.0, height=620.0)


@pytest.fixture
def sample_frames() -> list[np.ndarray]:
    """Generate a batch of 16 synthetic frames."""
    return [np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8) for _ in range(16)]
