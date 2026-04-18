"""Tests for FeatureExtractor — geometric features and impact detection."""

from __future__ import annotations

import numpy as np

from padelvision.core.feature_extractor import (
    FeatureExtractor,
    classify_stroke_phase,
    compute_court_position,
    compute_distance_to_ball,
    compute_player_speed,
)
from padelvision.types import (
    BallPosition,
    BBox,
    BoundingBox,
    ImpactEvent,
    Keypoint,
    PoseLandmarks,
)


def _make_keypoints(angles: dict[str, float] | None = None) -> list[Keypoint]:
    """Create 33 keypoints with optional angles for testing."""
    rng = np.random.default_rng(42)
    keypoints = []
    for i in range(33):
        x = 200.0 + rng.normal(0, 20)
        y = 100.0 + i * 8 + rng.normal(0, 10)
        visibility = 0.95
        keypoints.append(Keypoint(x=x, y=y, visibility=visibility))
    return keypoints


class TestComputeElbowAngle:
    def test_straight_arm(self) -> None:
        keypoints = _make_keypoints()
        # Place wrist, elbow, shoulder in a straight line
        keypoints[11] = Keypoint(x=100.0, y=200.0, visibility=0.95)  # left_shoulder
        keypoints[13] = Keypoint(x=200.0, y=200.0, visibility=0.95)  # left_elbow
        keypoints[15] = Keypoint(x=300.0, y=200.0, visibility=0.95)  # left_wrist
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_elbow_angle(pose, "left")
        assert angle is not None
        assert abs(angle - 180.0) < 5.0

    def test_right_angle(self) -> None:
        keypoints = _make_keypoints()
        # 90 degree angle: shoulder at (200,100), elbow at (200,200), wrist at (300,200)
        keypoints[12] = Keypoint(x=200.0, y=100.0, visibility=0.95)  # right_shoulder
        keypoints[14] = Keypoint(x=200.0, y=200.0, visibility=0.95)  # right_elbow
        keypoints[16] = Keypoint(x=300.0, y=200.0, visibility=0.95)  # right_wrist
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_elbow_angle(pose, "right")
        assert angle is not None
        assert abs(angle - 90.0) < 5.0

    def test_none_keypoints(self) -> None:
        assert FeatureExtractor.compute_elbow_angle(None, "left") is None

    def test_invisible_keypoints(self) -> None:
        keypoints = _make_keypoints()
        keypoints[11] = Keypoint(x=100.0, y=200.0, visibility=0.1)  # invisible shoulder
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_elbow_angle(pose, "left")
        assert angle is None


class TestComputeShoulderRotation:
    def test_horizontal_shoulders(self) -> None:
        keypoints = _make_keypoints()
        keypoints[11] = Keypoint(x=100.0, y=200.0, visibility=0.95)  # left_shoulder
        keypoints[12] = Keypoint(x=300.0, y=200.0, visibility=0.95)  # right_shoulder
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_shoulder_rotation(pose)
        assert angle is not None
        assert abs(angle) < 5.0

    def test_tilted_shoulders(self) -> None:
        keypoints = _make_keypoints()
        keypoints[11] = Keypoint(x=100.0, y=300.0, visibility=0.95)
        keypoints[12] = Keypoint(x=300.0, y=100.0, visibility=0.95)
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_shoulder_rotation(pose)
        assert angle is not None
        assert abs(angle) > 30.0

    def test_none_keypoints(self) -> None:
        assert FeatureExtractor.compute_shoulder_rotation(None) is None


class TestComputeHipRotation:
    def test_horizontal_hips(self) -> None:
        keypoints = _make_keypoints()
        keypoints[23] = Keypoint(x=100.0, y=400.0, visibility=0.95)
        keypoints[24] = Keypoint(x=300.0, y=400.0, visibility=0.95)
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_hip_rotation(pose)
        assert angle is not None
        assert abs(angle) < 5.0

    def test_none_keypoints(self) -> None:
        assert FeatureExtractor.compute_hip_rotation(None) is None


class TestComputeKneeBend:
    def test_straight_leg(self) -> None:
        keypoints = _make_keypoints()
        keypoints[23] = Keypoint(x=200.0, y=300.0, visibility=0.95)  # left_hip
        keypoints[25] = Keypoint(x=200.0, y=400.0, visibility=0.95)  # left_knee
        keypoints[27] = Keypoint(x=200.0, y=500.0, visibility=0.95)  # left_ankle
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_knee_bend(pose, "left")
        assert angle is not None
        assert abs(angle - 180.0) < 5.0

    def test_bent_knee(self) -> None:
        keypoints = _make_keypoints()
        keypoints[24] = Keypoint(x=200.0, y=300.0, visibility=0.95)  # right_hip
        keypoints[26] = Keypoint(x=200.0, y=400.0, visibility=0.95)  # right_knee
        keypoints[28] = Keypoint(x=300.0, y=400.0, visibility=0.95)  # right_ankle (offset)
        pose = PoseLandmarks(keypoints=keypoints, frame_idx=0)
        angle = FeatureExtractor.compute_knee_bend(pose, "right")
        assert angle is not None
        assert angle < 180.0

    def test_none_keypoints(self) -> None:
        assert FeatureExtractor.compute_knee_bend(None, "left") is None


class TestComputeCourtPosition:
    def test_inside_roi(self) -> None:
        bbox = BBox(x1=200, y1=100, x2=400, y2=300)
        roi = BoundingBox(x=100, y=50, width=600, height=400)
        result = compute_court_position(bbox, (800, 600), roi)
        assert result is not None
        x_norm, y_norm = result
        assert 0 < x_norm < 1
        assert 0 < y_norm < 1

    def test_no_roi(self) -> None:
        bbox = BBox(x1=400, y1=300, x2=600, y2=500)
        result = compute_court_position(bbox, (800, 600), None)
        assert result is not None
        x_norm, y_norm = result
        assert abs(x_norm - 0.625) < 0.01
        assert abs(y_norm - 0.667) < 0.01


class TestComputePlayerSpeed:
    def test_basic_speed(self) -> None:
        positions: list[tuple[float, float] | None] = [(0.0, 0.0), (10.0, 0.0)]
        speed = compute_player_speed(positions, 1, 25.0)
        assert speed is not None
        assert abs(speed - 250.0) < 1.0  # 10px * 25fps = 250 px/s

    def test_zero_speed(self) -> None:
        positions: list[tuple[float, float] | None] = [(100.0, 100.0), (100.0, 100.0)]
        speed = compute_player_speed(positions, 1, 25.0)
        assert speed is not None
        assert speed == 0.0

    def test_none_positions(self) -> None:
        positions = [(100.0, 100.0), None]
        speed = compute_player_speed(positions, 1, 25.0)
        assert speed is None


class TestComputeDistanceToBall:
    def test_detected_ball(self) -> None:
        ball = BallPosition(frame_idx=0, x=300.0, y=200.0, confidence=0.9)
        dist = compute_distance_to_ball((100.0, 200.0), ball)
        assert dist is not None
        assert abs(dist - 200.0) < 1.0

    def test_no_ball(self) -> None:
        dist = compute_distance_to_ball((100.0, 200.0), None)
        assert dist is None

    def test_undetected_ball(self) -> None:
        ball = BallPosition(frame_idx=0, x=None, y=None, confidence=0.0)
        dist = compute_distance_to_ball((100.0, 200.0), ball)
        assert dist is None


class TestClassifyStrokePhase:
    def test_impact_frame(self) -> None:
        impacts = {50: ImpactEvent(frame_idx=50, player_id=1, ball_x=100.0, ball_y=200.0, confidence=0.9)}
        result = classify_stroke_phase(50, impacts, [50])
        assert result == "impact"

    def test_loading_phase(self) -> None:
        impacts = {50: ImpactEvent(frame_idx=50, player_id=1, ball_x=100.0, ball_y=200.0, confidence=0.9)}
        result = classify_stroke_phase(45, impacts, [40, 45, 50])
        assert result == "loading"

    def test_followthrough_phase(self) -> None:
        impacts = {50: ImpactEvent(frame_idx=50, player_id=1, ball_x=100.0, ball_y=200.0, confidence=0.9)}
        result = classify_stroke_phase(55, impacts, [50, 55, 60])
        assert result == "followthrough"

    def test_no_impact(self) -> None:
        result = classify_stroke_phase(10, {}, [10])
        assert result == "none"

    def test_far_from_impact(self) -> None:
        impacts = {50: ImpactEvent(frame_idx=50, player_id=1, ball_x=100.0, ball_y=200.0, confidence=0.9)}
        result = classify_stroke_phase(200, impacts, [200])
        assert result == "none"
