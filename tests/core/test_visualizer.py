"""Tests for Visualizer module."""

from __future__ import annotations

import numpy as np

from padelvision.core.visualizer import Visualizer
from padelvision.types import BallPosition, BallTrajectory, BoundingBox, ImpactEvent


class TestDrawTrajectory:
    def test_basic_trajectory(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        positions = [
            BallPosition(frame_idx=0, x=100.0, y=200.0, confidence=0.9),
            BallPosition(frame_idx=1, x=120.0, y=210.0, confidence=0.85),
            BallPosition(frame_idx=2, x=140.0, y=220.0, confidence=0.88),
        ]
        trajectory = BallTrajectory(positions=positions)
        result = viz.draw_trajectory(frame, trajectory, current_frame=2)
        assert result.shape == frame.shape
        # Verify some pixels were modified (not all zeros)
        assert result.sum() > frame.sum()

    def test_empty_trajectory(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        trajectory = BallTrajectory(positions=[])
        result = viz.draw_trajectory(frame, trajectory, current_frame=0)
        assert result is not None


class TestDrawBounces:
    def test_single_bounce(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        positions = [
            BallPosition(frame_idx=0, x=100.0, y=200.0, confidence=0.9),
            BallPosition(frame_idx=1, x=120.0, y=350.0, confidence=0.85),
        ]
        result = viz.draw_bounces(frame, [1], positions, current_frame=1)
        assert result.shape == frame.shape

    def test_no_bounces(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.draw_bounces(frame, [], [], current_frame=0)
        assert result is not None


class TestDrawImpacts:
    def test_single_impact(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        impacts = [ImpactEvent(frame_idx=0, player_id=1, ball_x=300.0, ball_y=200.0, confidence=0.85)]
        result = viz.draw_impacts(frame, impacts, current_frame=0)
        assert result.shape == frame.shape

    def test_no_impacts(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.draw_impacts(frame, [], current_frame=0)
        assert result is not None


class TestDrawCourtROI:
    def test_with_roi(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        roi = BoundingBox(x=50, y=30, width=540, height=420)
        result = viz.draw_court_roi(frame, roi)
        assert result.shape == frame.shape
        assert result.sum() > frame.sum()

    def test_without_roi(self) -> None:
        viz = Visualizer()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = viz.draw_court_roi(frame, None)
        assert result is not None
        np.testing.assert_array_equal(result, frame)
