"""Tests for TrackNet model and BallTracker."""

from __future__ import annotations

import torch

from padelvision.core.tracknet_model import TrackNet, extract_ball_position


class TestTrackNetModel:
    def test_output_shape(self) -> None:
        model = TrackNet(input_channels=9, out_channels=3)
        x = torch.randn(1, 9, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 288, 512)

    def test_output_range(self) -> None:
        model = TrackNet(input_channels=9, out_channels=3)
        x = torch.randn(1, 9, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_inference(self) -> None:
        model = TrackNet(input_channels=9, out_channels=3)
        x = torch.randn(2, 9, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3, 288, 512)


class TestExtractBallPosition:
    def test_detected_ball(self) -> None:
        heatmap = torch.zeros(100, 200)
        # Create a small blob (5x5) so findContours detects it
        heatmap[48:53, 98:103] = 0.9
        result = extract_ball_position(heatmap, confidence_threshold=0.5)
        assert result is not None
        x, y, conf = result
        assert conf > 0.5
        assert 95 < x < 110
        assert 45 < y < 60

    def test_no_ball_detected(self) -> None:
        heatmap = torch.zeros(100, 200)
        heatmap[50, 50] = 0.3
        result = extract_ball_position(heatmap, confidence_threshold=0.5)
        assert result is None

    def test_with_original_size(self) -> None:
        heatmap = torch.zeros(100, 200)
        # Create a small blob for detection
        heatmap[48:53, 98:103] = 0.95
        result = extract_ball_position(heatmap, confidence_threshold=0.5, original_size=(1920, 1080))
        assert result is not None
        x, y, conf = result
        # Coordinates should be scaled to original size
        assert x > 500
        assert y > 200

    def test_empty_heatmap(self) -> None:
        heatmap = torch.zeros(100, 200)
        result = extract_ball_position(heatmap, confidence_threshold=0.5)
        assert result is None
