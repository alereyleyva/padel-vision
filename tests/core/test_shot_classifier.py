"""Tests for ShotClassifier and ScoringEngine."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from padelvision.core.scoring_engine import ScoringConfig, ScoringEngine
from padelvision.core.shot_classifier import ShotClassifier, ShotClassifierLSTM
from padelvision.types import (
    FrameFeatures,
    ImpactEvent,
    PlayerMetrics,
    PlayerScore,
    Shot,
)


class TestShotClassifierLSTM:
    def test_output_shape(self) -> None:
        model = ShotClassifierLSTM(input_size=66, hidden_size=128, n_classes=8)
        x = torch.randn(2, 25, 66)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 8)

    def test_single_batch(self) -> None:
        model = ShotClassifierLSTM(input_size=66, hidden_size=64, n_classes=8)
        x = torch.randn(1, 10, 66)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8)

    def test_variable_sequence_length(self) -> None:
        model = ShotClassifierLSTM(input_size=66, hidden_size=64, n_classes=8)
        for seq_len in [5, 15, 25]:
            x = torch.randn(1, seq_len, 66)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (1, 8)


def _make_features(
    frame_idx: int,
    elbow_angle: float | None = None,
    shoulder_rotation: float | None = None,
    court_position: tuple[float, float] | None = None,
    ball_above_head: bool = False,
    distance_to_ball: float | None = None,
) -> FrameFeatures:
    """Helper to create FrameFeatures with specific values."""
    return FrameFeatures(
        frame_idx=frame_idx,
        elbow_angle=elbow_angle,
        shoulder_rotation=shoulder_rotation,
        court_position=court_position,
        ball_above_head=ball_above_head,
        distance_to_ball=distance_to_ball,
    )


def _make_impact(frame_idx: int, player_id: int = 1) -> ImpactEvent:
    return ImpactEvent(
        frame_idx=frame_idx,
        player_id=player_id,
        ball_x=300.0,
        ball_y=200.0,
        confidence=0.8,
    )


class TestShotClassifier:
    def test_classify_smash(self) -> None:
        """Overhead + high speed → smash."""
        features = [_make_features(i, ball_above_head=(i == 10), distance_to_ball=abs(i - 10) * 2.0) for i in range(25)]
        impacts = [_make_impact(10)]
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all(features, impacts)
        assert len(shots) == 1
        assert shots[0].shot_type in ("smash", "bandeja")
        assert 0.0 <= shots[0].confidence <= 1.0

    def test_classify_volea(self) -> None:
        """Near net → volea."""
        features = [
            _make_features(
                i,
                court_position=(0.5, 0.2),
                shoulder_rotation=15.0,
                distance_to_ball=abs(i - 10) * 1.5,
            )
            for i in range(25)
        ]
        impacts = [_make_impact(10)]
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all(features, impacts)
        assert len(shots) == 1
        assert "volea" in shots[0].shot_type

    def test_classify_drive(self) -> None:
        """Near baseline → drive."""
        features = [
            _make_features(
                i,
                court_position=(0.5, 0.8),
                shoulder_rotation=20.0,
                distance_to_ball=abs(i - 10) * 1.5,
            )
            for i in range(25)
        ]
        impacts = [_make_impact(10)]
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all(features, impacts)
        assert len(shots) == 1
        assert "drive" in shots[0].shot_type

    def test_classify_lob(self) -> None:
        """Middle zone + low speed → lob."""
        features = [
            _make_features(
                i,
                court_position=(0.5, 0.5),
                shoulder_rotation=5.0,
                distance_to_ball=1.0,
            )
            for i in range(25)
        ]
        impacts = [_make_impact(10)]
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all(features, impacts)
        assert len(shots) == 1
        assert shots[0].shot_type == "lob"

    def test_empty_features(self) -> None:
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all([], [_make_impact(0)])
        assert shots == []

    def test_empty_impacts(self) -> None:
        features = [_make_features(i) for i in range(25)]
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all(features, [])
        assert shots == []

    def test_quality_score_range(self) -> None:
        features = [
            _make_features(
                i,
                elbow_angle=160.0,
                court_position=(0.5, 0.8),
                shoulder_rotation=20.0,
                distance_to_ball=abs(i - 10) * 1.5,
            )
            for i in range(25)
        ]
        impacts = [_make_impact(10)]
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all(features, impacts)
        assert len(shots) == 1
        assert 0.0 <= shots[0].quality_score <= 10.0

    def test_multiple_impacts(self) -> None:
        features = [
            _make_features(
                i,
                court_position=(0.5, 0.8),
                shoulder_rotation=20.0,
                distance_to_ball=abs(i - 10) * 1.5,
            )
            for i in range(50)
        ]
        impacts = [_make_impact(10), _make_impact(30)]
        classifier = ShotClassifier(fps=25.0)
        shots = classifier.classify_all(features, impacts)
        assert len(shots) == 2
        assert shots[0].frame_idx < shots[1].frame_idx

    def test_lstm_not_loaded_returns_unknown(self) -> None:
        classifier = ShotClassifier(fps=25.0)
        x = torch.randn(1, 25, 66)
        shot_type, confidence = classifier.classify_with_lstm(x)
        assert shot_type == "unknown"
        assert confidence == 0.0


class TestScoringConfig:
    def test_default_config(self) -> None:
        config = ScoringConfig()
        assert "consistency" in config.weights
        assert "shot_success_rate" in config.percentiles
        assert abs(config.weights["technique"] - 0.30) < 0.01

    def test_save_and_load(self, tmp_path: Path) -> None:
        config = ScoringConfig()
        config.weights["consistency"] = 0.50
        path = tmp_path / "scoring_config.json"
        config.save(path)

        loaded = ScoringConfig.from_file(path)
        assert loaded.weights["consistency"] == 0.50

    def test_load_partial_config(self, tmp_path: Path) -> None:
        data = {"weights": {"consistency": 0.40}}
        path = tmp_path / "partial.json"
        with open(path, "w") as f:
            json.dump(data, f)

        config = ScoringConfig.from_file(path)
        assert config.weights["consistency"] == 0.40
        assert config.weights["technique"] == 0.30  # Default preserved


class TestScoringEngine:
    def test_normalize_below_low(self) -> None:
        assert ScoringEngine._normalize(0.05, [0.1, 0.9]) == 0.0

    def test_normalize_above_high(self) -> None:
        assert ScoringEngine._normalize(0.95, [0.1, 0.9]) == 7.0

    def test_normalize_at_low(self) -> None:
        assert ScoringEngine._normalize(0.1, [0.1, 0.9]) == 0.0

    def test_normalize_at_high(self) -> None:
        assert ScoringEngine._normalize(0.9, [0.1, 0.9]) == 7.0

    def test_normalize_midpoint(self) -> None:
        result = ScoringEngine._normalize(0.5, [0.1, 0.9])
        assert abs(result - 3.5) < 0.01

    def test_normalize_empty_percentiles(self) -> None:
        assert ScoringEngine._normalize(0.5, []) == 0.0

    def test_compute_score_perfect_metrics(self) -> None:
        metrics = PlayerMetrics(
            shot_success_rate=0.95,
            avg_elbow_angle_quality=0.9,
            avg_speed_ms=5.0,
            avg_ball_speed_kmh=130.0,
            avg_position_optimality=0.8,
        )
        engine = ScoringEngine()
        score = engine.compute_score(metrics)
        assert 5.0 <= score.global_score <= 7.0
        assert score.consistency > 5.0
        assert score.technique > 5.0

    def test_compute_score_poor_metrics(self) -> None:
        metrics = PlayerMetrics(
            shot_success_rate=0.05,
            avg_elbow_angle_quality=0.1,
            avg_speed_ms=0.5,
            avg_ball_speed_kmh=30.0,
            avg_position_optimality=0.1,
        )
        engine = ScoringEngine()
        score = engine.compute_score(metrics)
        assert score.global_score < 2.0

    def test_compute_score_mid_metrics(self) -> None:
        metrics = PlayerMetrics(
            shot_success_rate=0.5,
            avg_elbow_angle_quality=0.55,
            avg_speed_ms=2.75,
            avg_ball_speed_kmh=90.0,
            avg_position_optimality=0.45,
        )
        engine = ScoringEngine()
        score = engine.compute_score(metrics)
        assert 2.0 <= score.global_score <= 5.0

    def test_compute_metrics_from_data(self) -> None:
        shots = [
            Shot(shot_type="drive_forehand", frame_idx=10, timestamp_sec=0.4, confidence=0.8),
            Shot(shot_type="volea_forehand", frame_idx=30, timestamp_sec=1.2, confidence=0.7),
            Shot(shot_type="smash", frame_idx=50, timestamp_sec=2.0, confidence=0.9),
        ]
        engine = ScoringEngine()
        metrics = engine.compute_metrics_from_data(
            shots=shots,
            avg_elbow_angle=155.0,
            avg_speed_ms=2.5,
            avg_ball_speed_kmh=95.0,
            avg_position_optimality=0.6,
            court_coverage_pct=0.7,
        )
        assert metrics.total_shots == 3
        assert metrics.shot_success_rate > 0.5
        assert "drive_forehand" in metrics.shot_distribution
        assert metrics.shot_distribution["drive_forehand"] == 1

    def test_compute_metrics_empty_shots(self) -> None:
        engine = ScoringEngine()
        metrics = engine.compute_metrics_from_data(
            shots=[],
            avg_elbow_angle=0.0,
            avg_speed_ms=0.0,
            avg_ball_speed_kmh=0.0,
            avg_position_optimality=0.0,
            court_coverage_pct=0.0,
        )
        assert metrics.total_shots == 0

    def test_compare_scores(self) -> None:
        score_a = PlayerScore(
            global_score=5.0,
            consistency=5.5,
            technique=4.5,
            mobility=5.0,
            power=5.5,
            positioning=4.5,
        )
        score_b = PlayerScore(
            global_score=4.0,
            consistency=4.0,
            technique=4.0,
            mobility=4.0,
            power=4.0,
            positioning=4.0,
        )
        engine = ScoringEngine()
        diff = engine.compare(score_a, score_b)
        assert diff["global"] == 1.0
        assert diff["consistency"] == 1.5
        assert diff["technique"] == 0.5

    def test_score_from_file_config(self, tmp_path: Path) -> None:
        config_data = {
            "percentiles": {
                "shot_success_rate": [0.0, 1.0],
                "avg_elbow_angle_quality": [0.0, 1.0],
                "avg_speed_ms": [0.0, 10.0],
                "avg_ball_speed_kmh": [0.0, 200.0],
                "avg_position_optimality": [0.0, 1.0],
            },
            "weights": {
                "consistency": 0.20,
                "technique": 0.25,
                "mobility": 0.25,
                "power": 0.20,
                "positioning": 0.10,
            },
        }
        config_path = tmp_path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f)

        engine = ScoringEngine.from_file(config_path)
        metrics = PlayerMetrics(
            shot_success_rate=0.5,
            avg_elbow_angle_quality=0.5,
            avg_speed_ms=5.0,
            avg_ball_speed_kmh=100.0,
            avg_position_optimality=0.5,
        )
        score = engine.compute_score(metrics)
        assert score.global_score > 0.0
        assert score.global_score < 7.0
