"""Scoring engine — computes Playtomic-style scores (0-7) from player metrics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from padelvision.types import PlayerMetrics, PlayerScore, Shot

logger = logging.getLogger(__name__)

DEFAULT_PERCENTILES: dict[str, list[float]] = {
    "shot_success_rate": [0.1, 0.9],
    "avg_elbow_angle_quality": [0.3, 0.8],
    "avg_speed_ms": [1.5, 4.0],
    "avg_ball_speed_kmh": [60.0, 120.0],
    "avg_position_optimality": [0.2, 0.7],
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "consistency": 0.25,
    "technique": 0.30,
    "mobility": 0.20,
    "power": 0.15,
    "positioning": 0.10,
}


@dataclass
class ScoringConfig:
    """Configuration for the scoring engine."""

    percentiles: dict[str, list[float]] = field(default_factory=lambda: dict(DEFAULT_PERCENTILES))
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))

    @classmethod
    def from_file(cls, path: str | Path) -> ScoringConfig:
        """Load configuration from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        config = cls()
        if "percentiles" in data:
            config.percentiles.update(data["percentiles"])
        if "weights" in data:
            config.weights.update(data["weights"])
        return config

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        data = {
            "percentiles": self.percentiles,
            "weights": self.weights,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class ScoringEngine:
    """Computes Playtomic-style scores (0-7) from aggregated player metrics.

    Each dimension is normalized to [0, 7] using percentile-based interpolation,
    then combined with configurable weights into a global score.
    """

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self._config = config or ScoringConfig()

    @classmethod
    def from_file(cls, config_path: str | Path) -> ScoringEngine:
        """Create a scoring engine from a configuration file."""
        config = ScoringConfig.from_file(config_path)
        return cls(config)

    def compute_score(self, metrics: PlayerMetrics) -> PlayerScore:
        """Compute the full score breakdown from player metrics.

        Args:
            metrics: Aggregated metrics for a single player.

        Returns:
            PlayerScore with global score and per-dimension breakdown.
        """
        consistency = self._normalize(
            metrics.shot_success_rate,
            self._config.percentiles.get("shot_success_rate", [0.1, 0.9]),
        )
        technique = self._normalize(
            metrics.avg_elbow_angle_quality,
            self._config.percentiles.get("avg_elbow_angle_quality", [0.3, 0.8]),
        )
        mobility = self._normalize(
            metrics.avg_speed_ms,
            self._config.percentiles.get("avg_speed_ms", [1.5, 4.0]),
        )
        power = self._normalize(
            metrics.avg_ball_speed_kmh,
            self._config.percentiles.get("avg_ball_speed_kmh", [60.0, 120.0]),
        )
        positioning = self._normalize(
            metrics.avg_position_optimality,
            self._config.percentiles.get("avg_position_optimality", [0.2, 0.7]),
        )

        weights = self._config.weights
        global_score = (
            weights.get("consistency", 0.25) * consistency
            + weights.get("technique", 0.30) * technique
            + weights.get("mobility", 0.20) * mobility
            + weights.get("power", 0.15) * power
            + weights.get("positioning", 0.10) * positioning
        )

        score = PlayerScore(
            global_score=round(global_score, 1),
            consistency=round(consistency, 1),
            technique=round(technique, 1),
            mobility=round(mobility, 1),
            power=round(power, 1),
            positioning=round(positioning, 1),
        )

        logger.info(
            f"Score computed: global={score.global_score} "
            f"(consistency={score.consistency}, technique={score.technique}, "
            f"mobility={score.mobility}, power={score.power}, "
            f"positioning={score.positioning})"
        )
        return score

    def compute_metrics_from_data(
        self,
        shots: list[Shot],
        avg_elbow_angle: float,
        avg_speed_ms: float,
        avg_ball_speed_kmh: float,
        avg_position_optimality: float,
        court_coverage_pct: float,
    ) -> PlayerMetrics:
        """Build PlayerMetrics from raw analysis data.

        This is the main entry point for converting pipeline output into
        metrics that can be scored.

        Args:
            shots: Classified shots for the player.
            avg_elbow_angle: Average elbow angle across all frames.
            avg_speed_ms: Average player speed in m/s.
            avg_ball_speed_kmh: Average ball speed in km/h.
            avg_position_optimality: How close to optimal positions (0-1).
            court_coverage_pct: Percentage of court covered (0-1).

        Returns:
            PlayerMetrics ready for scoring.
        """
        total_shots = len(shots)
        if total_shots == 0:
            return PlayerMetrics()

        # Shot success rate: ratio of shots with confidence > 0.5
        successful = sum(1 for s in shots if s.confidence > 0.5)
        shot_success_rate = successful / total_shots

        # Elbow angle quality: how close to ideal (160° is ideal for drives)
        ideal_elbow = 160.0
        elbow_deviation = abs(avg_elbow_angle - ideal_elbow)
        elbow_quality = max(0.0, 1.0 - elbow_deviation / 90.0)

        # Shot distribution
        shot_distribution: dict[str, int] = {}
        for shot in shots:
            shot_distribution[shot.shot_type] = shot_distribution.get(shot.shot_type, 0) + 1

        return PlayerMetrics(
            shot_success_rate=shot_success_rate,
            avg_elbow_angle_quality=elbow_quality,
            avg_speed_ms=avg_speed_ms,
            court_coverage_pct=court_coverage_pct,
            avg_ball_speed_kmh=avg_ball_speed_kmh,
            avg_position_optimality=avg_position_optimality,
            total_shots=total_shots,
            shot_distribution=shot_distribution,
        )

    @staticmethod
    def _normalize(value: float, percentiles: list[float]) -> float:
        """Normalize a value to [0, 7] using percentile-based interpolation.

        Args:
            value: The metric value to normalize.
            percentiles: [low, high] percentile reference values.

        Returns:
            Normalized value in [0, 7].
        """
        if not percentiles or len(percentiles) < 2:
            return 0.0

        low, high = percentiles[0], percentiles[1]

        if value <= low:
            return 0.0
        if value >= high:
            return 7.0

        return (value - low) / (high - low) * 7.0

    def compare(self, score_a: PlayerScore, score_b: PlayerScore) -> dict[str, float]:
        """Compare two player scores and return differences by dimension.

        Args:
            score_a: First player's score.
            score_b: Second player's score.

        Returns:
            Dictionary with differences per dimension (positive means A > B).
        """
        return {
            "global": round(score_a.global_score - score_b.global_score, 1),
            "consistency": round(score_a.consistency - score_b.consistency, 1),
            "technique": round(score_a.technique - score_b.technique, 1),
            "mobility": round(score_a.mobility - score_b.mobility, 1),
            "power": round(score_a.power - score_b.power, 1),
            "positioning": round(score_a.positioning - score_b.positioning, 1),
        }
