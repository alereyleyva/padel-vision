"""Builders for serializable pipeline results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from padelvision.core.pipeline_runtime import PipelineArtifacts


@dataclass
class PipelineResult:
    """Complete pipeline output ready for JSON serialization."""

    video_duration_sec: float
    fps_analyzed: float
    players: list[dict[str, Any]] = field(default_factory=list)
    ball_trajectory: dict[str, Any] | None = None
    impacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert the result dataclass into a JSON-friendly dictionary."""
        return asdict(self)


def build_pipeline_result(artifacts: PipelineArtifacts) -> PipelineResult:
    """Build the public pipeline result from shared runtime artifacts."""
    ball_data = None
    if artifacts.ball_trajectory:
        detected = sum(
            1
            for pos in artifacts.ball_trajectory.positions
            if pos.is_detected and not pos.interpolated
        )
        speeds = [speed for speed in artifacts.ball_trajectory.speed_kmh if speed > 0]
        ball_data = {
            "total_frames": len(artifacts.ball_trajectory.positions),
            "total_detected_frames": detected,
            "detection_rate": detected / max(1, len(artifacts.ball_trajectory.positions)),
            "avg_speed_kmh": sum(speeds) / len(speeds) if speeds else 0.0,
            "bounces": list(artifacts.ball_trajectory.bounces),
        }

    players = []
    all_track_ids = sorted(
        set(artifacts.player_tracks.keys()) | set(artifacts.player_shots.keys()) | set(artifacts.player_scores.keys())
    )
    avg_ball_speed = ball_data["avg_speed_kmh"] if ball_data else 0.0

    for track_id in all_track_ids:
        shots = artifacts.player_shots.get(track_id, [])
        score = artifacts.player_scores.get(track_id)
        track = artifacts.player_tracks.get(track_id)

        shot_distribution: dict[str, int] = {}
        for shot in shots:
            shot_distribution[shot.shot_type] = shot_distribution.get(shot.shot_type, 0) + 1

        player_data: dict[str, Any] = {
            "player_id": track_id,
            "team": track.team if track else None,
            "shots": [
                {
                    "type": shot.shot_type,
                    "frame": shot.frame_idx,
                    "timestamp_sec": shot.timestamp_sec,
                    "confidence": shot.confidence,
                    "quality_score": shot.quality_score,
                }
                for shot in shots
            ],
            "shot_distribution": shot_distribution,
            "stats": {
                "total_shots": len(shots),
                "shot_distribution": shot_distribution,
                "avg_ball_speed_kmh": avg_ball_speed,
                "court_coverage_pct": 0.0,
            },
        }

        if score:
            player_data["score"] = {
                "global": score.global_score,
                "breakdown": score.breakdown,
            }

        players.append(player_data)

    impacts = [
        {
            "frame": impact.frame_idx,
            "player_id": impact.player_id,
            "ball_x": impact.ball_x,
            "ball_y": impact.ball_y,
            "confidence": impact.confidence,
        }
        for impact in artifacts.impacts
    ]

    return PipelineResult(
        video_duration_sec=artifacts.metadata.duration_sec,
        fps_analyzed=artifacts.metadata.fps,
        players=players,
        ball_trajectory=ball_data,
        impacts=impacts,
    )
