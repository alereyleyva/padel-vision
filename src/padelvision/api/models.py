"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class AnalyzeOptions(BaseModel):
    """Options for video analysis."""

    players: list[int] = Field(default_factory=list, description="Player indices to analyze (empty = all)")
    include_ball: bool = Field(default=True, description="Include ball tracking data")
    include_poses: bool = Field(default=False, description="Include raw pose data (large payload)")


class ScoreBreakdown(BaseModel):
    """Per-dimension score breakdown."""

    consistency: float
    technique: float
    mobility: float
    power: float
    positioning: float


class PlayerScoreResponse(BaseModel):
    """Player score with global and per-dimension values."""

    model_config = ConfigDict(populate_by_name=True)

    global_score: float = Field(alias="global", description="Global score 0-7")
    breakdown: ScoreBreakdown


def _make_score(global_score: float, breakdown: ScoreBreakdown) -> PlayerScoreResponse:
    """Create a PlayerScoreResponse avoiding the reserved keyword issue."""
    return PlayerScoreResponse.model_validate({"global": global_score, "breakdown": breakdown})


class ShotResponse(BaseModel):
    """Classified shot information."""

    type: str
    frame: int
    timestamp_sec: float
    confidence: float
    quality_score: float


class PlayerStats(BaseModel):
    """Aggregated player statistics."""

    total_shots: int
    shot_distribution: dict[str, int]
    avg_ball_speed_kmh: float
    court_coverage_pct: float


class PlayerResponse(BaseModel):
    """Complete player analysis result."""

    player_id: int
    team: int | None
    score: PlayerScoreResponse
    shots: list[ShotResponse]
    stats: PlayerStats


class BallTrajectoryResponse(BaseModel):
    """Ball trajectory summary."""

    total_detected_frames: int
    detection_rate: float
    avg_speed_kmh: float


class AnalysisResult(BaseModel):
    """Complete analysis result returned by /results/{job_id}."""

    video_duration_sec: float
    fps_analyzed: float
    players: list[PlayerResponse]
    ball_trajectory: BallTrajectoryResponse | None = None


class JobSubmittedResponse(BaseModel):
    """Response from POST /analyze."""

    job_id: str
    status: str = "queued"
    estimated_seconds: float


class JobStatusResponse(BaseModel):
    """Response from GET /jobs/{job_id}."""

    job_id: str
    status: str
    progress: float
    result_url: str | None = None
    error: str | None = None
