"""Tests for API Pydantic models."""

from __future__ import annotations

from padelvision.api.models import (
    AnalysisResult,
    AnalyzeOptions,
    BallTrajectoryResponse,
    JobStatusResponse,
    JobSubmittedResponse,
    PlayerResponse,
    PlayerStats,
    ScoreBreakdown,
    ShotResponse,
    _make_score,
)


def test_analyze_options_defaults():
    opts = AnalyzeOptions()
    assert opts.players == []
    assert opts.include_ball is True
    assert opts.include_poses is False


def test_analyze_options_custom():
    opts = AnalyzeOptions(players=[0, 1], include_ball=False, include_poses=True)
    assert opts.players == [0, 1]
    assert opts.include_ball is False
    assert opts.include_poses is True


def test_score_breakdown():
    breakdown = ScoreBreakdown(
        consistency=4.5,
        technique=4.0,
        mobility=3.8,
        power=4.6,
        positioning=4.1,
    )
    assert breakdown.consistency == 4.5


def test_player_score_response():
    breakdown = ScoreBreakdown(
        consistency=4.5,
        technique=4.0,
        mobility=3.8,
        power=4.6,
        positioning=4.1,
    )
    score = _make_score(4.2, breakdown)
    assert score.global_score == 4.2
    assert score.breakdown.technique == 4.0


def test_shot_response():
    shot = ShotResponse(
        type="drive_forehand",
        frame=312,
        timestamp_sec=12.48,
        confidence=0.87,
        quality_score=5.1,
    )
    assert shot.type == "drive_forehand"
    assert shot.confidence == 0.87


def test_player_stats():
    stats = PlayerStats(
        total_shots=34,
        shot_distribution={"drive_forehand": 12, "bandeja": 5},
        avg_ball_speed_kmh=98.4,
        court_coverage_pct=0.68,
    )
    assert stats.total_shots == 34
    assert stats.shot_distribution["drive_forehand"] == 12


def test_player_response():
    breakdown = ScoreBreakdown(
        consistency=4.5,
        technique=4.0,
        mobility=3.8,
        power=4.6,
        positioning=4.1,
    )
    score = _make_score(4.2, breakdown)
    stats = PlayerStats(
        total_shots=34,
        shot_distribution={"drive_forehand": 12},
        avg_ball_speed_kmh=98.4,
        court_coverage_pct=0.68,
    )
    player = PlayerResponse(
        player_id=0,
        team=0,
        score=score,
        shots=[],
        stats=stats,
    )
    assert player.player_id == 0
    assert player.team == 0


def test_ball_trajectory_response():
    trajectory = BallTrajectoryResponse(
        total_detected_frames=1823,
        detection_rate=0.83,
        avg_speed_kmh=94.2,
    )
    assert trajectory.detection_rate == 0.83


def test_analysis_result():
    breakdown = ScoreBreakdown(
        consistency=4.5,
        technique=4.0,
        mobility=3.8,
        power=4.6,
        positioning=4.1,
    )
    score = _make_score(4.2, breakdown)
    stats = PlayerStats(
        total_shots=34,
        shot_distribution={"drive_forehand": 12},
        avg_ball_speed_kmh=98.4,
        court_coverage_pct=0.68,
    )
    player = PlayerResponse(
        player_id=0,
        team=0,
        score=score,
        shots=[],
        stats=stats,
    )
    trajectory = BallTrajectoryResponse(
        total_detected_frames=1823,
        detection_rate=0.83,
        avg_speed_kmh=94.2,
    )
    result = AnalysisResult(
        video_duration_sec=87.3,
        fps_analyzed=25.0,
        players=[player],
        ball_trajectory=trajectory,
    )
    assert result.video_duration_sec == 87.3
    assert len(result.players) == 1


def test_analysis_result_without_ball_trajectory():
    result = AnalysisResult(
        video_duration_sec=10.0,
        fps_analyzed=25.0,
        players=[],
    )
    assert result.ball_trajectory is None


def test_job_submitted_response():
    response = JobSubmittedResponse(
        job_id="abc123",
        status="queued",
        estimated_seconds=45.0,
    )
    assert response.job_id == "abc123"
    assert response.status == "queued"


def test_job_status_response():
    response = JobStatusResponse(
        job_id="abc123",
        status="completed",
        progress=1.0,
        result_url="/results/abc123",
    )
    assert response.status == "completed"
    assert response.result_url == "/results/abc123"


def test_job_status_response_failed():
    response = JobStatusResponse(
        job_id="abc123",
        status="failed",
        progress=0.5,
        error="File not found",
    )
    assert response.error == "File not found"
