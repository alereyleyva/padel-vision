"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from padelvision.api.main import app
from padelvision.api.pipeline_manager import PipelineResult
from padelvision.api.storage import ResultStore


@pytest.fixture
def client(tmp_path):
    """Create a test client with temporary directories."""
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    with patch("padelvision.api.main.UPLOAD_DIR", upload_dir):
        with patch(
            "padelvision.api.main.store",
            ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=24),
        ):
            with TestClient(app) as test_client:
                yield test_client


@pytest.fixture
def mock_video(tmp_path):
    """Create a minimal valid MP4-like file."""
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"\x00" * 1024)
    return video_path


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "0.1.0"


def test_analyze_video_invalid_extension(client, tmp_path):
    video_path = tmp_path / "test.txt"
    video_path.write_bytes(b"not a video")

    with open(video_path, "rb") as f:
        response = client.post("/analyze", files={"video": ("test.txt", f, "text/plain")})

    assert response.status_code == 400
    assert "Unsupported format" in response.json()["detail"]


def test_analyze_video_no_filename(client):
    response = client.post("/analyze", files={"video": ("", b"", "application/octet-stream")})
    assert response.status_code in (400, 422)


def test_get_job_status_not_found(client):
    response = client.get("/jobs/nonexistent-id")
    assert response.status_code == 404


def test_get_results_not_found(client):
    response = client.get("/results/nonexistent-id")
    assert response.status_code == 404


def test_delete_job_not_found(client):
    response = client.delete("/jobs/nonexistent-id")
    assert response.status_code == 404


def test_analyze_sync_runs_pipeline(client, tmp_path):
    """Test the sync endpoint with a mocked pipeline run."""
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"\x00" * 1024)

    mock_result = PipelineResult(
        video_duration_sec=10.0,
        fps_analyzed=25.0,
        players=[
            {
                "player_id": 0,
                "team": 0,
                "shots": [],
                "shot_distribution": {},
                "stats": {
                    "total_shots": 0,
                    "shot_distribution": {},
                    "avg_ball_speed_kmh": 0.0,
                    "court_coverage_pct": 0.0,
                },
            }
        ],
        ball_trajectory={
            "total_detected_frames": 100,
            "detection_rate": 0.5,
            "avg_speed_kmh": 80.0,
        },
    )

    test_store = ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=24)

    with patch("padelvision.api.main.store", test_store):
        with patch("padelvision.api.main.PipelineManager") as mock_manager:
            mock_instance = mock_manager.return_value
            mock_instance.run.return_value = mock_result

            with open(video_path, "rb") as f:
                response = client.post(
                    "/analyze/sync",
                    files={"video": ("test.mp4", f, "video/mp4")},
                )

    assert response.status_code == 200
    data = response.json()
    assert "video_duration_sec" in data, f"Response keys: {list(data.keys())}, body: {data}"
    assert data["video_duration_sec"] == 10.0
    assert len(data["players"]) == 1


def test_analyze_async_falls_back_to_sync_result(client, tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"\x00" * 1024)
    mock_result = PipelineResult(video_duration_sec=12.0, fps_analyzed=25.0, players=[])

    test_store = ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=24)

    with patch("padelvision.api.main.store", test_store):
        with patch("padelvision.api.main.PipelineManager") as mock_manager:
            with patch("padelvision.api.tasks.analyze_video_task.delay", side_effect=RuntimeError("broker down")):
                mock_manager.return_value.run.return_value = mock_result

                with open(video_path, "rb") as f:
                    response = client.post("/analyze", files={"video": ("test.mp4", f, "video/mp4")})

    assert response.status_code == 200
    assert response.json()["video_duration_sec"] == 12.0


def test_get_results_for_completed_job(client, tmp_path):
    """Test retrieving results for a completed job."""
    store = ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=24)
    job_id = "test-completed-job"

    store.create_job(job_id, str(tmp_path / "video.mp4"))

    result = PipelineResult(
        video_duration_sec=30.0,
        fps_analyzed=25.0,
        players=[],
        ball_trajectory={
            "total_detected_frames": 500,
            "detection_rate": 0.75,
            "avg_speed_kmh": 90.0,
        },
    )
    store.save_result(job_id, result)

    with patch("padelvision.api.main.store", store):
        response = client.get(f"/results/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["video_duration_sec"] == 30.0
        assert data["ball_trajectory"]["detection_rate"] == 0.75


def test_get_results_for_incomplete_job(client, tmp_path):
    """Test that incomplete jobs return 202 with status."""
    store = ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=24)
    job_id = "test-incomplete-job"
    store.create_job(job_id, str(tmp_path / "video.mp4"))
    store.update_job_status(job_id, "processing", 0.5)

    with patch("padelvision.api.main.store", store):
        response = client.get(f"/results/{job_id}")
        assert response.status_code == 202
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 0.5


def test_delete_job(client, tmp_path):
    """Test deleting a job and its results."""
    store = ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=24)
    job_id = "test-delete-job"
    store.create_job(job_id, str(tmp_path / "video.mp4"))

    result = PipelineResult(
        video_duration_sec=10.0,
        fps_analyzed=25.0,
        players=[],
    )
    store.save_result(job_id, result)

    with patch("padelvision.api.main.store", store):
        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "deleted"

        job = store.get_job(job_id)
        assert job is None


def test_analyze_sync_failure_marks_job_failed(client, tmp_path):
    video_path = tmp_path / "test.mp4"
    video_path.write_bytes(b"\x00" * 1024)
    test_store = ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=24)

    with patch("padelvision.api.main.store", test_store):
        with patch("padelvision.api.main.PipelineManager") as mock_manager:
            mock_manager.return_value.run.side_effect = RuntimeError("pipeline exploded")

            with open(video_path, "rb") as f:
                response = client.post("/analyze/sync", files={"video": ("test.mp4", f, "video/mp4")})

    assert response.status_code == 500
    with sqlite3.connect(test_store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT status, error FROM jobs").fetchone()

    assert row is not None
    assert row["status"] == "failed"
    assert "pipeline exploded" in row["error"]
