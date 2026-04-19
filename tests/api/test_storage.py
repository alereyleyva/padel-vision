"""Tests for result storage."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from padelvision.api.pipeline_manager import PipelineResult
from padelvision.api.storage import ResultStore


@pytest.fixture
def tmp_store(tmp_path):
    """Create a ResultStore in a temporary directory."""
    return ResultStore(base_dir=str(tmp_path / "results"), ttl_hours=1)


def test_create_and_get_job(tmp_store, tmp_path):
    job_id = "test-job-1"
    video_path = str(tmp_path / "video.mp4")
    tmp_store.create_job(job_id, video_path)

    job = tmp_store.get_job(job_id)
    assert job is not None
    assert job["job_id"] == job_id
    assert job["status"] == "queued"
    assert job["video_path"] == video_path
    assert job["progress"] == 0.0


def test_update_job_status(tmp_store, tmp_path):
    job_id = "test-job-2"
    tmp_store.create_job(job_id, str(tmp_path / "video.mp4"))

    tmp_store.update_job_status(job_id, "processing", 0.5)
    job = tmp_store.get_job(job_id)
    assert job["status"] == "processing"
    assert job["progress"] == 0.5

    tmp_store.update_job_status(job_id, "completed", 1.0)
    job = tmp_store.get_job(job_id)
    assert job["status"] == "completed"
    assert job["progress"] == 1.0


def test_update_job_error(tmp_store, tmp_path):
    job_id = "test-job-3"
    tmp_store.create_job(job_id, str(tmp_path / "video.mp4"))

    tmp_store.update_job_status(job_id, "failed", error="Video corrupted")
    job = tmp_store.get_job(job_id)
    assert job["status"] == "failed"
    assert job["error"] == "Video corrupted"


def test_save_result(tmp_store, tmp_path):
    job_id = "test-job-4"
    tmp_store.create_job(job_id, str(tmp_path / "video.mp4"))

    result = PipelineResult(
        video_duration_sec=10.0,
        fps_analyzed=25.0,
        players=[],
        ball_trajectory={"total_detected_frames": 100, "detection_rate": 0.5, "avg_speed_kmh": 80.0},
    )

    result_path = tmp_store.save_result(job_id, result)
    assert Path(result_path).exists()

    job = tmp_store.get_job(job_id)
    assert job["status"] == "completed"
    assert job["progress"] == 1.0
    assert job["result_path"] == result_path


def test_get_result(tmp_store, tmp_path):
    job_id = "test-job-5"
    tmp_store.create_job(job_id, str(tmp_path / "video.mp4"))

    result = PipelineResult(
        video_duration_sec=10.0,
        fps_analyzed=25.0,
        players=[{"player_id": 0, "team": 0, "shots": [], "shot_distribution": {}, "stats": {}}],
    )

    tmp_store.save_result(job_id, result)
    retrieved = tmp_store.get_result(job_id)

    assert retrieved is not None
    assert retrieved["video_duration_sec"] == 10.0
    assert len(retrieved["players"]) == 1


def test_get_nonexistent_job(tmp_store):
    job = tmp_store.get_job("nonexistent")
    assert job is None


def test_get_result_nonexistent_job(tmp_store):
    result = tmp_store.get_result("nonexistent")
    assert result is None


def test_cleanup_expired(tmp_store, tmp_path):
    job_id = "test-job-6"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    tmp_store.create_job(job_id, str(video_path))

    result = PipelineResult(
        video_duration_sec=10.0,
        fps_analyzed=25.0,
        players=[],
    )
    tmp_store.save_result(job_id, result)

    import sqlite3

    with sqlite3.connect(tmp_store.db_path) as conn:
        conn.execute(
            "UPDATE jobs SET created_at = ? WHERE job_id = ?",
            (time.time() - 7200, job_id),
        )

    deleted = tmp_store.cleanup_expired()
    assert deleted == 1
    assert not video_path.exists()

    job = tmp_store.get_job(job_id)
    assert job is None


def test_cleanup_no_expired(tmp_store, tmp_path):
    job_id = "test-job-7"
    tmp_store.create_job(job_id, str(tmp_path / "video.mp4"))

    result = PipelineResult(
        video_duration_sec=10.0,
        fps_analyzed=25.0,
        players=[],
    )
    tmp_store.save_result(job_id, result)

    deleted = tmp_store.cleanup_expired()
    assert deleted == 0

    job = tmp_store.get_job(job_id)
    assert job is not None


def test_delete_job_removes_video_and_result_files(tmp_store, tmp_path):
    job_id = "test-job-8"
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    tmp_store.create_job(job_id, str(video_path))

    result = PipelineResult(video_duration_sec=10.0, fps_analyzed=25.0, players=[])
    result_path = Path(tmp_store.save_result(job_id, result))

    assert video_path.exists()
    assert result_path.exists()

    deleted = tmp_store.delete_job(job_id)

    assert deleted is True
    assert not video_path.exists()
    assert not result_path.exists()
    assert tmp_store.get_job(job_id) is None
