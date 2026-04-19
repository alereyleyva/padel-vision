"""Result storage — JSON on disk + SQLite metadata with TTL cleanup."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from padelvision.api.pipeline_manager import PipelineResult


class ResultStore:
    """Stores analysis results as JSON files with SQLite metadata."""

    def __init__(self, base_dir: str | Path = "data/results", ttl_hours: int = 24) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.db_path = self.base_dir / "metadata.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite database with jobs table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0.0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    video_path TEXT,
                    result_path TEXT,
                    error TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")

    def create_job(self, job_id: str, video_path: str) -> None:
        """Register a new job in the store."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, progress, created_at, updated_at, video_path) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, "queued", 0.0, now, now, video_path),
            )

    def update_job_status(self, job_id: str, status: str, progress: float = 0.0, error: str | None = None) -> None:
        """Update job status and progress."""
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, progress = ?, updated_at = ?, error = ? WHERE job_id = ?",
                (status, progress, now, error, job_id),
            )

    def save_result(self, job_id: str, result: PipelineResult) -> str:
        """Save pipeline result as JSON and update job record."""
        result_path = self.base_dir / f"{job_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

        self.update_job_status(job_id, "completed", 1.0)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET result_path = ? WHERE job_id = ?",
                (str(result_path), job_id),
            )
        return str(result_path)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get job metadata by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def get_result(self, job_id: str) -> dict[str, Any] | None:
        """Load result JSON for a completed job."""
        job = self.get_job(job_id)
        if not job or not job.get("result_path"):
            return None

        result_path = Path(job["result_path"])
        if not result_path.exists():
            return None

        with open(result_path, encoding="utf-8") as f:
            return json.load(f)

    def delete_job(self, job_id: str) -> bool:
        """Delete a job row and any files associated with it."""
        job = self.get_job(job_id)
        if job is None:
            return False

        self._unlink_if_present(job.get("result_path"))
        self._unlink_if_present(job.get("video_path"))

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))

        return True

    def cleanup_expired(self) -> int:
        """Remove results older than TTL. Returns count of deleted jobs."""
        cutoff = time.time() - self.ttl_seconds
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            expired = conn.execute(
                "SELECT job_id, result_path, video_path FROM jobs WHERE created_at < ?",
                (cutoff,),
            ).fetchall()

            deleted = 0
            for row in expired:
                self._unlink_if_present(row["result_path"])
                self._unlink_if_present(row["video_path"])
                conn.execute("DELETE FROM jobs WHERE job_id = ?", (row["job_id"],))
                deleted += 1

            return deleted

    @staticmethod
    def _unlink_if_present(path: str | None) -> None:
        if path:
            Path(path).unlink(missing_ok=True)
