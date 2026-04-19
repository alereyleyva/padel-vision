"""FastAPI application for PadelVision REST API."""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from padelvision.api.models import (
    AnalysisResult,
    AnalyzeOptions,
    JobStatusResponse,
    JobSubmittedResponse,
)
from padelvision.api.pipeline_manager import PipelineConfig, PipelineManager
from padelvision.api.storage import ResultStore

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

store = ResultStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean up expired results on startup."""
    deleted = store.cleanup_expired()
    if deleted:
        logger.info(f"Cleaned up {deleted} expired results")
    yield


app = FastAPI(
    title="PadelVision API",
    description="Computer vision pipeline for padel tennis — shot classification, ball tracking, and scoring.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/analyze", response_model=JobSubmittedResponse | AnalysisResult, status_code=202)
async def analyze_video(
    video: UploadFile,
    options: AnalyzeOptions | None = None,
):
    """Upload a video for analysis. Returns a job_id for async processing."""
    ext = _validate_video_filename(video.filename)

    job_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{job_id}{ext}"

    try:
        content = await video.read()
        temp_path.write_bytes(content)
        logger.info(f"Saved upload {video.filename} -> {temp_path} ({len(content)} bytes)")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    store.create_job(job_id, str(temp_path))

    options_dict = options.model_dump() if options else {}
    config_dict = {
        "skip_pose": not options_dict.get("include_poses", False),
        "skip_ball": not options_dict.get("include_ball", True),
    }

    try:
        from padelvision.api.tasks import analyze_video_task

        analyze_video_task.delay(
            video_path=str(temp_path),
            job_id=job_id,
            config_dict=config_dict,
        )
        estimated = _estimate_duration(temp_path)

        return JobSubmittedResponse(
            job_id=job_id,
            status="queued",
            estimated_seconds=estimated,
        )
    except Exception:
        logger.warning("Celery not available, running synchronously")
        result = _run_sync_analysis(job_id, str(temp_path), config_dict)
        return JSONResponse(status_code=200, content=result)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Check the status of an analysis job."""
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result_url = f"/results/{job_id}" if job["status"] == "completed" else None

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        result_url=result_url,
        error=job.get("error"),
    )


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Retrieve the full analysis result for a completed job."""
    job = store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != "completed":
        return JSONResponse(
            status_code=202,
            content={
                "job_id": job_id,
                "status": job["status"],
                "progress": job["progress"],
                "message": "Analysis still in progress",
            },
        )

    result = store.get_result(job_id)
    if not result:
        raise HTTPException(status_code=500, detail="Result file not found")

    return result


@app.post("/analyze/sync")
async def analyze_video_sync(video: UploadFile, options: AnalyzeOptions | None = None):
    """Upload and analyze video synchronously (for testing / short clips)."""
    ext = _validate_video_filename(video.filename)

    job_id = str(uuid.uuid4())
    temp_path = UPLOAD_DIR / f"{job_id}{ext}"

    try:
        content = await video.read()
        temp_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    store.create_job(job_id, str(temp_path))

    options_dict = options.model_dump() if options else {}
    config_dict = {
        "skip_pose": not options_dict.get("include_poses", False),
        "skip_ball": not options_dict.get("include_ball", True),
    }

    return _run_sync_analysis(job_id, str(temp_path), config_dict)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if not store.delete_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return {"status": "deleted", "job_id": job_id}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


def _run_sync_analysis(job_id: str, video_path: str, config_dict: dict) -> dict:
    """Run analysis synchronously and return results."""
    store.update_job_status(job_id, "processing", 0.05)

    config = PipelineConfig(**config_dict)

    def progress_cb(progress: float, message: str) -> None:
        store.update_job_status(job_id, "processing", progress)

    try:
        manager = PipelineManager(config=config, progress_cb=progress_cb)
        result = manager.run(video_path)
        store.save_result(job_id, result)
    except Exception as exc:
        logger.exception("Synchronous analysis failed for job %s", job_id)
        store.update_job_status(job_id, "failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

    raw_result = store.get_result(job_id)
    return raw_result or {"error": "Failed to retrieve result"}


def _validate_video_filename(filename: str | None) -> str:
    """Validate that the upload has a supported filename and extension."""
    if not filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".avi", ".mov", ".mp4"}
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    return ext


def _estimate_duration(video_path: Path) -> float:
    """Estimate analysis time based on video duration."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps > 0:
            duration = total_frames / fps
            return duration * 1.5
    except Exception:
        pass

    return 60.0
