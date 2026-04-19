"""Celery task for async video analysis."""

from __future__ import annotations

import logging
import os
import uuid

from celery import Celery

from padelvision.api.pipeline_manager import PipelineConfig, PipelineManager
from padelvision.api.storage import ResultStore

logger = logging.getLogger(__name__)

redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")

celery_app = Celery(
    "padelvision",
    broker=redis_url,
    backend=result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, max_retries=2)
def analyze_video_task(
    self,
    video_path: str,
    job_id: str | None = None,
    config_dict: dict | None = None,
) -> dict:
    """Async Celery task that runs the PadelVision pipeline."""
    job_id = job_id or str(uuid.uuid4())
    store = ResultStore()

    try:
        store.update_job_status(job_id, "processing", 0.05)

        config = PipelineConfig(**config_dict) if config_dict else PipelineConfig()

        def progress_cb(progress: float, message: str) -> None:
            store.update_job_status(job_id, "processing", progress)
            logger.info(f"[{job_id}] [{progress:.0%}] {message}")

        manager = PipelineManager(config=config, progress_cb=progress_cb)
        result = manager.run(video_path)

        result_path = store.save_result(job_id, result)
        logger.info(f"[{job_id}] Analysis complete: {result_path}")

        return {
            "job_id": job_id,
            "status": "completed",
            "result_path": result_path,
        }

    except Exception as exc:
        logger.exception(f"[{job_id}] Analysis failed")
        max_retries = getattr(self, "max_retries", 0) or 0
        if self.request.retries < max_retries:
            store.update_job_status(job_id, "queued", 0.0, error=f"Retrying after error: {exc}")
            raise self.retry(exc=exc, countdown=5)

        store.update_job_status(job_id, "failed", error=str(exc))
        raise
