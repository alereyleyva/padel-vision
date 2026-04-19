"""PipelineManager — thin API wrapper around the shared pipeline runtime."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from padelvision.core.pipeline_runtime import (
    PipelineArtifacts,
    PipelineConfig,
    ProgressCallback,
    build_player_tracks,
    run_pipeline_runtime,
)
from padelvision.core.result_builder import PipelineResult, build_pipeline_result
from padelvision.types import PlayerTrack

logger = logging.getLogger(__name__)


class PipelineManager:
    """Run the shared pipeline runtime and build API-ready results."""

    def __init__(self, config: PipelineConfig | None = None, progress_cb: ProgressCallback | None = None) -> None:
        self.config = config or PipelineConfig()
        self.progress_cb = progress_cb or self._default_progress
        self._report(0.0, "Initializing pipeline")

    def run(self, video_path: str | Path) -> PipelineResult:
        """Run the full pipeline and return structured results."""
        start_time = time.time()
        artifacts = run_pipeline_runtime(video_path, self.config, self.progress_cb)
        self._report(1.0, "Building results")
        result = build_pipeline_result(artifacts)
        logger.info("Pipeline completed in %.1fs", time.time() - start_time)
        return result

    def _build_player_tracks(
        self,
        all_detections: list[list],
        all_poses: list[list],
    ) -> dict[int, PlayerTrack]:
        """Compatibility wrapper used by tests around the shared helper."""
        return build_player_tracks(all_detections, all_poses)

    def _report(self, progress: float, message: str) -> None:
        self.progress_cb(progress, message)

    @staticmethod
    def _default_progress(progress: float, message: str) -> None:
        logger.info("[%s%%] %s", int(progress * 100), message)


__all__ = [
    "PipelineArtifacts",
    "PipelineConfig",
    "PipelineManager",
    "PipelineResult",
    "ProgressCallback",
]
