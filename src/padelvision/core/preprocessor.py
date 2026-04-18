"""Video preprocessor — ingests video files, normalizes FPS, and yields frame batches."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from padelvision.types import BoundingBox, FrameBatch, VideoMetadata

logger = logging.getLogger(__name__)

MIN_DURATION_SEC = 5.0
MAX_DURATION_SEC = 120.0
DEFAULT_TARGET_FPS = 25
DEFAULT_BATCH_SIZE = 16


class VideoPreprocessor:
    """Loads and preprocesses video files for the PadelVision pipeline.

    Handles FPS normalization, validation, and batch iteration over frames.
    Uses OpenCV as the primary video I/O backend.
    """

    def __init__(self, target_fps: int = DEFAULT_TARGET_FPS, max_dimension: int = 1280) -> None:
        self._target_fps = target_fps
        self._max_dimension = max_dimension
        self._cap: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None
        self._source: Path | None = None

    def load(self, source: str | Path) -> VideoMetadata:
        """Open a video file and extract metadata. Validates duration constraints."""
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {source}")

        self._source = source
        self._cap = cv2.VideoCapture(str(source))

        if not self._cap.isOpened():
            raise ValueError(f"Cannot open video: {source}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        duration_sec = total_frames / fps if fps > 0 else 0.0

        self._metadata = VideoMetadata(
            source=source,
            fps=fps,
            total_frames=total_frames,
            width=width,
            height=height,
            duration_sec=duration_sec,
            codec=codec,
        )

        if not MIN_DURATION_SEC <= duration_sec <= MAX_DURATION_SEC:
            logger.warning(
                f"Video duration {duration_sec:.1f}s outside recommended range ({MIN_DURATION_SEC}-{MAX_DURATION_SEC}s)"
            )

        logger.info(
            f"Loaded video: {source.name} | {width}x{height} | "
            f"{fps:.1f}fps | {total_frames} frames | {duration_sec:.1f}s"
        )

        return self._metadata

    def frame_batches(self, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[FrameBatch]:
        """Yield batches of frames as numpy arrays (BGR format).

        If the source FPS exceeds target_fps, frames are subsampled.
        If source FPS is lower than target_fps, no upsampling is performed.
        """
        if self._cap is None or self._metadata is None:
            raise RuntimeError("Call load() before iterating frames")

        source_fps = self._metadata.fps
        target_fps = min(self._target_fps, source_fps)

        subsample_ratio = max(1, round(source_fps / target_fps))

        batch: list[np.ndarray] = []
        batch_start_idx = 0
        frame_idx = 0

        try:
            while True:
                if subsample_ratio > 1:
                    for skip_idx in range(subsample_ratio - 1):
                        self._cap.grab()
                        frame_idx += 1

                ret, frame = self._cap.read()
                if not ret:
                    break

                if self._max_dimension < max(frame.shape[1], frame.shape[0]):
                    frame = self._resize_frame(frame)

                batch.append(frame)

                if len(batch) >= batch_size:
                    timestamp = batch_start_idx / source_fps
                    yield FrameBatch(
                        frames=batch,
                        start_idx=batch_start_idx,
                        timestamp_sec=timestamp,
                    )
                    batch = []
                    batch_start_idx = frame_idx + 1

                frame_idx += 1

        finally:
            if batch:
                timestamp = batch_start_idx / source_fps
                yield FrameBatch(
                    frames=batch,
                    start_idx=batch_start_idx,
                    timestamp_sec=timestamp,
                )

    def get_court_roi(self) -> BoundingBox | None:
        """Detect the court region of interest using edge detection.

        Returns None if court detection confidence is below threshold.
        This is a baseline implementation — improved court detection is planned for Phase 2.
        """
        if self._cap is None or self._metadata is None:
            raise RuntimeError("Call load() before detecting court ROI")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self._cap.read()
        if not ret:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        frame_area = frame.shape[0] * frame.shape[1]

        if area < 0.1 * frame_area:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        roi = BoundingBox(x=float(x), y=float(y), width=float(w), height=float(h))

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return roi

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self._max_dimension / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoPreprocessor:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
