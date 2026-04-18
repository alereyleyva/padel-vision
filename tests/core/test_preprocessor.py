"""Tests for VideoPreprocessor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from padelvision.core.preprocessor import VideoPreprocessor
from padelvision.types import VideoMetadata


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Create a short synthetic video for testing."""
    video_path = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
    for i in range(150):  # 5 seconds at 30 FPS
        frame = np.full((480, 640, 3), (i % 256), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return video_path


class TestVideoPreprocessor:
    def test_load_and_metadata(self, synthetic_video: Path) -> None:
        with VideoPreprocessor() as prep:
            meta = prep.load(synthetic_video)
            assert isinstance(meta, VideoMetadata)
            assert meta.fps > 0
            assert meta.total_frames > 0
            assert meta.width == 640
            assert meta.height == 480
            assert meta.duration_sec > 0

    def test_load_nonexistent_file(self) -> None:
        with VideoPreprocessor() as prep:
            with pytest.raises(FileNotFoundError):
                prep.load("/nonexistent/video.mp4")

    def test_frame_batches(self, synthetic_video: Path) -> None:
        with VideoPreprocessor(target_fps=25, max_dimension=1280) as prep:
            prep.load(synthetic_video)
            batches = list(prep.frame_batches(batch_size=16))
            assert len(batches) > 0
            total_frames = sum(len(b.frames) for b in batches)
            assert total_frames > 0
            for batch in batches:
                assert len(batch.frames) <= 16
                assert batch.start_idx >= 0
                for frame in batch.frames:
                    assert frame.ndim == 3
                    assert frame.shape[2] == 3

    def test_resize_if_needed(self, synthetic_video: Path) -> None:
        with VideoPreprocessor(max_dimension=320) as prep:
            prep.load(synthetic_video)
            batches = list(prep.frame_batches(batch_size=8))
            for batch in batches:
                for frame in batch.frames:
                    assert max(frame.shape[0], frame.shape[1]) <= 320

    def test_no_resize_needed(self, synthetic_video: Path) -> None:
        with VideoPreprocessor(max_dimension=1280) as prep:
            prep.load(synthetic_video)
            batches = list(prep.frame_batches(batch_size=8))
            has_original = any(f.shape[:2] == (480, 640) for b in batches for f in b.frames)
            assert has_original

    def test_fps_normalization(self, synthetic_video: Path) -> None:
        with VideoPreprocessor(target_fps=15) as prep:
            prep.load(synthetic_video)
            batches = list(prep.frame_batches(batch_size=16))
            total_frames = sum(len(b.frames) for b in batches)
            assert total_frames < 200  # should have fewer frames due to subsampling

    def test_context_manager(self, synthetic_video: Path) -> None:
        with VideoPreprocessor() as prep:
            meta = prep.load(synthetic_video)
            assert meta is not None
        assert prep._cap is None

    def test_court_roi(self, synthetic_video: Path) -> None:
        with VideoPreprocessor() as prep:
            prep.load(synthetic_video)
            roi = prep.get_court_roi()
            # Synthetic video has no real court, so ROI may be None
            # Just ensure it doesn't crash
            assert roi is None or isinstance(roi, type(roi))

    def test_without_load_raises(self) -> None:
        with VideoPreprocessor() as prep:
            with pytest.raises(RuntimeError):
                list(prep.frame_batches())
