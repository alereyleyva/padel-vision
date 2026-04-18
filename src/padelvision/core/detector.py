"""Player detector — YOLOv11 + ByteTrack for stable player detection and tracking."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from padelvision.types import BBox, BoundingBox, PlayerDetection

logger = logging.getLogger(__name__)

PERSON_CLASS_ID = 0
DEFAULT_MODEL_SIZE = "s"
DEFAULT_CONF_THRESHOLD = 0.45
DEFAULT_IOU_THRESHOLD = 0.5


class PlayerDetector:
    """Detects and tracks players in video frames using YOLOv11 + ByteTrack.

    Provides stable track IDs across frames and assigns teams based on
    court position (left vs right).
    """

    def __init__(
        self,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "auto",
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    ) -> None:
        model_name = f"yolo11{model_size}.pt"
        self._device = self._resolve_device(device)
        self._conf = conf_threshold
        self._iou = iou_threshold

        from ultralytics.models.yolo.model import YOLO

        self._model = YOLO(model_name)
        logger.info(f"Loaded YOLOv11 model: {model_name} on {self._device}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def detect_frames(
        self,
        frames: Sequence[np.ndarray],
        court_roi: BoundingBox | None = None,
    ) -> list[list[PlayerDetection]]:
        """Detect players in a batch of frames.

        Returns a list (one per frame) of PlayerDetection objects.
        """
        results = self._model.predict(
            source=list(frames),
            classes=[PERSON_CLASS_ID],
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        all_detections: list[list[PlayerDetection]] = []

        for result in results:
            frame_detections: list[PlayerDetection] = []
            boxes = result.boxes

            if boxes is None:
                all_detections.append(frame_detections)
                continue

            for box in boxes:
                track_ids = box.id
                track_id = int(track_ids[0]) if track_ids is not None else -1

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))

                detection = PlayerDetection(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                )

                if court_roi is not None and not self._is_in_court(detection, court_roi):
                    continue

                frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections

    def track_video(
        self,
        video_path: str | Path,
        court_roi: BoundingBox | None = None,
    ) -> dict[int, list[PlayerDetection]]:
        """Track players across an entire video using ByteTrack.

        Returns a dict mapping track_id to a list of PlayerDetection objects
        ordered by frame index.
        """
        from ultralytics.models.yolo.model import YOLO

        model_name = self._model.ckpt_path or "yolo11s.pt"
        tracker_model = YOLO(model_name)

        results = tracker_model.track(
            source=str(video_path),
            classes=[PERSON_CLASS_ID],
            conf=self._conf,
            iou=self._iou,
            tracker="bytetrack.yaml",
            device=self._device,
            persist=True,
            verbose=False,
        )

        tracks: dict[int, list[PlayerDetection]] = {}

        for frame_idx, result in enumerate(results):
            boxes = result.boxes

            if boxes is None:
                continue

            for box in boxes:
                track_ids = box.id
                if track_ids is None:
                    continue

                track_id = int(track_ids[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())

                bbox = BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2))
                detection = PlayerDetection(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=confidence,
                )

                if court_roi is not None and not self._is_in_court(detection, court_roi):
                    continue

                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append(detection)

        tracks = self._assign_teams(tracks, court_roi)

        logger.info(f"Tracked {len(tracks)} unique player IDs across video")
        return tracks

    @staticmethod
    def _is_in_court(detection: PlayerDetection, court_roi: BoundingBox) -> bool:
        cx = detection.bbox.center_x
        cy = detection.bbox.center_y
        roi = court_roi.as_bbox
        horizontal_margin = roi.width * 0.1
        vertical_margin = roi.height * 0.1
        return (
            roi.x1 - horizontal_margin <= cx <= roi.x2 + horizontal_margin
            and roi.y1 - vertical_margin <= cy <= roi.y2 + vertical_margin
        )

    @staticmethod
    def _assign_teams(
        tracks: dict[int, list[PlayerDetection]],
        court_roi: BoundingBox | None = None,
    ) -> dict[int, list[PlayerDetection]]:
        """Assign teams based on horizontal position (left=0, right=1)."""
        if not tracks:
            return tracks

        avg_x: dict[int, float] = {}
        for track_id, detections in tracks.items():
            if detections:
                avg_x[track_id] = sum(d.bbox.center_x for d in detections) / len(detections)

        if len(avg_x) < 2:
            for track_id in tracks:
                for det in tracks[track_id]:
                    object.__setattr__(det, "team", 0)
            return tracks

        sorted_ids = sorted(avg_x.keys(), key=lambda tid: avg_x[tid])
        n_left = len(sorted_ids) // 2

        for i, track_id in enumerate(sorted_ids):
            team = 0 if i < n_left else 1
            for det in tracks[track_id]:
                object.__setattr__(det, "team", team)

        return tracks
