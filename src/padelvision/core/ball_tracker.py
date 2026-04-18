"""Ball tracker — TrackNet v2 inference, heatmap post-processing, interpolation, and bounce detection."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as transforms_f
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

from padelvision.core.tracknet_model import TrackNet, extract_ball_position
from padelvision.types import BallPosition, BallTrajectory

logger = logging.getLogger(__name__)

DEFAULT_INPUT_SIZE = (288, 512)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_MODEL_PATH = Path("models/tracknet/track.pt")


class BallTracker:
    """Tracks a ball across video frames using TrackNet v2.

    Processes 3 consecutive frames at a time to produce ball position heatmaps.
    Gaps in detection are filled with cubic spline interpolation.
    Bounces are detected by vertical velocity sign changes.
    """

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_MODEL_PATH,
        device: str = "auto",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
    ) -> None:
        self._device = self._resolve_device(device)
        self._confidence_threshold = confidence_threshold
        self._input_size = input_size

        self._model = TrackNet(input_channels=9, out_channels=3)
        weights_path = Path(weights_path)

        if weights_path.exists():
            state_dict = torch.load(str(weights_path), map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info(f"Loaded TrackNet weights from {weights_path}")
        else:
            logger.warning(
                f"TrackNet weights not found at {weights_path}. "
                f"Run scripts/download_models.py to download pretrained weights."
            )

        self._model.to(self._device)
        self._model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            return "cpu"
        return device

    def track(
        self,
        frames: list[np.ndarray],
        fps: float = 25.0,
    ) -> BallTrajectory:
        """Track the ball across all frames and return a BallTrajectory.

        Args:
            frames: List of BGR frames (numpy arrays).
            fps: Video FPS, used for speed computation.

        Returns:
            BallTrajectory with positions, bounces, and speeds.
        """
        if len(frames) < 3:
            logger.warning("Need at least 3 frames for ball tracking")
            return BallTrajectory()

        positions = self._predict_all_frames(frames)
        positions = self._interpolate_gaps(positions)
        trajectory = BallTrajectory(positions=positions)

        trajectory.bounces = self._detect_bounces(trajectory, fps)
        trajectory.speed_kmh = self._compute_speeds(trajectory, fps)

        detected_count = sum(1 for p in positions if p.is_detected and not p.interpolated)
        total_count = len(positions)
        rate = detected_count / total_count if total_count > 0 else 0.0
        logger.info(f"Ball tracking: {detected_count}/{total_count} frames detected ({rate:.1%})")

        return trajectory

    def _predict_all_frames(self, frames: list[np.ndarray]) -> list[BallPosition]:
        """Run TrackNet inference across all frames using sliding 3-frame windows."""
        positions: list[BallPosition] = []

        with torch.no_grad():
            for i in range(len(frames)):
                triplet = self._get_triplet(frames, i)
                input_tensor = self._preprocess(triplet).to(self._device)

                output = self._model(input_tensor)
                center_heatmap = output[0, 1].cpu()

                orig_w = frames[0].shape[1]
                orig_h = frames[0].shape[0]

                result = extract_ball_position(
                    center_heatmap,
                    self._confidence_threshold,
                    original_size=(orig_w, orig_h),
                )

                if result is not None:
                    x, y, conf = result
                    positions.append(BallPosition(frame_idx=i, x=x, y=y, confidence=conf, interpolated=False))
                else:
                    positions.append(BallPosition(frame_idx=i, x=None, y=None, confidence=0.0, interpolated=False))

        return positions

    def _get_triplet(self, frames: list[np.ndarray], idx: int) -> list[np.ndarray]:
        """Get 3 consecutive frames centered on idx, clamping at edges."""
        prev_frame = frames[max(0, idx - 1)]
        curr_frame = frames[idx]
        next_frame = frames[min(len(frames) - 1, idx + 1)]
        return [prev_frame, curr_frame, next_frame]

    def _preprocess(self, frame_triplet: list[np.ndarray]) -> torch.Tensor:
        """Convert 3 BGR frames to model input tensor (1, 9, H, W)."""
        tensors = []
        for frame in frame_triplet:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            tensor = transforms_f.resize(tensor, list(self._input_size))
            tensors.append(tensor)

        return torch.cat(tensors, dim=0).unsqueeze(0)

    def _interpolate_gaps(self, positions: list[BallPosition]) -> list[BallPosition]:
        """Fill gaps in ball detection using cubic spline interpolation."""
        detected = [(p.frame_idx, p.x, p.y) for p in positions if p.is_detected]

        if len(detected) < 4:
            return positions

        indices = [d[0] for d in detected]
        xs = [d[1] for d in detected]
        ys = [d[2] for d in detected]

        try:
            cs_x = CubicSpline(indices, xs)
            cs_y = CubicSpline(indices, ys)
        except ValueError:
            return positions

        result = []
        for p in positions:
            if p.is_detected and not p.interpolated:
                result.append(p)
            elif p.is_detected and p.interpolated:
                result.append(p)
            else:
                try:
                    interp_x = float(cs_x(p.frame_idx))
                    interp_y = float(cs_y(p.frame_idx))
                    result.append(
                        BallPosition(
                            frame_idx=p.frame_idx,
                            x=interp_x,
                            y=interp_y,
                            confidence=0.0,
                            interpolated=True,
                        )
                    )
                except ValueError:
                    result.append(p)

        return result

    def _detect_bounces(self, trajectory: BallTrajectory, fps: float) -> list[int]:
        """Detect bounces by finding vertical direction changes.

        A bounce is where the smoothed y-velocity changes from positive (downward)
        to negative (upward), indicating the ball hit the ground or was redirected upward.
        """
        detected = [p for p in trajectory.positions if p.is_detected]

        if len(detected) < 5:
            return []

        ys = [p.y for p in detected if p.y is not None]
        indices = [p.frame_idx for p in detected if p.y is not None]

        if len(ys) < 5:
            return []

        ys_arr = np.array(ys, dtype=np.float64)

        if len(ys_arr) >= 7:
            window = min(7, len(ys_arr) | 1)
            if window % 2 == 0:
                window -= 1
            if window >= 5:
                ys_smooth = savgol_filter(ys_arr, window_length=window, polyorder=2)
            else:
                ys_smooth = ys_arr
        else:
            ys_smooth = ys_arr

        ys_smooth_final: np.ndarray = np.asarray(ys_smooth, dtype=np.float64)
        velocities = np.diff(ys_smooth_final)

        bounces = []
        for i in range(1, len(velocities)):
            if velocities[i - 1] > 0 and velocities[i] < 0:
                threshold = np.mean(np.abs(velocities)) * 0.5
                if abs(velocities[i] - velocities[i - 1]) > threshold:
                    bounce_idx = indices[min(i, len(indices) - 1)]
                    bounces.append(bounce_idx)

        return bounces

    def _compute_speeds(self, trajectory: BallTrajectory, fps: float) -> list[float]:
        """Compute ball speed in km/h between consecutive detections."""
        detected = [p for p in trajectory.positions if p.is_detected]

        if len(detected) < 2 or fps <= 0:
            return []

        speeds: list[float] = []

        for i in range(1, len(detected)):
            if detected[i].x is None or detected[i].y is None:
                speeds.append(0.0)
                continue
            if detected[i - 1].x is None or detected[i - 1].y is None:
                speeds.append(0.0)
                continue

            dx = detected[i].x - detected[i - 1].x  # type: ignore[operator]
            dy = detected[i].y - detected[i - 1].y  # type: ignore[operator]
            dist_px = np.sqrt(dx**2 + dy**2)

            frame_delta = detected[i].frame_idx - detected[i - 1].frame_idx
            if frame_delta <= 0:
                speeds.append(0.0)
                continue

            dt_sec = frame_delta / fps
            speed_px_per_sec = dist_px / dt_sec
            speeds.append(speed_px_per_sec)

        return speeds

    def close(self) -> None:
        self._model = self._model.cpu()

    def __enter__(self) -> BallTracker:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
