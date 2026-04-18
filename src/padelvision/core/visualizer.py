"""Pipeline visualizer — draws trajectories, heatmaps, bounces, impacts, and features onto video frames."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from padelvision.types import BallPosition, BallTrajectory, BoundingBox, ImpactEvent

TRAJECTORY_COLOR_DETECTED = (0, 255, 255)
TRAJECTORY_COLOR_INTERP = (0, 165, 255)
BOUNCE_MARKER_COLOR = (0, 0, 255)
IMPACT_MARKER_COLOR = (255, 0, 255)
HEATMAP_ALPHA = 0.3
TRAIL_LENGTH = 30


class Visualizer:
    """Draws ball trajectory, heatmap, bounces, impacts, and feature overlays on video frames."""

    def draw_trajectory(
        self,
        frame: np.ndarray,
        trajectory: BallTrajectory,
        current_frame: int,
        trail_length: int = TRAIL_LENGTH,
    ) -> np.ndarray:
        """Draw ball trajectory trail on the current frame."""
        annotated = frame.copy()

        start = max(0, current_frame - trail_length)
        positions = trajectory.positions[start : current_frame + 1]

        for pos in positions:
            if not pos.is_detected or pos.x is None or pos.y is None:
                continue
            color = TRAJECTORY_COLOR_DETECTED if not pos.interpolated else TRAJECTORY_COLOR_INTERP
            pt = (int(pos.x), int(pos.y))
            cv2.circle(annotated, pt, 3, color, -1)

        if current_frame < len(trajectory.positions):
            current_pos = trajectory.positions[current_frame]
            if current_pos.is_detected and current_pos.x is not None and current_pos.y is not None:
                cv2.circle(
                    annotated,
                    (int(current_pos.x), int(current_pos.y)),
                    6,
                    (0, 255, 0),
                    -1,
                )

        return annotated

    def draw_heatmap(
        self,
        positions: Sequence[BallPosition],
        frame_size: tuple[int, int],
        alpha: float = HEATMAP_ALPHA,
    ) -> np.ndarray:
        """Generate a heatmap overlay showing ball position frequency."""
        h, w = frame_size[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        detected = [p for p in positions if p.is_detected and p.x is not None and p.y is not None]
        if not detected:
            return np.zeros((h, w, 3), dtype=np.uint8)

        for pos in detected:
            x = int(pos.x) if pos.x is not None else 0
            y = int(pos.y) if pos.y is not None else 0
            if 0 <= x < w and 0 <= y < h:
                heatmap[y, x] += 1.0

        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        mask = heatmap > 0.01
        overlay[mask] = heatmap_colored[mask]

        result = cv2.addWeighted(overlay, alpha, np.zeros_like(overlay), 0, 0)
        return result

    def overlay_heatmap(
        self,
        frame: np.ndarray,
        positions: Sequence[BallPosition],
        alpha: float = HEATMAP_ALPHA,
    ) -> np.ndarray:
        """Overlay the ball position heatmap onto a frame."""
        heatmap = self.draw_heatmap(positions, frame.shape)
        mask = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8) * 255

        blended = cv2.addWeighted(frame, 1.0 - alpha, heatmap, alpha, 0)
        result = frame.copy()
        result[mask > 0] = blended[mask > 0]
        return result

    def draw_bounces(
        self,
        frame: np.ndarray,
        bounces: list[int],
        positions: Sequence[BallPosition],
        current_frame: int,
    ) -> np.ndarray:
        """Draw bounce markers on the frame."""
        annotated = frame.copy()

        for bounce_frame in bounces:
            if bounce_frame < len(positions):
                pos = positions[bounce_frame]
                if pos.is_detected and pos.x is not None and pos.y is not None:
                    pt = (int(pos.x), int(pos.y))
                    cv2.drawMarker(annotated, pt, BOUNCE_MARKER_COLOR, cv2.MARKER_TRIANGLE_UP, 15, 2)
                    if abs(bounce_frame - current_frame) < 5:
                        cv2.circle(annotated, pt, 12, BOUNCE_MARKER_COLOR, 2)

        return annotated

    def draw_impacts(
        self,
        frame: np.ndarray,
        impacts: list[ImpactEvent],
        current_frame: int,
    ) -> np.ndarray:
        """Draw impact event markers on the frame."""
        annotated = frame.copy()

        for impact in impacts:
            if abs(impact.frame_idx - current_frame) < 5:
                pt = (int(impact.ball_x), int(impact.ball_y))
                cv2.drawMarker(annotated, pt, IMPACT_MARKER_COLOR, cv2.MARKER_STAR, 20, 2)
                label = f"Impact P{impact.player_id} ({impact.confidence:.0%})"
                cv2.putText(
                    annotated,
                    label,
                    (pt[0] + 10, pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    IMPACT_MARKER_COLOR,
                    1,
                )

        return annotated

    def draw_court_roi(
        self,
        frame: np.ndarray,
        court_roi: BoundingBox | None,
    ) -> np.ndarray:
        """Draw court ROI rectangle on the frame."""
        if court_roi is None:
            return frame

        annotated = frame.copy()
        pt1 = (int(court_roi.x), int(court_roi.y))
        pt2 = (int(court_roi.x + court_roi.width), int(court_roi.y + court_roi.height))
        cv2.rectangle(annotated, pt1, pt2, (255, 200, 0), 2)
        cv2.putText(
            annotated,
            "Court ROI",
            (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 200, 0),
            1,
        )
        return annotated
