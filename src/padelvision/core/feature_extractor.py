"""Feature extractor — geometric and semantic features from detections, poses, and ball positions."""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.signal import savgol_filter

from padelvision.types import (
    BallPosition,
    BallTrajectory,
    BBox,
    BoundingBox,
    FrameFeatures,
    ImpactEvent,
    Keypoint,
    PlayerTrack,
    PoseLandmarks,
    StrokePhase,
)

logger = logging.getLogger(__name__)

LANDMARK_WRIST_L = 15
LANDMARK_WRIST_R = 16
LANDMARK_ELBOW_L = 13
LANDMARK_ELBOW_R = 14
LANDMARK_SHOULDER_L = 11
LANDMARK_SHOULDER_R = 12
LANDMARK_HIP_L = 23
LANDMARK_HIP_R = 24
LANDMARK_KNEE_L = 25
LANDMARK_KNEE_R = 26
LANDMARK_ANKLE_L = 27
LANDMARK_ANKLE_R = 28

IMPACT_WINDOW_BEFORE = 10
IMPACT_WINDOW_AFTER = 15
IMPACT_DISTANCE_THRESHOLD_PCT = 0.15
IMPACT_VELOCITY_CHANGE_THRESHOLD = 0.3


class FeatureExtractor:
    """Extracts geometric and semantic features from player tracking data.

    Computes per-frame features like joint angles, speed, court position,
    and distance to ball. Also detects player-ball impact events.
    """

    def extract_features(
        self,
        track: PlayerTrack,
        ball: BallTrajectory,
        frame_size: tuple[int, int],
        court_roi: BoundingBox | None = None,
        fps: float = 25.0,
    ) -> list[FrameFeatures]:
        """Extract frame-by-frame features for a single player track."""
        features: list[FrameFeatures] = []

        frame_indices = sorted(track.detections.keys())
        center_positions = self._compute_center_positions(track, frame_indices)

        for i, frame_idx in enumerate(frame_indices):
            detection = track.detections[frame_idx]
            pose = track.poses.get(frame_idx)
            ball_pos = self._find_ball_position(ball, frame_idx)

            elbow_angle = (
                self.compute_elbow_angle(pose, "right")
                if pose and len(pose.keypoints) >= LANDMARK_WRIST_R + 1
                else self.compute_elbow_angle(pose, "left")
                if pose
                else None
            )

            features.append(
                FrameFeatures(
                    frame_idx=frame_idx,
                    elbow_angle=elbow_angle,
                    shoulder_rotation=self.compute_shoulder_rotation(pose),
                    hip_rotation=self.compute_hip_rotation(pose),
                    knee_bend=self._compute_best_knee_bend(pose),
                    weight_transfer=self._compute_weight_transfer(center_positions, i),
                    court_position=compute_court_position(detection.bbox, frame_size, court_roi),
                    player_speed=compute_player_speed(center_positions, i, fps),
                    distance_to_ball=compute_distance_to_ball(
                        (detection.bbox.center_x, detection.bbox.center_y), ball_pos
                    ),
                    stroke_phase="none",
                )
            )

        impacts = self.detect_impacts({track.track_id: track}, ball, fps)
        impact_frames = {imp.frame_idx: imp for imp in impacts if imp.player_id == track.track_id}

        for feat in features:
            if feat.frame_idx in impact_frames:
                feat = FrameFeatures(
                    frame_idx=feat.frame_idx,
                    elbow_angle=feat.elbow_angle,
                    shoulder_rotation=feat.shoulder_rotation,
                    hip_rotation=feat.hip_rotation,
                    knee_bend=feat.knee_bend,
                    weight_transfer=feat.weight_transfer,
                    court_position=feat.court_position,
                    player_speed=feat.player_speed,
                    distance_to_ball=feat.distance_to_ball,
                    stroke_phase="impact",
                )
            else:
                phase = classify_stroke_phase(feat.frame_idx, impact_frames, frame_indices)
                feat = FrameFeatures(
                    frame_idx=feat.frame_idx,
                    elbow_angle=feat.elbow_angle,
                    shoulder_rotation=feat.shoulder_rotation,
                    hip_rotation=feat.hip_rotation,
                    knee_bend=feat.knee_bend,
                    weight_transfer=feat.weight_transfer,
                    court_position=feat.court_position,
                    player_speed=feat.player_speed,
                    distance_to_ball=feat.distance_to_ball,
                    stroke_phase=phase,
                )
            features[features.index(next(f for f in features if f.frame_idx == feat.frame_idx))] = feat

        return features

    @staticmethod
    def compute_elbow_angle(keypoints: PoseLandmarks | None, side: str) -> float | None:
        """Compute the elbow angle in degrees (wrist-elbow-shoulder)."""
        if keypoints is None or len(keypoints.keypoints) < max(LANDMARK_WRIST_L, LANDMARK_WRIST_R) + 1:
            return None

        if side == "left":
            wrist = keypoints.keypoints[LANDMARK_WRIST_L]
            elbow = keypoints.keypoints[LANDMARK_ELBOW_L]
            shoulder = keypoints.keypoints[LANDMARK_SHOULDER_L]
        else:
            wrist = keypoints.keypoints[LANDMARK_WRIST_R]
            elbow = keypoints.keypoints[LANDMARK_ELBOW_R]
            shoulder = keypoints.keypoints[LANDMARK_SHOULDER_R]

        if not (wrist.is_visible and elbow.is_visible and shoulder.is_visible):
            return None

        return _angle_between_points(wrist, elbow, shoulder)

    @staticmethod
    def compute_shoulder_rotation(keypoints: PoseLandmarks | None) -> float | None:
        """Compute shoulder rotation angle relative to horizontal axis."""
        if keypoints is None or len(keypoints.keypoints) < LANDMARK_SHOULDER_R + 1:
            return None

        ls = keypoints.keypoints[LANDMARK_SHOULDER_L]
        rs = keypoints.keypoints[LANDMARK_SHOULDER_R]

        if not (ls.is_visible and rs.is_visible):
            return None

        dx = rs.x - ls.x
        dy = rs.y - ls.y
        return float(math.degrees(math.atan2(dy, dx)))

    @staticmethod
    def compute_hip_rotation(keypoints: PoseLandmarks | None) -> float | None:
        """Compute hip rotation angle relative to horizontal axis."""
        if keypoints is None or len(keypoints.keypoints) < LANDMARK_HIP_R + 1:
            return None

        lh = keypoints.keypoints[LANDMARK_HIP_L]
        rh = keypoints.keypoints[LANDMARK_HIP_R]

        if not (lh.is_visible and rh.is_visible):
            return None

        dx = rh.x - lh.x
        dy = rh.y - lh.y
        return float(math.degrees(math.atan2(dy, dx)))

    @staticmethod
    def compute_knee_bend(keypoints: PoseLandmarks | None, side: str) -> float | None:
        """Compute knee bend angle in degrees (hip-knee-ankle)."""
        if keypoints is None or len(keypoints.keypoints) < max(LANDMARK_ANKLE_L, LANDMARK_ANKLE_R) + 1:
            return None

        if side == "left":
            hip = keypoints.keypoints[LANDMARK_HIP_L]
            knee = keypoints.keypoints[LANDMARK_KNEE_L]
            ankle = keypoints.keypoints[LANDMARK_ANKLE_L]
        else:
            hip = keypoints.keypoints[LANDMARK_HIP_R]
            knee = keypoints.keypoints[LANDMARK_KNEE_R]
            ankle = keypoints.keypoints[LANDMARK_ANKLE_R]

        if not (hip.is_visible and knee.is_visible and ankle.is_visible):
            return None

        return _angle_between_points(hip, knee, ankle)

    def _compute_best_knee_bend(self, keypoints: PoseLandmarks | None) -> float | None:
        """Return the more visible knee's bend angle."""
        left = self.compute_knee_bend(keypoints, "left")
        right = self.compute_knee_bend(keypoints, "right")
        if left is not None and right is not None:
            return min(left, right)
        return left if left is not None else right

    def _compute_weight_transfer(
        self,
        center_positions: list[tuple[float, float] | None],
        idx: int,
    ) -> float | None:
        """Compute center-of-mass displacement over the last 3 frames."""
        if idx < 3 or len(center_positions) < 4:
            return None

        prev = center_positions[idx - 3]
        curr = center_positions[idx]

        if prev is None or curr is None:
            return None

        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        return float(math.sqrt(dx**2 + dy**2))

    def _compute_center_positions(
        self,
        track: PlayerTrack,
        frame_indices: list[int],
    ) -> list[tuple[float, float] | None]:
        """Compute bbox center positions for each frame."""
        positions: list[tuple[float, float] | None] = []
        for idx in frame_indices:
            if idx in track.detections:
                det = track.detections[idx]
                positions.append((det.bbox.center_x, det.bbox.center_y))
            else:
                positions.append(None)
        return positions

    @staticmethod
    def _find_ball_position(trajectory: BallTrajectory, frame_idx: int) -> BallPosition | None:
        """Find ball position at a given frame index."""
        if frame_idx < len(trajectory.positions) and trajectory.positions[frame_idx].frame_idx == frame_idx:
            return trajectory.positions[frame_idx]
        for pos in trajectory.positions:
            if pos.frame_idx == frame_idx:
                return pos
        return None

    def detect_impacts(
        self,
        tracks: dict[int, PlayerTrack],
        ball: BallTrajectory,
        fps: float,
    ) -> list[ImpactEvent]:
        """Detect player-ball impact events.

        An impact occurs where a player's bbox center is closest to the ball,
        combined with a sharp change in ball velocity.
        """
        impacts: list[ImpactEvent] = []

        ball_positions_by_frame: dict[int, BallPosition] = {p.frame_idx: p for p in ball.positions}

        ball_vy = self._compute_ball_y_velocities(ball, fps)

        min_distance_frames = self._find_min_distance_frames(tracks, ball_positions_by_frame)

        for player_id, (frame_idx, min_dist, bx, by) in min_distance_frames.items():
            if frame_idx not in ball_vy:
                continue

            vy = ball_vy[frame_idx - 1] if frame_idx > 0 and frame_idx - 1 in ball_vy else 0
            vy_after = ball_vy.get(frame_idx + 1, vy)
            velocity_change = abs(vy_after - vy)

            confidence = min(1.0, (1.0 - min_dist) * 0.7 + velocity_change * 0.3)

            if min_dist < IMPACT_DISTANCE_THRESHOLD_PCT or velocity_change > IMPACT_VELOCITY_CHANGE_THRESHOLD:
                impacts.append(
                    ImpactEvent(
                        frame_idx=frame_idx,
                        player_id=player_id,
                        ball_x=bx,
                        ball_y=by,
                        confidence=confidence,
                    )
                )

        return impacts

    def _compute_ball_y_velocities(self, ball: BallTrajectory, fps: float) -> dict[int, float]:
        """Compute smoothed y-velocity of the ball for each frame."""
        detected = [p for p in ball.positions if p.is_detected and p.y is not None]
        if len(detected) < 5:
            return {}

        ys = np.array([p.y for p in detected if p.y is not None])
        indices = [p.frame_idx for p in detected if p.y is not None]

        if len(ys) >= 7:
            window = min(7, len(ys) | 1)
            if window % 2 == 0:
                window -= 1
            ys_smooth = savgol_filter(ys, window_length=max(5, window), polyorder=2)
        else:
            ys_smooth = ys

        velocities: dict[int, float] = {}
        for i in range(1, len(indices)):
            dt = (indices[i] - indices[i - 1]) / fps
            if dt > 0:
                vy = float((ys_smooth[i] - ys_smooth[i - 1]) / dt)
                velocities[indices[i]] = vy

        return velocities

    def _find_min_distance_frames(
        self,
        tracks: dict[int, PlayerTrack],
        ball_positions: dict[int, BallPosition],
    ) -> dict[int, tuple[int, float, float, float]]:
        """For each player, find the frame with minimum distance to the ball."""
        result: dict[int, tuple[int, float, float, float]] = {}

        for player_id, track in tracks.items():
            frame_indices = sorted(track.detections.keys())
            all_frame_sizes = {
                idx: (
                    track.detections[idx].bbox.width,
                    track.detections[idx].bbox.height,
                )
                for idx in frame_indices
            }

            best_dist = float("inf")
            best_frame = 0
            best_ball_x = 0.0
            best_ball_y = 0.0

            for idx in frame_indices:
                if idx not in ball_positions:
                    continue

                bp = ball_positions[idx]
                if not bp.is_detected or bp.x is None or bp.y is None:
                    continue

                det = track.detections[idx]
                dx = det.bbox.center_x - bp.x
                dy = det.bbox.center_y - bp.y
                dist = math.sqrt(dx**2 + dy**2)

                w, h = all_frame_sizes.get(idx, (1.0, 1.0))
                frame_diag = math.sqrt(w**2 + h**2)
                norm_dist = dist / frame_diag if frame_diag > 0 else dist

                if norm_dist < best_dist:
                    best_dist = norm_dist
                    best_frame = idx
                    best_ball_x = bp.x
                    best_ball_y = bp.y

            if best_dist < float("inf"):
                result[player_id] = (best_frame, best_dist, best_ball_x, best_ball_y)

        return result


def compute_court_position(
    bbox: BBox,
    frame_size: tuple[int, int],
    court_roi: BoundingBox | None = None,
) -> tuple[float, float]:
    """Normalize bbox center position to [0, 1] within the court ROI."""
    if court_roi is not None:
        roi = court_roi.as_bbox
        x_norm = (bbox.center_x - roi.x1) / roi.width if roi.width > 0 else 0.5
        y_norm = (bbox.center_y - roi.y1) / roi.height if roi.height > 0 else 0.5
    else:
        w, h = frame_size
        x_norm = bbox.center_x / w if w > 0 else 0.5
        y_norm = bbox.center_y / h if h > 0 else 0.5

    return (max(0.0, min(1.0, x_norm)), max(0.0, min(1.0, y_norm)))


def compute_player_speed(
    center_positions: list[tuple[float, float] | None],
    idx: int,
    fps: float,
) -> float | None:
    """Compute player speed in pixels/second between consecutive frames."""
    if idx < 1 or fps <= 0:
        return None

    curr = center_positions[idx]
    prev = center_positions[idx - 1]

    if curr is None or prev is None:
        return None

    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    return float(math.sqrt(dx**2 + dy**2) * fps)


def compute_distance_to_ball(
    player_center: tuple[float, float],
    ball_pos: BallPosition | None,
) -> float | None:
    """Compute Euclidean distance from player center to ball position."""
    if ball_pos is None or not ball_pos.is_detected or ball_pos.x is None or ball_pos.y is None:
        return None

    dx = player_center[0] - ball_pos.x
    dy = player_center[1] - ball_pos.y
    return float(math.sqrt(dx**2 + dy**2))


def classify_stroke_phase(
    frame_idx: int,
    impact_frames: dict[int, ImpactEvent],
    all_frames: list[int],
) -> StrokePhase:
    """Classify the stroke phase based on proximity to impact frames."""
    if not impact_frames:
        return "none"

    closest_impact_idx = min(impact_frames.keys(), key=lambda fi: abs(fi - frame_idx))
    delta = frame_idx - closest_impact_idx

    if delta == 0:
        return "impact"
    elif -IMPACT_WINDOW_BEFORE <= delta < 0:
        return "loading"
    elif 0 < delta <= IMPACT_WINDOW_AFTER:
        return "followthrough"

    return "none"


def _angle_between_points(p1: Keypoint, p2: Keypoint, p3: Keypoint) -> float:
    """Compute the angle at p2 formed by the line segments p1-p2 and p2-p3."""

    v1x = p1.x - p2.x
    v1y = p1.y - p2.y
    v2x = p3.x - p2.x
    v2y = p3.y - p2.y

    dot = v1x * v2x + v1y * v2y
    mag1 = math.sqrt(v1x**2 + v1y**2)
    mag2 = math.sqrt(v2x**2 + v2y**2)

    if mag1 == 0 or mag2 == 0:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return float(math.degrees(math.acos(cos_angle)))
