"""PipelineManager — orchestrates all PadelVision modules programmatically with progress reporting."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from padelvision.core.ball_tracker import BallTracker
from padelvision.core.detector import PlayerDetector
from padelvision.core.feature_extractor import FeatureExtractor
from padelvision.core.pose_estimator import PoseEstimator
from padelvision.core.preprocessor import VideoPreprocessor
from padelvision.core.scoring_engine import ScoringEngine
from padelvision.core.shot_classifier import ShotClassifier
from padelvision.types import (
    BallTrajectory,
    FrameFeatures,
    ImpactEvent,
    PlayerDetection,
    PlayerMetrics,
    PlayerScore,
    PlayerTrack,
    PoseLandmarks,
    Shot,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[float, str], None]


@dataclass
class PipelineResult:
    """Complete pipeline output ready for JSON serialization."""

    video_duration_sec: float
    fps_analyzed: float
    players: list[dict[str, Any]] = field(default_factory=list)
    ball_trajectory: dict[str, Any] | None = None
    impacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline run."""

    model_size: str = "s"
    device: str = "auto"
    pose_complexity: str = "full"
    skip_pose: bool = False
    skip_ball: bool = False
    skip_features: bool = False
    skip_classifier: bool = False
    skip_scoring: bool = False
    scoring_config_path: str | Path | None = None
    target_fps: int = 25
    max_dimension: int = 1280
    tracknet_weights: str | Path | None = None


class PipelineManager:
    """Orchestrates the full PadelVision pipeline with progress reporting and error handling."""

    def __init__(self, config: PipelineConfig | None = None, progress_cb: ProgressCallback | None = None) -> None:
        self.config = config or PipelineConfig()
        self.progress_cb = progress_cb or self._default_progress
        self._report(0.0, "Initializing pipeline")

    def run(self, video_path: str | Path) -> PipelineResult:
        """Run the full pipeline and return structured results."""
        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._report(0.0, "Loading video")
        preprocessor = VideoPreprocessor(
            target_fps=self.config.target_fps,
            max_dimension=self.config.max_dimension,
        )
        metadata = preprocessor.load(video_path)
        logger.info(
            f"Video: {metadata.duration_sec:.1f}s | {metadata.fps:.1f} FPS | {metadata.width}x{metadata.height}"
        )

        court_roi = preprocessor.get_court_roi()

        self._report(0.05, "Initializing modules")
        detector = PlayerDetector(model_size=self.config.model_size, device=self.config.device)

        pose_estimator = None if self.config.skip_pose else PoseEstimator(model_complexity=self.config.pose_complexity)

        ball_tracker = None
        if not self.config.skip_ball:
            weights = self.config.tracknet_weights or "models/tracknet/track.pt"
            ball_tracker = BallTracker(weights_path=weights, device=self.config.device)

        feature_extractor = None if self.config.skip_features else FeatureExtractor()

        shot_classifier = None
        if not self.config.skip_classifier:
            shot_classifier = ShotClassifier(fps=metadata.fps)

        scoring_engine = None
        if not self.config.skip_scoring:
            if self.config.scoring_config_path:
                scoring_engine = ScoringEngine.from_file(self.config.scoring_config_path)
            else:
                scoring_engine = ScoringEngine()

        self._report(0.10, "Detecting players and poses")
        all_frames, all_detections, all_poses = self._collect_frames_and_detections(
            preprocessor,
            detector,
            pose_estimator,
            court_roi,
        )

        self._report(0.40, "Tracking ball")
        ball_trajectory = None
        if ball_tracker:
            ball_trajectory = ball_tracker.track(all_frames, fps=metadata.fps)
            detected = sum(1 for p in ball_trajectory.positions if p.is_detected and not p.interpolated)
            logger.info(f"Ball detected in {detected}/{len(ball_trajectory.positions)} frames")

        self._report(0.60, "Extracting features")
        player_features: dict[int, list[FrameFeatures]] = {}
        all_impacts: list[ImpactEvent] = []
        if feature_extractor and ball_trajectory:
            player_tracks = self._build_player_tracks(all_detections, all_poses)
            all_impacts = feature_extractor.detect_impacts(player_tracks, ball_trajectory, metadata.fps)
            player_features = self._extract_features_with_ball_height(
                feature_extractor,
                player_tracks,
                ball_trajectory,
                (metadata.width, metadata.height),
                court_roi,
                metadata.fps,
            )

        self._report(0.75, "Classifying shots")
        player_shots: dict[int, list[Shot]] = {}
        if shot_classifier and ball_trajectory:
            for track_id, features in player_features.items():
                impacts_for_player = [i for i in all_impacts if i.player_id == track_id]
                player_shots[track_id] = shot_classifier.classify_all(features, impacts_for_player, metadata.fps)

        self._report(0.90, "Computing scores")
        player_scores: dict[int, PlayerScore] = {}
        if scoring_engine and ball_trajectory:
            for track_id, features in player_features.items():
                shots = player_shots.get(track_id, [])
                metrics = self._compute_player_metrics(
                    shots=shots,
                    features=features,
                    ball_trajectory=ball_trajectory,
                )
                player_scores[track_id] = scoring_engine.compute_score(metrics)

        self._report(1.0, "Building results")
        result = self._build_result(
            metadata,
            ball_trajectory,
            all_impacts,
            player_shots,
            player_scores,
            all_detections,
        )

        elapsed = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed:.1f}s")

        self._cleanup(preprocessor, pose_estimator, ball_tracker)
        return result

    def _collect_frames_and_detections(
        self,
        preprocessor: VideoPreprocessor,
        detector: PlayerDetector,
        pose_estimator: PoseEstimator | None,
        court_roi,
    ) -> tuple[list[np.ndarray], list[list], list[list[PoseLandmarks]]]:
        """Collect all frames, detections, and poses in a single pass."""
        all_frames: list[np.ndarray] = []
        all_detections: list[list] = []
        all_poses: list[list[PoseLandmarks]] = []

        for batch in preprocessor.frame_batches():
            detections_batch = detector.detect_frames(batch.frames, court_roi=court_roi)
            poses_batch: list[list[PoseLandmarks]] = [[] for _ in batch.frames]

            if pose_estimator:
                poses_batch = pose_estimator.estimate_batch(batch.frames, detections_batch)

            for frame, detections, poses in zip(batch.frames, detections_batch, poses_batch):
                all_frames.append(frame)
                all_detections.append(detections)
                all_poses.append(poses)

        return all_frames, all_detections, all_poses

    def _build_player_tracks(
        self,
        all_detections: list[list],
        all_poses: list[list[PoseLandmarks]],
    ) -> dict[int, PlayerTrack]:
        """Build player tracks from detection and pose data."""
        tracks: dict[int, PlayerTrack] = {}

        for frame_idx, (detections, poses) in enumerate(zip(all_detections, all_poses)):
            for detection in detections:
                tid = detection.track_id
                if tid not in tracks:
                    tracks[tid] = PlayerTrack(track_id=tid, team=detection.team, detections={}, poses={})
                tracks[tid].detections[frame_idx] = detection

            for pose in poses:
                for detection in detections:
                    tid = detection.track_id
                    if tid in tracks:
                        tracks[tid].poses[frame_idx] = pose
                    break

        return tracks

    def _compute_ball_above_head(
        self,
        detection: PlayerDetection | None,
        pose: PoseLandmarks | None,
        ball_x: float | None,
        ball_y: float | None,
    ) -> bool:
        """Check if the ball is above the player's head."""
        if pose is None or ball_y is None or len(pose.keypoints) < 12:
            return False

        left_shoulder = pose.keypoints[11]
        right_shoulder = pose.keypoints[12]

        if not (left_shoulder.is_visible and right_shoulder.is_visible):
            return False

        shoulder_y = min(left_shoulder.y, right_shoulder.y)
        return ball_y < shoulder_y

    def _extract_features_with_ball_height(
        self,
        feature_extractor: FeatureExtractor,
        player_tracks: dict[int, PlayerTrack],
        ball_trajectory: BallTrajectory,
        video_size: tuple[int, int],
        court_roi,
        fps: float,
    ) -> dict[int, list[FrameFeatures]]:
        """Extract features and compute ball_above_head for each player."""
        ball_pos_by_frame = {p.frame_idx: p for p in ball_trajectory.positions}
        player_features: dict[int, list[FrameFeatures]] = {}

        for track_id, track in player_tracks.items():
            features = feature_extractor.extract_features(
                track,
                ball_trajectory,
                video_size,
                court_roi,
                fps,
            )

            enhanced: list[FrameFeatures] = []
            for feat in features:
                det = track.detections.get(feat.frame_idx)
                pose = track.poses.get(feat.frame_idx)
                ball_pos = ball_pos_by_frame.get(feat.frame_idx)
                ball_x = ball_pos.x if ball_pos and ball_pos.is_detected else None
                ball_y = ball_pos.y if ball_pos and ball_pos.is_detected else None
                ball_above = self._compute_ball_above_head(det, pose, ball_x, ball_y)

                enhanced.append(
                    FrameFeatures(
                        frame_idx=feat.frame_idx,
                        elbow_angle=feat.elbow_angle,
                        shoulder_rotation=feat.shoulder_rotation,
                        hip_rotation=feat.hip_rotation,
                        knee_bend=feat.knee_bend,
                        weight_transfer=feat.weight_transfer,
                        court_position=feat.court_position,
                        player_speed=feat.player_speed,
                        distance_to_ball=feat.distance_to_ball,
                        stroke_phase=feat.stroke_phase,
                        ball_above_head=ball_above,
                    )
                )
            player_features[track_id] = enhanced

        return player_features

    def _compute_average_elbow_angle(self, features: list[FrameFeatures]) -> float:
        """Compute average elbow angle across all frames."""
        angles = [f.elbow_angle for f in features if f.elbow_angle is not None]
        return sum(angles) / len(angles) if angles else 0.0

    def _compute_average_speed(self, features: list[FrameFeatures]) -> float:
        """Compute average player speed in m/s (approximate from px/s)."""
        speeds = [f.player_speed for f in features if f.player_speed is not None]
        if not speeds:
            return 0.0
        avg_px_s = sum(speeds) / len(speeds)
        return avg_px_s / 100.0

    def _compute_avg_ball_speed(self, ball_trajectory: BallTrajectory) -> float:
        """Compute average ball speed in km/h."""
        if not ball_trajectory.speed_kmh:
            return 0.0
        speeds = [s for s in ball_trajectory.speed_kmh if s > 0]
        return sum(speeds) / len(speeds) if speeds else 0.0

    def _compute_position_optimality(self, features: list[FrameFeatures]) -> float:
        """Compute how close to optimal court positions (0-1)."""
        positions = [f.court_position for f in features if f.court_position is not None]
        if not positions:
            return 0.0

        optimal_center = (0.5, 0.5)
        distances = []
        for x, y in positions:
            dist = ((x - optimal_center[0]) ** 2 + (y - optimal_center[1]) ** 2) ** 0.5
            max_dist = (0.5**2 + 0.5**2) ** 0.5
            distances.append(1.0 - dist / max_dist)

        return sum(distances) / len(distances)

    def _compute_court_coverage(self, features: list[FrameFeatures], grid_size: int = 4) -> float:
        """Compute percentage of court grid cells visited by player."""
        positions = [f.court_position for f in features if f.court_position is not None]
        if not positions:
            return 0.0

        visited: set[tuple[int, int]] = set()
        for x, y in positions:
            cell_x = min(int(x * grid_size), grid_size - 1)
            cell_y = min(int(y * grid_size), grid_size - 1)
            visited.add((cell_x, cell_y))

        total_cells = grid_size * grid_size
        return len(visited) / total_cells

    def _compute_player_metrics(
        self,
        shots: list[Shot],
        features: list[FrameFeatures],
        ball_trajectory: BallTrajectory,
    ) -> PlayerMetrics:
        """Compute aggregated metrics for a player."""
        return ScoringEngine().compute_metrics_from_data(
            shots=shots,
            avg_elbow_angle=self._compute_average_elbow_angle(features),
            avg_speed_ms=self._compute_average_speed(features),
            avg_ball_speed_kmh=self._compute_avg_ball_speed(ball_trajectory),
            avg_position_optimality=self._compute_position_optimality(features),
            court_coverage_pct=self._compute_court_coverage(features),
        )

    def _build_result(
        self,
        metadata,
        ball_trajectory: BallTrajectory | None,
        impacts: list[ImpactEvent],
        player_shots: dict[int, list[Shot]],
        player_scores: dict[int, PlayerScore],
        all_detections: list[list],
    ) -> PipelineResult:
        """Build the final PipelineResult from all pipeline outputs."""
        ball_data: dict[str, Any] | None = None
        if ball_trajectory:
            detected = sum(1 for p in ball_trajectory.positions if p.is_detected and not p.interpolated)
            speeds = [s for s in ball_trajectory.speed_kmh if s > 0]
            ball_data = {
                "total_detected_frames": detected,
                "detection_rate": detected / max(1, len(ball_trajectory.positions)),
                "avg_speed_kmh": sum(speeds) / len(speeds) if speeds else 0.0,
            }

        impact_data = [
            {
                "frame": imp.frame_idx,
                "player_id": imp.player_id,
                "ball_x": imp.ball_x,
                "ball_y": imp.ball_y,
                "confidence": imp.confidence,
            }
            for imp in impacts
        ]

        players_data: list[dict[str, Any]] = []
        all_track_ids = set(player_shots.keys()) | set(player_scores.keys())

        for track_id in sorted(all_track_ids):
            shots = player_shots.get(track_id, [])
            score = player_scores.get(track_id)

            shot_distribution: dict[str, int] = {}
            for s in shots:
                shot_distribution[s.shot_type] = shot_distribution.get(s.shot_type, 0) + 1

            team = self._get_player_team(track_id, all_detections)

            player_data: dict[str, Any] = {
                "player_id": track_id,
                "team": team,
                "shots": [
                    {
                        "type": s.shot_type,
                        "frame": s.frame_idx,
                        "timestamp_sec": s.timestamp_sec,
                        "confidence": s.confidence,
                        "quality_score": s.quality_score,
                    }
                    for s in shots
                ],
                "shot_distribution": shot_distribution,
                "stats": {
                    "total_shots": len(shots),
                    "shot_distribution": shot_distribution,
                    "avg_ball_speed_kmh": ball_data["avg_speed_kmh"] if ball_data else 0.0,
                    "court_coverage_pct": 0.0,
                },
            }

            if score:
                player_data["score"] = {
                    "global": score.global_score,
                    "breakdown": score.breakdown,
                }

            players_data.append(player_data)

        return PipelineResult(
            video_duration_sec=metadata.duration_sec,
            fps_analyzed=metadata.fps,
            players=players_data,
            ball_trajectory=ball_data,
            impacts=impact_data,
        )

    def _get_player_team(self, track_id: int, all_detections: list[list]) -> int | None:
        """Get the team assignment for a player from detections."""
        for detections in all_detections:
            for det in detections:
                if det.track_id == track_id:
                    return det.team
        return None

    def _report(self, progress: float, message: str) -> None:
        """Report progress via callback."""
        self.progress_cb(progress, message)

    @staticmethod
    def _default_progress(progress: float, message: str) -> None:
        """Default progress handler that logs."""
        logger.info(f"[{progress:.0%}] {message}")

    @staticmethod
    def _cleanup(
        preprocessor: VideoPreprocessor,
        pose_estimator: PoseEstimator | None,
        ball_tracker: BallTracker | None,
    ) -> None:
        """Clean up resources."""
        preprocessor.close()
        if pose_estimator:
            pose_estimator.close()
        if ball_tracker:
            ball_tracker.close()
