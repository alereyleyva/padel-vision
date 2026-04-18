"""CLI entry point for PadelVision pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from padelvision.core.ball_tracker import BallTracker
from padelvision.core.detector import PlayerDetector
from padelvision.core.feature_extractor import FeatureExtractor
from padelvision.core.pose_estimator import PoseEstimator
from padelvision.core.preprocessor import VideoPreprocessor
from padelvision.core.scoring_engine import ScoringEngine
from padelvision.core.shot_classifier import ShotClassifier
from padelvision.core.visualizer import Visualizer
from padelvision.types import (
    FrameFeatures,
    ImpactEvent,
    PlayerScore,
    PlayerTrack,
    PoseLandmarks,
    Shot,
)

logger = logging.getLogger(__name__)

TEAM_COLORS = {
    0: (0, 255, 0),  # Green for team 0
    1: (0, 0, 255),  # Red for team 1
    None: (255, 255, 0),  # Cyan for unassigned
}

SKELETON_CONNECTIONS = [
    (11, 12),  # left_shoulder -> right_shoulder
    (11, 13),  # left_shoulder -> left_elbow
    (13, 15),  # left_elbow -> left_wrist
    (12, 14),  # right_shoulder -> right_elbow
    (14, 16),  # right_elbow -> right_wrist
    (11, 23),  # left_shoulder -> left_hip
    (12, 24),  # right_shoulder -> right_hip
    (23, 24),  # left_hip -> right_hip
    (23, 25),  # left_hip -> left_knee
    (25, 27),  # left_knee -> left_ankle
    (24, 26),  # right_hip -> right_knee
    (26, 28),  # right_knee -> right_ankle
]

LANDMARK_SHOULDER_L = 11
LANDMARK_SHOULDER_R = 12


def draw_detections(
    frame: np.ndarray,
    detections: list,
    poses: list[PoseLandmarks] | None = None,
) -> np.ndarray:
    """Draw player bounding boxes, track IDs, and pose skeletons on a frame."""
    annotated = frame.copy()

    for detection in detections:
        color = TEAM_COLORS.get(detection.team, TEAM_COLORS[None])
        bbox = detection.bbox

        cv2.rectangle(
            annotated,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            color,
            2,
        )

        label = f"P{detection.track_id}"
        if detection.team is not None:
            label += f" T{detection.team}"
        cv2.putText(
            annotated,
            label,
            (int(bbox.x1), int(bbox.y1) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if poses:
        for pose in poses:
            _draw_skeleton(annotated, pose)

    return annotated


def _draw_skeleton(frame: np.ndarray, pose: PoseLandmarks) -> None:
    """Draw pose skeleton connections on the frame in-place."""
    if len(pose.keypoints) < 29:
        return

    for start_idx, end_idx in SKELETON_CONNECTIONS:
        if start_idx >= len(pose.keypoints) or end_idx >= len(pose.keypoints):
            continue
        kp1 = pose.keypoints[start_idx]
        kp2 = pose.keypoints[end_idx]

        if kp1.is_visible and kp2.is_visible:
            cv2.line(
                frame,
                (int(kp1.x), int(kp1.y)),
                (int(kp2.x), int(kp2.y)),
                (255, 200, 100),
                2,
            )

    for kp in pose.keypoints:
        if kp.is_visible:
            cv2.circle(frame, (int(kp.x), int(kp.y)), 3, (0, 255, 255), -1)


def _compute_ball_above_head(
    detection,
    pose: PoseLandmarks | None,
    ball_x: float | None,
    ball_y: float | None,
) -> bool:
    """Check if the ball is above the player's head at this frame."""
    if pose is None or ball_y is None or len(pose.keypoints) < LANDMARK_SHOULDER_R + 1:
        return False

    left_shoulder = pose.keypoints[LANDMARK_SHOULDER_L]
    right_shoulder = pose.keypoints[LANDMARK_SHOULDER_R]

    if not (left_shoulder.is_visible and right_shoulder.is_visible):
        return False

    shoulder_y = min(left_shoulder.y, right_shoulder.y)
    return ball_y < shoulder_y


def _compute_average_elbow_angle(
    all_features: list[FrameFeatures],
) -> float:
    """Compute average elbow angle across all frames."""
    angles = [f.elbow_angle for f in all_features if f.elbow_angle is not None]
    if not angles:
        return 0.0
    return sum(angles) / len(angles)


def _compute_average_speed(
    all_features: list[FrameFeatures],
) -> float:
    """Compute average player speed in m/s (approximate from px/s)."""
    speeds = [f.player_speed for f in all_features if f.player_speed is not None]
    if not speeds:
        return 0.0
    avg_px_s = sum(speeds) / len(speeds)
    return avg_px_s / 100.0


def _compute_avg_ball_speed(
    ball_trajectory,
) -> float:
    """Compute average ball speed in km/h."""
    if not ball_trajectory or not ball_trajectory.speed_kmh:
        return 0.0
    speeds = [s for s in ball_trajectory.speed_kmh if s > 0]
    if not speeds:
        return 0.0
    return sum(speeds) / len(speeds)


def _compute_position_optimality(
    all_features: list[FrameFeatures],
) -> float:
    """Compute how close to optimal court positions (0-1)."""
    positions = [f.court_position for f in all_features if f.court_position is not None]
    if not positions:
        return 0.0

    optimal_center = (0.5, 0.5)
    distances = []
    for x, y in positions:
        dist = ((x - optimal_center[0]) ** 2 + (y - optimal_center[1]) ** 2) ** 0.5
        max_dist = (0.5**2 + 0.5**2) ** 0.5
        distances.append(1.0 - dist / max_dist)

    return sum(distances) / len(distances)


def _compute_court_coverage(
    all_features: list[FrameFeatures],
    grid_size: int = 4,
) -> float:
    """Compute percentage of court grid cells visited by player."""
    positions = [f.court_position for f in all_features if f.court_position is not None]
    if not positions:
        return 0.0

    visited: set[tuple[int, int]] = set()
    for x, y in positions:
        cell_x = min(int(x * grid_size), grid_size - 1)
        cell_y = min(int(y * grid_size), grid_size - 1)
        visited.add((cell_x, cell_y))

    total_cells = grid_size * grid_size
    return len(visited) / total_cells


def run_pipeline(
    video_path: str | Path,
    output_path: str | Path | None = None,
    model_size: str = "s",
    device: str = "auto",
    pose_complexity: str = "full",
    skip_pose: bool = False,
    skip_ball: bool = False,
    skip_features: bool = False,
    skip_classifier: bool = False,
    skip_scoring: bool = False,
    scoring_config_path: str | Path | None = None,
    target_fps: int = 25,
    max_dimension: int = 1280,
    show: bool = False,
    show_heatmap: bool = False,
    show_bounces: bool = False,
    show_impacts: bool = False,
    tracknet_weights: str | Path | None = None,
    json_output: str | Path | None = None,
) -> None:
    """Run the full PadelVision detection pipeline on a video."""
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("PadelVision Pipeline")
    logger.info("=" * 60)

    # 1. Load and preprocess video
    logger.info("[1/7] Loading video...")
    preprocessor = VideoPreprocessor(target_fps=target_fps, max_dimension=max_dimension)
    metadata = preprocessor.load(video_path)
    logger.info(
        f"  Duration: {metadata.duration_sec:.1f}s | {metadata.fps:.1f} FPS | {metadata.width}x{metadata.height}"
    )

    court_roi = preprocessor.get_court_roi()
    if court_roi:
        logger.info(
            f"  Court ROI detected: x={court_roi.x:.0f} y={court_roi.y:.0f} "
            f"w={court_roi.width:.0f} h={court_roi.height:.0f}"
        )
    else:
        logger.info("  No court ROI detected, using full frame")

    # 2. Player detection
    logger.info("[2/7] Initializing player detector...")
    detector = PlayerDetector(model_size=model_size, device=device)

    # 3. Pose estimation
    pose_estimator = None
    if not skip_pose:
        logger.info(f"[3/7] Initializing pose estimator (complexity={pose_complexity})...")
        pose_estimator = PoseEstimator(model_complexity=pose_complexity)

    # 4. Ball tracking
    ball_tracker = None
    if not skip_ball:
        weights = tracknet_weights or "models/tracknet/track.pt"
        logger.info(f"[4/7] Initializing ball tracker (weights={weights})...")
        ball_tracker = BallTracker(weights_path=weights, device=device)

    # 5. Feature extraction
    feature_extractor = None
    if not skip_features:
        logger.info("[5/7] Initializing feature extractor...")
        feature_extractor = FeatureExtractor()

    # 6. Shot classification
    shot_classifier = None
    if not skip_classifier:
        logger.info("[6/7] Initializing shot classifier...")
        shot_classifier = ShotClassifier(fps=metadata.fps)

    # 7. Scoring engine
    scoring_engine = None
    if not skip_scoring:
        if scoring_config_path:
            logger.info(f"[7/7] Initializing scoring engine (config={scoring_config_path})...")
            scoring_engine = ScoringEngine.from_file(scoring_config_path)
        else:
            logger.info("[7/7] Initializing scoring engine (defaults)...")
            scoring_engine = ScoringEngine()

    visualizer = Visualizer()

    # Collect all frames for ball tracking (needs full video)
    all_frames: list[np.ndarray] = []
    all_detections: list[list] = []
    all_poses: list[list] = []

    # First pass: collect frames, detections, and poses
    logger.info("Processing frames (pass 1: detection + pose)...")
    process_start = time.time()

    for batch in preprocessor.frame_batches():
        detections_batch = detector.detect_frames(batch.frames, court_roi=court_roi)

        poses_batch: list[list] = [[] for _ in batch.frames]
        if pose_estimator:
            poses_batch = pose_estimator.estimate_batch(batch.frames, detections_batch)

        for i, (frame, detections, poses) in enumerate(zip(batch.frames, detections_batch, poses_batch)):
            all_frames.append(frame)
            all_detections.append(detections)
            all_poses.append(poses)

    # Ball tracking (full video)
    ball_trajectory = None
    if ball_tracker:
        logger.info(f"Tracking ball across {len(all_frames)} frames...")
        ball_trajectory = ball_tracker.track(all_frames, fps=metadata.fps)
        detected = sum(1 for p in ball_trajectory.positions if p.is_detected and not p.interpolated)
        logger.info(
            f"  Ball detected in {detected}/{len(ball_trajectory.positions)} frames "
            f"({detected / max(1, len(ball_trajectory.positions)):.1%})"
        )

    # Feature extraction
    all_impacts: list[ImpactEvent] = []
    player_features: dict[int, list[FrameFeatures]] = {}
    if feature_extractor and ball_trajectory:
        player_tracks = _build_player_tracks(all_detections, all_poses)
        all_impacts = feature_extractor.detect_impacts(player_tracks, ball_trajectory, metadata.fps)
        logger.info(f"  Detected {len(all_impacts)} impact events")

        # Extract features per player and compute ball_above_head
        ball_pos_by_frame = {p.frame_idx: p for p in ball_trajectory.positions}
        for track_id, track in player_tracks.items():
            features = feature_extractor.extract_features(
                track,
                ball_trajectory,
                (metadata.width, metadata.height),
                court_roi,
                metadata.fps,
            )
            # Compute ball_above_head for each feature
            enhanced_features: list[FrameFeatures] = []
            for feat in features:
                det = track.detections.get(feat.frame_idx)
                pose = track.poses.get(feat.frame_idx)
                ball_pos = ball_pos_by_frame.get(feat.frame_idx)
                ball_x = ball_pos.x if ball_pos and ball_pos.is_detected else None
                ball_y = ball_pos.y if ball_pos and ball_pos.is_detected else None
                ball_above = _compute_ball_above_head(det, pose, ball_x, ball_y)
                enhanced_features.append(
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
            player_features[track_id] = enhanced_features

    # Shot classification
    player_shots: dict[int, list[Shot]] = {}
    if shot_classifier and ball_trajectory:
        for track_id, features in player_features.items():
            impacts_for_player = [i for i in all_impacts if i.player_id == track_id]
            shots = shot_classifier.classify_all(features, impacts_for_player, metadata.fps)
            player_shots[track_id] = shots
            if shots:
                logger.info(f"  Player {track_id}: {len(shots)} shots classified")
                shot_types = {}
                for s in shots:
                    shot_types[s.shot_type] = shot_types.get(s.shot_type, 0) + 1
                logger.info(f"    Distribution: {shot_types}")

    # Scoring
    player_scores: dict[int, PlayerScore] = {}
    if scoring_engine and ball_trajectory:
        for track_id, features in player_features.items():
            shots = player_shots.get(track_id, [])
            metrics = ScoringEngine().compute_metrics_from_data(
                shots=shots,
                avg_elbow_angle=_compute_average_elbow_angle(features),
                avg_speed_ms=_compute_average_speed(features),
                avg_ball_speed_kmh=_compute_avg_ball_speed(ball_trajectory),
                avg_position_optimality=_compute_position_optimality(features),
                court_coverage_pct=_compute_court_coverage(features),
            )
            score = scoring_engine.compute_score(metrics)
            player_scores[track_id] = score

    # Second pass: annotate and write output
    logger.info("Annotating output...")
    writer = None
    frame_count = len(all_frames)

    for i, (frame, detections, poses) in enumerate(zip(all_frames, all_detections, all_poses)):
        annotated = draw_detections(frame, detections, poses)

        if ball_trajectory:
            annotated = visualizer.draw_trajectory(annotated, ball_trajectory, i)

            if show_bounces:
                annotated = visualizer.draw_bounces(
                    annotated,
                    ball_trajectory.bounces,
                    ball_trajectory.positions,
                    i,
                )

        if show_impacts and all_impacts:
            annotated = visualizer.draw_impacts(annotated, all_impacts, i)

        if show_heatmap and ball_trajectory:
            annotated = visualizer.overlay_heatmap(annotated, ball_trajectory.positions)

        if court_roi:
            annotated = visualizer.draw_court_roi(annotated, court_roi)

        if output_path:
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, metadata.fps, (w, h))
            writer.write(annotated)

        if show:
            cv2.imshow("PadelVision", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.time() - process_start
    fps_achieved = frame_count / elapsed if elapsed > 0 else 0

    logger.info("-" * 60)
    logger.info(f"Processed {frame_count} frames in {elapsed:.1f}s ({fps_achieved:.1f} FPS)")
    logger.info(f"Total pipeline time: {time.time() - start_time:.1f}s")

    # JSON output
    if json_output:
        _write_json_output(
            json_output,
            metadata,
            ball_trajectory,
            all_impacts,
            player_shots,
            player_scores,
        )

    # Cleanup
    if writer is not None:
        writer.release()
    preprocessor.close()
    if pose_estimator:
        pose_estimator.close()
    if ball_tracker:
        ball_tracker.close()

    if show:
        cv2.destroyAllWindows()

    if output_path:
        logger.info(f"Output saved to: {output_path}")


def _build_player_tracks(
    all_detections: list[list],
    all_poses: list[list],
) -> dict[int, PlayerTrack]:
    """Build player tracks from detection and pose data across all frames."""
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


def _write_json_output(
    json_path: str | Path,
    metadata,
    ball_trajectory,
    impacts: list[ImpactEvent],
    player_shots: dict[int, list[Shot]],
    player_scores: dict[int, PlayerScore],
) -> None:
    """Write analysis results to a JSON file."""
    result: dict = {
        "video_duration_sec": metadata.duration_sec,
        "fps_analyzed": metadata.fps,
    }

    if ball_trajectory:
        result["ball_trajectory"] = {
            "total_frames": len(ball_trajectory.positions),
            "detected_frames": sum(1 for p in ball_trajectory.positions if p.is_detected and not p.interpolated),
            "detection_rate": sum(1 for p in ball_trajectory.positions if p.is_detected and not p.interpolated)
            / max(1, len(ball_trajectory.positions)),
            "bounces": ball_trajectory.bounces,
        }

    result["impacts"] = [
        {
            "frame": imp.frame_idx,
            "player_id": imp.player_id,
            "ball_x": imp.ball_x,
            "ball_y": imp.ball_y,
            "confidence": imp.confidence,
        }
        for imp in impacts
    ]

    if player_shots:
        result["players"] = []
        for track_id in sorted(player_shots.keys()):
            shots = player_shots[track_id]
            player_data: dict = {
                "player_id": track_id,
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
                "shot_distribution": {},
            }
            for s in shots:
                player_data["shot_distribution"][s.shot_type] = player_data["shot_distribution"].get(s.shot_type, 0) + 1

            if track_id in player_scores:
                score = player_scores[track_id]
                player_data["score"] = {
                    "global": score.global_score,
                    "breakdown": score.breakdown,
                }

            result["players"].append(player_data)

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"JSON results saved to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PadelVision — Analyze padel video: players, poses, ball, shots, and scores.",
    )
    parser.add_argument(
        "video",
        type=str,
        help="Path to input video file (.mp4, .mov, .avi)",
    )
    parser.add_argument("--output", "-o", type=str, default=None, help="Output annotated video")
    parser.add_argument("--json", type=str, default=None, help="Output JSON results file")
    parser.add_argument(
        "--model-size",
        type=str,
        default="s",
        choices=["n", "s", "m", "l", "x"],
        help="YOLOv11 model size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Compute device",
    )
    parser.add_argument(
        "--pose-complexity",
        type=str,
        default="full",
        choices=["lite", "full", "heavy"],
        help="MediaPipe Pose complexity",
    )
    parser.add_argument("--no-pose", action="store_true", help="Skip pose estimation")
    parser.add_argument("--no-ball", action="store_true", help="Skip ball tracking")
    parser.add_argument("--no-features", action="store_true", help="Skip feature extraction")
    parser.add_argument("--no-classifier", action="store_true", help="Skip shot classification")
    parser.add_argument("--no-scoring", action="store_true", help="Skip scoring engine")
    parser.add_argument(
        "--scoring-config",
        type=str,
        default=None,
        help="Path to scoring config JSON",
    )
    parser.add_argument("--target-fps", type=int, default=25, help="Target FPS")
    parser.add_argument("--max-dimension", type=int, default=1280, help="Max dimension")
    parser.add_argument("--show", action="store_true", help="Display in real-time")
    parser.add_argument("--heatmap", action="store_true", help="Overlay ball position heatmap")
    parser.add_argument("--show-bounces", action="store_true", help="Mark ball bounces")
    parser.add_argument("--show-impacts", action="store_true", help="Mark player-ball impacts")
    parser.add_argument("--tracknet-weights", type=str, default=None, help="TrackNet weights path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not Path(args.video).exists():
        logger.error(f"Video file not found: {args.video}")
        sys.exit(1)

    run_pipeline(
        video_path=args.video,
        output_path=args.output,
        model_size=args.model_size,
        device=args.device,
        pose_complexity=args.pose_complexity,
        skip_pose=args.no_pose,
        skip_ball=args.no_ball,
        skip_features=args.no_features,
        skip_classifier=args.no_classifier,
        skip_scoring=args.no_scoring,
        scoring_config_path=args.scoring_config,
        target_fps=args.target_fps,
        max_dimension=args.max_dimension,
        show=args.show,
        show_heatmap=args.heatmap,
        show_bounces=args.show_bounces,
        show_impacts=args.show_impacts,
        tracknet_weights=args.tracknet_weights,
        json_output=args.json,
    )


if __name__ == "__main__":
    main()
