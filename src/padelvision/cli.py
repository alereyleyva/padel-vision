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
from padelvision.core.visualizer import Visualizer
from padelvision.types import ImpactEvent, PoseLandmarks

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


def run_pipeline(
    video_path: str | Path,
    output_path: str | Path | None = None,
    model_size: str = "s",
    device: str = "auto",
    pose_complexity: str = "full",
    skip_pose: bool = False,
    skip_ball: bool = False,
    skip_features: bool = False,
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
    logger.info("[1/5] Loading video...")
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
    logger.info("[2/5] Initializing player detector...")
    detector = PlayerDetector(model_size=model_size, device=device)

    # 3. Pose estimation
    pose_estimator = None
    if not skip_pose:
        logger.info(f"[3/5] Initializing pose estimator (complexity={pose_complexity})...")
        pose_estimator = PoseEstimator(model_complexity=pose_complexity)

    # 4. Ball tracking
    ball_tracker = None
    if not skip_ball:
        weights = tracknet_weights or "models/tracknet/track.pt"
        logger.info(f"[4/5]Initializing ball tracker (weights={weights})...")
        ball_tracker = BallTracker(weights_path=weights, device=device)

    # 5. Feature extraction
    feature_extractor = None
    if not skip_features:
        logger.info("[5/5] Initializing feature extractor...")
        feature_extractor = FeatureExtractor()

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
    if feature_extractor and ball_trajectory:
        # Build player tracks for feature extraction
        player_tracks = _build_player_tracks(all_detections, all_poses)
        all_impacts = feature_extractor.detect_impacts(player_tracks, ball_trajectory, metadata.fps)
        logger.info(f"  Detected {len(all_impacts)} impact events")

    # Second pass: annotate and write output
    logger.info("Annotating output...")
    writer = None
    frame_count = len(all_frames)

    for i, (frame, detections, poses) in enumerate(zip(all_frames, all_detections, all_poses)):
        annotated = draw_detections(frame, detections, poses)

        if ball_trajectory:
            annotated = visualizer.draw_trajectory(annotated, ball_trajectory, i)

            if show_bounces:
                annotated = visualizer.draw_bounces(annotated, ball_trajectory.bounces, ball_trajectory.positions, i)

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
    if json_output and ball_trajectory:
        _write_json_output(json_output, metadata, ball_trajectory, all_impacts)

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
) -> dict:
    """Build player tracks from detection and pose data across all frames."""
    from padelvision.types import PlayerTrack

    tracks: dict[int, PlayerTrack] = {}

    for frame_idx, (detections, poses) in enumerate(zip(all_detections, all_poses)):
        for detection in detections:
            tid = detection.track_id
            if tid not in tracks:
                tracks[tid] = PlayerTrack(track_id=tid, team=detection.team, detections={}, poses={})
            tracks[tid].detections[frame_idx] = detection

        for pose in poses:
            # Match pose to player by proximity (simplified)
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
) -> None:
    """Write analysis results to a JSON file."""
    result = {
        "video_duration_sec": metadata.duration_sec,
        "fps_analyzed": metadata.fps,
        "ball_trajectory": {
            "total_frames": len(ball_trajectory.positions),
            "detected_frames": sum(1 for p in ball_trajectory.positions if p.is_detected and not p.interpolated),
            "detection_rate": sum(1 for p in ball_trajectory.positions if p.is_detected and not p.interpolated)
            / max(1, len(ball_trajectory.positions)),
            "bounces": ball_trajectory.bounces,
        },
        "impacts": [
            {
                "frame": imp.frame_idx,
                "player_id": imp.player_id,
                "ball_x": imp.ball_x,
                "ball_y": imp.ball_y,
                "confidence": imp.confidence,
            }
            for imp in impacts
        ],
    }

    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"JSON results saved to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PadelVision — Analyze padel video: players, poses, ball, and features.",
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
