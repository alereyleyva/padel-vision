"""CLI entry point for PadelVision pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from padelvision.core.pipeline_runtime import PipelineArtifacts, PipelineConfig, run_pipeline_runtime
from padelvision.core.result_builder import build_pipeline_result
from padelvision.core.visualizer import Visualizer
from padelvision.types import PoseLandmarks

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
    config = PipelineConfig(
        model_size=model_size,
        device=device,
        pose_complexity=pose_complexity,
        skip_pose=skip_pose,
        skip_ball=skip_ball,
        skip_features=skip_features,
        skip_classifier=skip_classifier,
        skip_scoring=skip_scoring,
        scoring_config_path=scoring_config_path,
        target_fps=target_fps,
        max_dimension=max_dimension,
        tracknet_weights=tracknet_weights,
    )

    logger.info("=" * 60)
    logger.info("PadelVision Pipeline")
    logger.info("=" * 60)

    visualizer = Visualizer()
    process_start = time.time()
    artifacts = run_pipeline_runtime(video_path, config)

    # Second pass: annotate and write output
    logger.info("Annotating output...")
    writer = None
    frame_count = len(artifacts.frames)

    for i, (frame, detections, poses) in enumerate(
        zip(artifacts.frames, artifacts.detections_per_frame, artifacts.poses_per_frame)
    ):
        annotated = draw_detections(frame, detections, poses)

        if artifacts.ball_trajectory:
            annotated = visualizer.draw_trajectory(annotated, artifacts.ball_trajectory, i)

            if show_bounces:
                annotated = visualizer.draw_bounces(
                    annotated,
                    artifacts.ball_trajectory.bounces,
                    artifacts.ball_trajectory.positions,
                    i,
                )

        if show_impacts and artifacts.impacts:
            annotated = visualizer.draw_impacts(annotated, artifacts.impacts, i)

        if show_heatmap and artifacts.ball_trajectory:
            annotated = visualizer.overlay_heatmap(annotated, artifacts.ball_trajectory.positions)

        if artifacts.court_roi:
            annotated = visualizer.draw_court_roi(annotated, artifacts.court_roi)

        if output_path:
            if writer is None:
                h, w = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, artifacts.metadata.fps, (w, h))
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
        _write_json_output(json_output, artifacts)

    # Cleanup
    if writer is not None:
        writer.release()

    if show:
        cv2.destroyAllWindows()

    if output_path:
        logger.info(f"Output saved to: {output_path}")
def _write_json_output(
    json_path: str | Path,
    artifacts: PipelineArtifacts,
) -> None:
    """Write the shared pipeline result to a JSON file."""
    result = build_pipeline_result(artifacts)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
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
