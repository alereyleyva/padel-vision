#!/usr/bin/env python3
"""Download pretrained model weights for PadelVision.

Downloads:
  - TrackNet v2 (track.pt) — ball tracking model weights

Models already auto-downloaded by their libraries:
  - YOLOv11 (ultralytics) — player detection
  - MediaPipe Pose — pose estimation

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --tracknet-only
    python scripts/download_models.py --output-dir custom/path
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

TRACKNET_WEIGHTS_URL = "https://github.com/ChgygLin/TrackNetV2-pytorch/raw/master/tf2torch/track.pt"
TRACKNET_DEFAULT_DIR = "models/tracknet"


def download_tracknet(output_dir: str) -> Path:
    """Download TrackNet v2 pretrained weights."""
    output_path = Path(output_dir) / "track.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"TrackNet weights already exist at {output_path}")
        return output_path

    print(f"Downloading TrackNet v2 weights to {output_path}...")
    print(f"  URL: {TRACKNET_WEIGHTS_URL}")

    urllib.request.urlretrieve(TRACKNET_WEIGHTS_URL, str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Downloaded: {size_mb:.1f} MB")
    return output_path


def verify_yolo() -> None:
    """Verify YOLOv11 model weights can be auto-downloaded."""
    try:
        from ultralytics.models.yolo.model import YOLO

        model = YOLO("yolo11s.pt")
        print(f"YOLOv11 weights verified: {model.model_name}")
    except Exception as e:
        print(f"Warning: YOLOv11 auto-download failed: {e}")


def verify_mediapipe() -> None:
    """Verify MediaPipe pose model exists."""
    model_dir = Path("models/mediapipe")
    expected = ["pose_landmarker_lite.task", "pose_landmarker_full.task", "pose_landmarker_heavy.task"]
    for name in expected:
        path = model_dir / name
        if path.exists():
            print(f"MediaPipe model verified: {path}")
        else:
            print(f"Warning: MediaPipe model missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download PadelVision model weights")
    parser.add_argument("--tracknet-only", action="store_true", help="Only download TrackNet weights")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=TRACKNET_DEFAULT_DIR,
        help="Output directory for TrackNet weights",
    )
    args = parser.parse_args()

    download_tracknet(args.output_dir)

    if not args.tracknet_only:
        verify_yolo()
        verify_mediapipe()

    print("\nDone! All requested models are ready.")


if __name__ == "__main__":
    main()
