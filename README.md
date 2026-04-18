# PadelVision

> Computer vision applied to padel — analyze video clips and produce Playtomic-style skill scores, shot classification, and ball trajectory analysis.

## What It Does

PadelVision takes a padel match video and runs it through a multi-stage computer vision pipeline:

```
VIDEO INPUT (.mp4 / .mov / .avi)
        │
        ▼
[1] PREPROCESSOR       Load video, normalize FPS, extract frame batches
        │
        ▼
[2] PLAYER DETECTOR    YOLOv11 + ByteTrack → detect & track all 4 players
        │                    \
        │                     [3] BALL TRACKER  TrackNet v2 → ball position per frame
        │
        ▼
[4] POSE ESTIMATOR     MediaPipe Pose → 33 keypoints per player per frame
        │
        ▼
[5] FEATURE EXTRACTOR  Elbow angles, shoulder rotation, court position, speed, etc.
        │
        ▼
[6] IMPACT DETECTION   Find exact frames where players hit the ball
        │
        ▼
OUTPUT: Annotated video + JSON with ball trajectory, impacts, and features
```

## Quick Start

### 1. Install dependencies

```bash
uv sync
```

### 2. Download TrackNet model weights (required for ball tracking)

```bash
python scripts/download_models.py
```

This downloads ~43MB of pretrained weights to `models/tracknet/track.pt`.

> **Note:** YOLOv11 and MediaPipe models are auto-downloaded on first use.

### 3. Run the pipeline

```bash
# Basic: player detection + pose estimation (fastest, no ball tracking needed)
python analyze.py data/raw/padel_clip.mp4 -o annotated.mp4

# Full pipeline: players + ball tracking + features
python analyze.py data/raw/padel_clip.mp4 -o annotated.mp4 --heatmap --show-bounces --show-impacts

# Show results in real-time
python analyze.py data/raw/padel_clip.mp4 --show

# Export JSON results
python analyze.py data/raw/padel_clip.mp4 --json results.json

# Verbose logging
python analyze.py data/raw/padel_clip.mp4 -o annotated.mp4 -v
```

### 4. CLI Options

| Flag | Description |
|---|---|
| `--output`, `-o` | Path to output annotated video |
| `--json` | Export analysis results to JSON |
| `--model-size` | YOLOv11 size: `n` (fastest) → `x` (most accurate). Default: `s` |
| `--device` | Compute device: `auto`, `cpu`, `mps`, `cuda`. Default: `auto` |
| `--pose-complexity` | MediaPipe model: `lite`, `full`, `heavy`. Default: `full` |
| `--no-pose` | Skip pose estimation |
| `--no-ball` | Skip ball tracking |
| `--no-features` | Skip feature extraction |
| `--heatmap` | Overlay ball position heatmap on output |
| `--show-bounces` | Mark ball bounce points |
| `--show-impacts` | Mark player-ball impact events |
| `--target-fps` | Target FPS for frame extraction. Default: 25 |
| `--max-dimension` | Max frame dimension for resize. Default: 1280 |
| `--show` | Display frames in real-time |
| `--verbose`, `-v` | Verbose logging |

## Project Structure

```
padel-vision/
├── analyze.py                      # CLI entry point
├── pyproject.toml                  # Dependencies and tool config
├── pyrightconfig.json              # Type checking configuration
├── scripts/
│   └── download_models.py          # Download pretrained model weights
├── src/padelvision/
│   ├── __init__.py
│   ├── cli.py                      # Pipeline orchestration + CLI
│   ├── types.py                    # Shared dataclasses (22 types)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessor.py         # Video loading, FPS normalization, batch iteration
│   │   ├── detector.py             # Player detection (YOLOv11 + ByteTrack)
│   │   ├── ball_tracker.py         # Ball tracking (TrackNet v2 + interpolation)
│   │   ├── pose_estimator.py       # Pose estimation (MediaPipe Pose)
│   │   ├── feature_extractor.py    # Geometric features + impact detection
│   │   ├── tracknet_model.py       # TrackNet v2 PyTorch model definition
│   │   └── visualizer.py           # Trajectory, heatmap, bounce/impact overlays
│   └── api/                        # (Phase 4 placeholder)
├── tests/                          # 79 unit + integration tests
│   ├── conftest.py
│   └── core/
│       ├── test_types.py
│       ├── test_preprocessor.py
│       ├── test_detector.py
│       ├── test_pose_estimator.py
│       ├── test_ball_tracker.py
│       ├── test_feature_extractor.py
│       └── test_visualizer.py
└── poc/                            # Archived proof-of-concept scripts
```

## Architecture

### Core Modules

| Module | Input | Output | Key Tech |
|---|---|---|---|
| **VideoPreprocessor** | Video file | Frame batches + metadata | OpenCV |
| **PlayerDetector** | Frames | Player bboxes + track IDs | YOLOv11 + ByteTrack |
| **BallTracker** | Frames | Ball trajectory (x, y per frame) | TrackNet v2 (PyTorch) |
| **PoseEstimator** | Frames + detections | 33 keypoints per player | MediaPipe Pose |
| **FeatureExtractor** | Detections + poses + ball | Per-frame features + impacts | NumPy + SciPy |
| **Visualizer** | Frames + analysis data | Annotated frames | OpenCV |

### Ball Tracking Details

TrackNet v2 is a VGG16 encoder + U-Net decoder that processes 3 consecutive frames to produce a ball position heatmap. Key features:

- **Input:** 3 frames → `(B, 9, 288, 512)` tensor
- **Output:** 3 heatmaps → `(B, 3, 288, 512)` with values in `[0, 1]`
- **Post-processing:** Threshold heatmap → find contours → extract centroid
- **Interpolation:** Cubic spline fills gaps when ball is not detected
- **Bounce detection:** Savitzky-Golay smoothed velocity sign changes
- **Speed computation:** Pixel distance / time between consecutive detections

### Feature Extraction

9 geometric features computed per player per frame:

| Feature | Calculation | Use Case |
|---|---|---|
| `elbow_angle` | Wrist-elbow-shoulder angle | Shot classification, technique quality |
| `shoulder_rotation` | Shoulder line vs horizontal | Body rotation technique |
| `hip_rotation` | Hip line vs horizontal | Weight transfer |
| `knee_bend` | Hip-knee-ankle angle | Stance quality |
| `weight_transfer` | CoM displacement over 3 frames | Dynamism, anticipation |
| `court_position` | Normalized position in court [0,1] | Tactics, positioning |
| `player_speed` | CoM distance / delta_t | Athletic level |
| `distance_to_ball` | Player-to-ball distance | Timing of contact |
| `stroke_phase` | Loading / Impact / Followthrough | Shot classification input |

### Impact Detection

Impacts are detected by finding the frame where a player's bounding box center is closest to the ball, validated with ball velocity changes. A Savitzky-Golay filter smooths ball velocity to reduce noise.

## Testing

```bash
# Run all tests
uv run pytest

# Run specific module tests
uv run pytest tests/core/test_ball_tracker.py -v

# Run with coverage
uv run pytest --cov=src/padelvision

# Run tests matching a pattern
uv run pytest -k "elbow" -v
```

## Type Checking

```bash
# Run pyright (0 errors expected)
uv run pyright

# Run ruff linter
uv run ruff check src/padelvision/ tests/

# Run ruff formatter
uv run ruff format src/padelvision/ tests/
```

## Requirements

- **Python:** >= 3.12
- **OS:** macOS (Apple Silicon recommended), Linux, Windows
- **GPU:** Apple Silicon (MPS) recommended. CUDA supported. CPU fallback available.
- **Memory:** 8GB+ RAM (16GB+ recommended for large videos)

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| ultralytics | >= 8.4.38 | YOLOv11 player detection |
| opencv-python | >= 4.13.0 | Video I/O, image processing |
| mediapipe | >= 0.10.33 | Pose estimation |
| torch | >= 2.11.0 | TrackNet inference (MPS) |
| scipy | >= 1.17.1 | Spline interpolation, Savitzky-Golay |
| numpy | >= 2.4.4 | Array operations |

## Roadmap

| Phase | Status | Description |
|---|---|---|
| **Phase 1** | ✅ Done | Pipeline foundations: player detection + pose estimation |
| **Phase 2** | ✅ Done | Ball tracking + feature engineering |
| **Phase 3** | ⏳ Next | Shot classifier (LSTM) + Scoring Engine (0-7 Playtomic-style) |
| **Phase 4** | ⏳ Future | FastAPI REST API + Celery async jobs + Docker deploy |

## License

AGPL-3.0
