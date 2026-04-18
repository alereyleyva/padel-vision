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
[7] SHOT CLASSIFIER    Heuristic rules → classify each impact (drive, volea, smash, etc.)
        │
        ▼
[8] SCORING ENGINE     Aggregate metrics → Playtomic-style score 0–7 per dimension
        │
        ▼
OUTPUT: Annotated video + JSON with ball trajectory, shots, scores, and features
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

# Full pipeline: players + ball + features + shots + scores
python analyze.py data/raw/padel_clip.mp4 -o annotated.mp4 --heatmap --show-bounces --show-impacts --json results.json

# Show results in real-time
python analyze.py data/raw/padel_clip.mp4 --show

# Export JSON results with scores and shot distribution
python analyze.py data/raw/padel_clip.mp4 --json results.json

# Custom scoring config
python analyze.py data/raw/padel_clip.mp4 --json results.json --scoring-config scoring.json

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
| `--no-classifier` | Skip shot classification |
| `--no-scoring` | Skip scoring engine |
| `--scoring-config` | Path to custom scoring config JSON |
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
│   ├── types.py                    # Shared dataclasses (22+ types)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessor.py         # Video loading, FPS normalization, batch iteration
│   │   ├── detector.py             # Player detection (YOLOv11 + ByteTrack)
│   │   ├── ball_tracker.py         # Ball tracking (TrackNet v2 + interpolation)
│   │   ├── pose_estimator.py       # Pose estimation (MediaPipe Pose)
│   │   ├── feature_extractor.py    # Geometric features + impact detection
│   │   ├── shot_classifier.py      # Heuristic shot classification + LSTM stub
│   │   ├── scoring_engine.py       # Playtomic-style scoring 0–7
│   │   ├── tracknet_model.py       # TrackNet v2 PyTorch model definition
│   │   └── visualizer.py           # Trajectory, heatmap, bounce/impact overlays
│   └── api/                        # FastAPI REST API + Celery tasks + PipelineManager
│       ├── main.py                 # FastAPI app with /analyze, /jobs, /results endpoints
│       ├── models.py               # Pydantic request/response schemas
│       ├── pipeline_manager.py     # Programmatic pipeline orchestration with progress
│       ├── tasks.py                # Celery async task definitions
│       └── storage.py              # JSON result storage + SQLite metadata
├── docker/
│   ├── Dockerfile                  # API server container
│   ├── Dockerfile.worker           # Celery worker container
│   └── docker-compose.yml          # Full stack: API + Worker + Redis
├── tests/                          # 139 unit + integration tests
│   ├── conftest.py
│   ├── api/
│   │   ├── test_models.py
│   │   ├── test_storage.py
│   │   └── test_api.py
│   └── core/
│       ├── test_types.py
│       ├── test_preprocessor.py
│       ├── test_detector.py
│       ├── test_pose_estimator.py
│       ├── test_ball_tracker.py
│       ├── test_feature_extractor.py
│       ├── test_visualizer.py
│       └── test_shot_classifier.py
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
| **ShotClassifier** | Features + impacts | Classified shots (8 types) | Heuristic rules + LSTM stub |
| **ScoringEngine** | Player metrics | Playtomic-style score 0–7 | Percentile normalization |
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
| `ball_above_head` | Ball y vs shoulder y | Overhead shot detection |
| `stroke_phase` | Loading / Impact / Followthrough | Shot classification input |

### Shot Classification

Heuristic baseline classifies 8 shot types using geometric features around each impact:

| Shot Type | Detection Logic |
|---|---|
| **smash** | Ball above head + high ball speed |
| **bandeja** | Ball above head + moderate speed |
| **volea_forehand/backhand** | Near net zone + shoulder rotation |
| **drive_forehand/backhand** | Near baseline + shoulder rotation |
| **lob** | Middle zone + low ball speed |
| **unknown** | Insufficient confidence |

Quality score per shot = `0.6 × technique_quality + 0.4 × classification_confidence`.

An LSTM model stub (`ShotClassifierLSTM`) is included and ready for training when annotated data is available.

### Scoring Engine

Playtomic-style score (0–7) computed from 5 weighted dimensions:

| Dimension | Metric | Weight |
|---|---|---|
| **Consistency** | Shot success rate | 25% |
| **Technique** | Elbow angle quality vs ideal | 30% |
| **Mobility** | Average player speed | 20% |
| **Power** | Average ball speed | 15% |
| **Positioning** | Court position optimality | 10% |

Each dimension is normalized to [0, 7] using percentile-based interpolation. Default percentiles are calibrated for amateur-to-pro range. Custom calibration can be loaded via `--scoring-config`.

### Impact Detection

Impacts are detected by finding the frame where a player's bounding box center is closest to the ball, validated with ball velocity changes. A Savitzky-Golay filter smooths ball velocity to reduce noise.

## API Server

PadelVision exposes a REST API for async video analysis:

```bash
# Start the full stack (API + Celery worker + Redis)
docker-compose -f docker/docker-compose.yml up

# Or run locally
uv run fastapi run src/padelvision/api/main.py
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/analyze` | Upload video for async analysis → returns `job_id` |
| `POST` | `/analyze/sync` | Upload and analyze synchronously (testing/short clips) |
| `GET` | `/jobs/{job_id}` | Check job status and progress |
| `GET` | `/results/{job_id}` | Retrieve full analysis result |
| `DELETE` | `/jobs/{job_id}` | Delete a job and its results |
| `GET` | `/health` | Health check |

### Example Usage

```bash
# Upload video for analysis
curl -X POST http://localhost:8000/analyze \
  -F "video=@clip.mp4" \
  -F 'options={"include_ball": true, "include_poses": false}'

# → {"job_id": "abc123", "status": "queued", "estimated_seconds": 45}

# Check status
curl http://localhost:8000/jobs/abc123
# → {"job_id": "abc123", "status": "processing", "progress": 0.5}

# Get results
curl http://localhost:8000/results/abc123
# → Full JSON with scores, shots, ball trajectory
```

Swagger UI is available at `http://localhost:8000/docs`.

## Testing

```bash
# Run all tests
uv run pytest

# Run API tests only
uv run pytest tests/api/ -v

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
| **Phase 3** | ✅ Done | Shot classifier + Scoring Engine (0-7 Playtomic-style) |
| **Phase 4** | ✅ Done | FastAPI REST API + Celery async jobs + Docker deploy |
| **Phase 5** | ⏳ Next | Model training, data collection, accuracy improvements |

## License

AGPL-3.0
