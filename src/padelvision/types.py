"""Shared data types for the PadelVision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


@dataclass(frozen=True)
class BoundingBox:
    """ROI specified by top-left corner + dimensions."""

    x: float
    y: float
    width: float
    height: float

    @property
    def as_bbox(self) -> BBox:
        return BBox(x1=self.x, y1=self.y, x2=self.x + self.width, y2=self.y + self.height)


@dataclass(frozen=True)
class VideoMetadata:
    """Metadata extracted from a video file."""

    source: Path
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float
    codec: str

    @property
    def is_valid_duration(self) -> bool:
        return 5.0 <= self.duration_sec <= 120.0

    @property
    def needs_fps_normalization(self, target_fps: int = 25) -> bool:
        return self.fps > target_fps + 2 or self.fps < target_fps - 2


@dataclass(frozen=True)
class FrameBatch:
    """A batch of video frames with metadata."""

    frames: list  # list[np.ndarray] — can't type numpy easily in dataclass
    start_idx: int
    timestamp_sec: float


@dataclass(frozen=True)
class Keypoint:
    """A single 2D keypoint with visibility."""

    x: float
    y: float
    visibility: float = 1.0

    @property
    def is_visible(self) -> bool:
        return self.visibility > 0.5


@dataclass
class PoseLandmarks:
    """33 MediaPipe Pose keypoints for a single person in a single frame."""

    keypoints: list[Keypoint] = field(default_factory=list)
    frame_idx: int = 0

    MEDIAPIPE_KEYPOINT_NAMES: list[str] = field(
        default_factory=lambda: [
            "nose",
            "left_eye_inner",
            "left_eye",
            "left_eye_outer",
            "right_eye_inner",
            "right_eye",
            "right_eye_outer",
            "left_ear",
            "right_ear",
            "mouth_left",
            "mouth_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_thumb",
            "right_thumb",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ]
    )

    def get(self, name: str) -> Keypoint | None:
        try:
            idx = self.MEDIAPIPE_KEYPOINT_NAMES.index(name)
            return self.keypoints[idx] if idx < len(self.keypoints) else None
        except ValueError:
            return None


@dataclass(frozen=True)
class PlayerDetection:
    """A detected player in a single frame."""

    track_id: int
    bbox: BBox
    confidence: float
    team: int | None = None


@dataclass
class PoseResult:
    """Pose estimation results for all detected players in a single frame."""

    frame_idx: int
    poses: list[PoseLandmarks]


@dataclass(frozen=True)
class BallPosition:
    """Ball position in a single frame."""

    frame_idx: int
    x: float | None
    y: float | None
    confidence: float = 0.0
    interpolated: bool = False

    @property
    def is_detected(self) -> bool:
        return self.x is not None and self.y is not None


@dataclass
class BallTrajectory:
    """Full ball trajectory across all frames."""

    positions: list[BallPosition] = field(default_factory=list)
    bounces: list[int] = field(default_factory=list)
    speed_kmh: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class PlayerTrack:
    """A player's track across multiple frames."""

    track_id: int
    team: int | None
    detections: dict[int, PlayerDetection]  # frame_idx -> detection
    poses: dict[int, PoseLandmarks]  # frame_idx -> pose


ShotType = Literal[
    "drive_forehand",
    "drive_backhand",
    "volea_forehand",
    "volea_backhand",
    "bandeja",
    "smash",
    "lob",
    "unknown",
]


@dataclass(frozen=True)
class Shot:
    """A classified shot with metadata."""

    shot_type: ShotType
    frame_idx: int
    timestamp_sec: float
    confidence: float
    quality_score: float = 0.0


@dataclass(frozen=True)
class PlayerScore:
    """Playtomic-style score (0-7) with per-dimension breakdown."""

    global_score: float
    consistency: float = 0.0
    technique: float = 0.0
    mobility: float = 0.0
    power: float = 0.0
    positioning: float = 0.0

    @property
    def breakdown(self) -> dict[str, float]:
        return {
            "consistency": self.consistency,
            "technique": self.technique,
            "mobility": self.mobility,
            "power": self.power,
            "positioning": self.positioning,
        }


StrokePhase = Literal["loading", "impact", "followthrough", "none"]


@dataclass(frozen=True)
class FrameFeatures:
    """Geometric and semantic features for one player in one frame."""

    frame_idx: int
    elbow_angle: float | None = None
    shoulder_rotation: float | None = None
    hip_rotation: float | None = None
    knee_bend: float | None = None
    weight_transfer: float | None = None
    court_position: tuple[float, float] | None = None
    player_speed: float | None = None
    distance_to_ball: float | None = None
    stroke_phase: StrokePhase = "none"
    ball_above_head: bool = False


@dataclass(frozen=True)
class ImpactEvent:
    """A detected impact between a player and the ball."""

    frame_idx: int
    player_id: int
    ball_x: float
    ball_y: float
    confidence: float


@dataclass
class PlayerMetrics:
    """Aggregated metrics for a player, used to compute the score."""

    shot_success_rate: float = 0.0
    avg_elbow_angle_quality: float = 0.0
    avg_shoulder_rotation_quality: float = 0.0
    avg_speed_ms: float = 0.0
    court_coverage_pct: float = 0.0
    avg_ball_speed_kmh: float = 0.0
    avg_position_optimality: float = 0.0
    total_shots: int = 0
    shot_distribution: dict[str, int] = field(default_factory=dict)
