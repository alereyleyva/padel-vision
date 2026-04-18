"""Shot classifier — heuristic baseline + LSTM model stub for classifying padel shots."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from padelvision.types import (
    FrameFeatures,
    ImpactEvent,
    Shot,
    ShotType,
)

logger = logging.getLogger(__name__)

SHOT_CLASSES: list[ShotType] = [
    "drive_forehand",
    "drive_backhand",
    "volea_forehand",
    "volea_backhand",
    "bandeja",
    "smash",
    "lob",
    "unknown",
]

# Ideal elbow angles (degrees) for each shot type at impact
IDEAL_ELBOW_ANGLES: dict[ShotType, float] = {
    "drive_forehand": 160.0,
    "drive_backhand": 150.0,
    "volea_forehand": 140.0,
    "volea_backhand": 130.0,
    "bandeja": 155.0,
    "smash": 170.0,
    "lob": 120.0,
    "unknown": 135.0,
}

# Court position thresholds (normalized y, 0=net, 1=baseline)
NET_ZONE_THRESHOLD = 0.35
BASELINE_THRESHOLD = 0.65

# Ball speed thresholds (px/frame, approximate)
LOW_SPEED_THRESHOLD = 5.0
HIGH_SPEED_THRESHOLD = 20.0

# Shoulder rotation thresholds (degrees)
FOREHAND_THRESHOLD = 10.0
BACKHAND_THRESHOLD = -10.0


class ShotClassifier:
    """Classifies padel shots using heuristic rules.

    Uses geometric features around impact events to determine shot type.
    Also provides an LSTM model stub for future training-based classification.
    """

    def __init__(self, fps: float = 25.0) -> None:
        self._fps = fps
        self._lstm_model: ShotClassifierLSTM | None = None
        self._lstm_ready = False

    def classify_all(
        self,
        features: list[FrameFeatures],
        impacts: list[ImpactEvent],
        fps: float | None = None,
    ) -> list[Shot]:
        """Classify all impacts into shot types.

        Args:
            features: Per-frame features for a single player.
            impacts: Detected impact events for this player.
            fps: Frames per second (overrides constructor value).

        Returns:
            List of classified shots ordered by frame index.
        """
        effective_fps = fps or self._fps
        feature_map = {f.frame_idx: f for f in features}

        shots: list[Shot] = []
        for impact in impacts:
            window = self._get_impact_window(feature_map, impact.frame_idx)
            if not window:
                continue

            shot_type, confidence = self._classify_shot(window, impact)
            quality = self._compute_quality_score(window, impact, shot_type)
            timestamp = impact.frame_idx / effective_fps

            shots.append(
                Shot(
                    shot_type=shot_type,
                    frame_idx=impact.frame_idx,
                    timestamp_sec=round(timestamp, 2),
                    confidence=round(confidence, 2),
                    quality_score=round(quality, 1),
                )
            )

        shots.sort(key=lambda s: s.frame_idx)
        logger.info(f"Classified {len(shots)} shots")
        return shots

    def _get_impact_window(
        self,
        feature_map: dict[int, FrameFeatures],
        impact_frame: int,
        before: int = 10,
        after: int = 15,
    ) -> list[FrameFeatures]:
        """Get features in the window around an impact."""
        window: list[FrameFeatures] = []
        for i in range(impact_frame - before, impact_frame + after + 1):
            if i in feature_map:
                window.append(feature_map[i])
        return window

    def _classify_shot(
        self,
        window: list[FrameFeatures],
        impact: ImpactEvent,
    ) -> tuple[ShotType, float]:
        """Classify a shot based on heuristic rules."""
        # Find the frame closest to the impact
        impact_feature = next((f for f in window if f.frame_idx == impact.frame_idx), None)
        if impact_feature is None:
            return "unknown", 0.0

        is_overhead = self._is_overhead_shot(window, impact)
        is_net_zone = self._is_near_net(impact_feature)
        is_baseline = self._is_near_baseline(impact_feature)
        is_forehand = self._is_forehand(impact_feature)
        ball_speed = self._estimate_ball_speed(window)

        # Overhead shots
        if is_overhead:
            if ball_speed > HIGH_SPEED_THRESHOLD:
                return "smash", 0.85
            return "bandeja", 0.75

        # Net zone → volea
        if is_net_zone:
            if is_forehand:
                return "volea_forehand", 0.70
            return "volea_backhand", 0.70

        # Baseline → drive
        if is_baseline:
            if is_forehand:
                return "drive_forehand", 0.75
            return "drive_backhand", 0.75

        # Middle zone — use ball speed and elbow angle
        if ball_speed < LOW_SPEED_THRESHOLD:
            return "lob", 0.60

        if is_forehand:
            return "drive_forehand", 0.55
        return "drive_backhand", 0.55

    def _is_overhead_shot(self, window: list[FrameFeatures], impact: ImpactEvent) -> bool:
        """Check if the ball was above the player's head at impact."""
        impact_feature = next((f for f in window if f.frame_idx == impact.frame_idx), None)
        if impact_feature is None:
            return False

        return impact_feature.ball_above_head

    def _is_near_net(self, feature: FrameFeatures) -> bool:
        """Check if player is near the net (low y in court position)."""
        if feature.court_position is None:
            return False
        _, y_norm = feature.court_position
        return y_norm < NET_ZONE_THRESHOLD

    def _is_near_baseline(self, feature: FrameFeatures) -> bool:
        """Check if player is near the baseline (high y in court position)."""
        if feature.court_position is None:
            return False
        _, y_norm = feature.court_position
        return y_norm > BASELINE_THRESHOLD

    def _is_forehand(self, feature: FrameFeatures) -> bool:
        """Determine if the shot is likely forehand based on shoulder rotation."""
        if feature.shoulder_rotation is None:
            return True  # Default to forehand if unknown
        return feature.shoulder_rotation > BACKHAND_THRESHOLD

    def _estimate_ball_speed(self, window: list[FrameFeatures]) -> float:
        """Estimate ball speed from distance changes around impact."""
        distances = [f.distance_to_ball for f in window if f.distance_to_ball is not None]

        if len(distances) < 3:
            return 0.0

        # Speed is approximated by the rate of change of distance to ball
        changes = [abs(distances[i] - distances[i - 1]) for i in range(1, len(distances))]
        return max(changes) if changes else 0.0

    def _compute_quality_score(
        self,
        window: list[FrameFeatures],
        impact: ImpactEvent,
        shot_type: ShotType,
    ) -> float:
        """Compute quality score for a classified shot.

        Quality = 0.6 * technique_quality + 0.4 * classification_confidence
        """
        impact_feature = next((f for f in window if f.frame_idx == impact.frame_idx), None)
        if impact_feature is None:
            return 0.0

        # Technique quality: how close is the elbow angle to ideal?
        technique_quality = 0.5  # Default
        if impact_feature.elbow_angle is not None:
            ideal = IDEAL_ELBOW_ANGLES.get(shot_type, 135.0)
            deviation = abs(impact_feature.elbow_angle - ideal)
            technique_quality = max(0.0, 1.0 - deviation / 90.0)

        # Classification confidence (heuristic)
        confidence = 0.6  # Baseline confidence for heuristic
        if impact_feature.ball_above_head and shot_type in ("smash", "bandeja"):
            confidence = 0.8
        if impact_feature.court_position is not None:
            confidence = min(confidence + 0.1, 1.0)

        return 0.6 * technique_quality + 0.4 * confidence

    def load_lstm_model(self, checkpoint_path: str) -> None:
        """Load a trained LSTM model from checkpoint.

        Args:
            checkpoint_path: Path to the PyTorch checkpoint file.
        """
        try:
            self._lstm_model = ShotClassifierLSTM(n_classes=len(SHOT_CLASSES))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self._lstm_model.load_state_dict(checkpoint["model_state_dict"])
            self._lstm_model.eval()
            self._lstm_ready = True
            logger.info(f"Loaded LSTM model from {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            self._lstm_ready = False

    def classify_with_lstm(
        self,
        keypoints_sequence: torch.Tensor,
    ) -> tuple[ShotType, float]:
        """Classify a shot using the LSTM model.

        Args:
            keypoints_sequence: Tensor of shape (1, n_frames, 66) — 33 keypoints × 2 coords.

        Returns:
            (shot_type, confidence) tuple.
        """
        if not self._lstm_ready or self._lstm_model is None:
            return "unknown", 0.0

        with torch.no_grad():
            logits = self._lstm_model(keypoints_sequence)
            probs = torch.softmax(logits, dim=1)
            pred_idx = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_idx].item())

        shot_type = SHOT_CLASSES[pred_idx] if pred_idx < len(SHOT_CLASSES) else "unknown"
        return shot_type, confidence


class ShotClassifierLSTM(nn.Module):
    """LSTM-based shot classifier with attention mechanism.

    Input:  (batch, n_frames, 66) — 33 keypoints × (x, y)
    Output: (batch, n_classes) logits
    """

    def __init__(
        self,
        input_size: int = 66,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_classes: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        return self.classifier(attn_out[:, -1, :])
