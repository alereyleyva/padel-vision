"""Tests for PipelineManager internals."""

from __future__ import annotations

from pathlib import Path

from padelvision.api.pipeline_manager import PipelineArtifacts, PipelineManager
from padelvision.core.result_builder import build_pipeline_result
from padelvision.types import BBox, Keypoint, PlayerDetection, PoseLandmarks, Shot, VideoMetadata


def _make_pose(player_id: int) -> PoseLandmarks:
    return PoseLandmarks(
        keypoints=[Keypoint(x=10.0, y=20.0, visibility=0.9) for _ in range(33)],
        frame_idx=0,
        player_id=player_id,
    )


def test_build_player_tracks_matches_pose_to_player_id() -> None:
    manager = PipelineManager(progress_cb=lambda *_: None)
    det_left = PlayerDetection(track_id=11, bbox=BBox(10, 10, 40, 80), confidence=0.9)
    det_right = PlayerDetection(track_id=22, bbox=BBox(100, 10, 130, 80), confidence=0.9)
    pose_right = _make_pose(22)
    pose_left = _make_pose(11)

    tracks = manager._build_player_tracks(
        all_detections=[[det_left, det_right]],
        all_poses=[[pose_right, pose_left]],
    )

    assert tracks[11].poses[0].player_id == 11
    assert tracks[22].poses[0].player_id == 22
    assert tracks[11].team == 0
    assert tracks[22].team == 1


def test_build_pipeline_result_preserves_multiple_players() -> None:
    manager = PipelineManager(progress_cb=lambda *_: None)
    det_left = PlayerDetection(track_id=11, bbox=BBox(10, 10, 40, 80), confidence=0.9)
    det_right = PlayerDetection(track_id=22, bbox=BBox(100, 10, 130, 80), confidence=0.9)
    tracks = manager._build_player_tracks(
        all_detections=[[det_left, det_right]],
        all_poses=[[_make_pose(11), _make_pose(22)]],
    )

    artifacts = PipelineArtifacts(
        metadata=VideoMetadata(
            source=Path("/tmp/video.mp4"),
            fps=25.0,
            total_frames=100,
            width=1280,
            height=720,
            duration_sec=4.0,
            codec="mp4v",
        ),
        court_roi=None,
        player_tracks=tracks,
        player_shots={
            11: [Shot("drive_forehand", 5, 0.2, 0.8, 0.7)],
            22: [Shot("lob", 8, 0.32, 0.75, 0.6)],
        },
    )

    result = build_pipeline_result(artifacts)

    assert [player["player_id"] for player in result.players] == [11, 22]
    assert result.players[0]["team"] == 0
    assert result.players[1]["team"] == 1
    assert result.players[0]["shots"][0]["type"] == "drive_forehand"
    assert result.players[1]["shots"][0]["type"] == "lob"
