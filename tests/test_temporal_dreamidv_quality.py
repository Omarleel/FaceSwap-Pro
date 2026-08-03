from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from faceswap_pro.dreamidv_backend import DreamIDVClipPlanner, DreamIDVOptions
from faceswap_pro.identity import SourceReferenceSample, select_diverse_source_references
from faceswap_pro.modeling import FaceData
from faceswap_pro.temporal_video import (
    TargetAwareCompositor,
    TargetFaceInstance,
    TargetTrackFrame,
    analyze_target_track,
)


def _options(tmp_path: Path, **overrides) -> DreamIDVOptions:
    values = {
        "repository_path": str(tmp_path / "DreamID-V"),
        "wan_checkpoint_dir": str(tmp_path / "Wan"),
        "python_executable": sys.executable,
        "variant": "faster",
        "size": "832*480",
        "frame_num": 49,
        "sample_fps": 16,
        "sample_steps": 16,
        "chunking": True,
        "chunk_overlap_frames": 17,
        "scene_aware_chunking": True,
        "scene_cut_window_frames": 12,
        "seed": 100,
        "seed_mode": "absolute_frame",
    }
    values.update(overrides)
    config = SimpleNamespace(engine=SimpleNamespace(options=values))
    return DreamIDVOptions.from_config(config)


def test_clip_planner_uses_overlap_and_absolute_timeline_seeds(tmp_path):
    planner = DreamIDVClipPlanner(_options(tmp_path))

    clips = planner.plan_frames(120, ())

    assert [clip.start_frame for clip in clips] == [0, 32, 64, 96]
    assert [clip.overlap_before for clip in clips] == [0, 17, 17, 17]
    assert [clip.seed for clip in clips] == [100, 132, 164, 196]
    assert clips[-1].valid_frames == 24


def test_clip_planner_prefers_nearby_scene_cut_and_marks_hard_boundary(tmp_path):
    planner = DreamIDVClipPlanner(_options(tmp_path))

    clips = planner.plan_frames(100, (29,))

    assert clips[1].start_frame == 29
    assert clips[1].hard_cut_before is True
    assert clips[1].overlap_before == 20


def test_clip_planner_rejects_truncation_when_chunking_is_disabled(tmp_path):
    planner = DreamIDVClipPlanner(_options(tmp_path, chunking=False))

    with pytest.raises(ValueError, match="no truncar"):
        planner.plan_frames(50, ())


def _sample(path: Path, yaw: float, quality: float) -> SourceReferenceSample:
    face = FaceData(
        bbox=np.array([5, 5, 55, 55], dtype=np.float32),
        kps=np.array([[20, 25], [40, 25], [30, 35], [22, 45], [38, 45]], dtype=np.float32),
        det_score=0.99,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    return SourceReferenceSample(
        path=path,
        image=np.zeros((64, 64, 3), dtype=np.uint8),
        face=face,
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        weight=quality,
        quality=quality,
        yaw=yaw,
        pitch=0.0,
    )


def test_reference_bank_preserves_left_frontal_and_right_views(tmp_path):
    samples = [
        _sample(tmp_path / "left.png", -35.0, 0.75),
        _sample(tmp_path / "front.png", 0.0, 0.80),
        _sample(tmp_path / "right.png", 34.0, 0.78),
        _sample(tmp_path / "front_best.png", 5.0, 0.95),
    ]

    selected = select_diverse_source_references(samples, limit=3)

    assert {sample.path.name for sample in selected} == {
        "left.png",
        "front_best.png",
        "right.png",
    }


def test_target_aware_compositor_preserves_pixels_outside_mask_exactly():
    original = np.full((120, 160, 3), 40, dtype=np.uint8)
    generated = np.full_like(original, 220)
    track = TargetTrackFrame(
        index=0,
        bbox=(55.0, 25.0, 105.0, 95.0),
        kps=((68.0, 50.0), (92.0, 50.0), (80.0, 63.0), (70.0, 78.0), (90.0, 78.0)),
    )
    compositor = TargetAwareCompositor(
        temporal_smoothing=0.0,
        occlusion_strength=0.0,
        color_match=False,
    )

    output, alpha, _ = compositor.compose(original, generated, track)

    outside = alpha == 0.0
    assert np.any(outside)
    assert np.array_equal(output[outside], original[outside])
    assert not np.array_equal(output[50:75, 70:90], original[50:75, 70:90])


class _Capture:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.index = 0

    def isOpened(self):
        return True

    def get(self, prop):
        # CAP_PROP_FRAME_WIDTH=3, HEIGHT=4
        return self.frames[0].shape[1] if int(prop) == 3 else self.frames[0].shape[0]

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        return None


class _SequenceAnalyzer:
    def __init__(self):
        self.index = 0

    @staticmethod
    def _face(embedding, x1):
        return FaceData(
            bbox=np.array([x1, 15, x1 + 30, 55], dtype=np.float32),
            kps=np.array(
                [[x1 + 9, 30], [x1 + 21, 30], [x1 + 15, 39], [x1 + 10, 48], [x1 + 20, 48]],
                dtype=np.float32,
            ),
            det_score=0.99,
            embedding=np.asarray(embedding, dtype=np.float32),
        )

    def analyze(self, frame, previous_bbox, full_scan):
        del frame, previous_bbox, full_scan
        if self.index == 1:
            faces = [self._face([1.0, 0.0], 10), self._face([0.999, 0.035], 55)]
        else:
            faces = [self._face([1.0, 0.0], 10), self._face([0.0, 1.0], 55)]
        self.index += 1
        return faces, SimpleNamespace(detected=2, recognized=2, full_scan=True)


def test_target_tracking_rejects_ambiguous_person_without_switching(monkeypatch):
    frames = [np.full((72, 96, 3), 40 + i, dtype=np.uint8) for i in range(3)]
    monkeypatch.setattr(
        "faceswap_pro.temporal_video.cv2.VideoCapture",
        lambda path: _Capture(frames),
    )
    tracking = SimpleNamespace(
        smoothing=0.0,
        max_missing_frames=2,
        scene_cut_threshold=1.0,
        optical_flow=False,
        flow_win_size=31,
        flow_max_level=3,
        flow_max_error=25.0,
        detection_interval=1,
        full_scan_interval=1,
    )

    track = analyze_target_track(
        Path("proxy.mkv"),
        _SequenceAnalyzer(),
        np.array([1.0, 0.0], dtype=np.float32),
        tracking,
        fps=16,
        min_similarity=0.3,
        ambiguity_margin=0.05,
    )

    assert track.coverage == pytest.approx(2 / 3)
    assert track.ambiguous_ratio == pytest.approx(1 / 3)
    assert track.frames[1].ambiguous is True
    assert track.frames[1].bbox is None
    assert track.frames[2].bbox is not None


def test_target_tracking_accepts_actor_and_mirror_when_multi_target_is_enabled(
    monkeypatch,
):
    frames = [np.full((72, 96, 3), 40 + i, dtype=np.uint8) for i in range(3)]
    monkeypatch.setattr(
        "faceswap_pro.temporal_video.cv2.VideoCapture",
        lambda path: _Capture(frames),
    )
    tracking = SimpleNamespace(
        smoothing=0.0,
        max_missing_frames=2,
        scene_cut_threshold=1.0,
        optical_flow=False,
        flow_win_size=31,
        flow_max_level=3,
        flow_max_error=25.0,
        detection_interval=1,
        full_scan_interval=1,
        max_target_faces=2,
    )

    track = analyze_target_track(
        Path("proxy.mkv"),
        _SequenceAnalyzer(),
        np.array([1.0, 0.0], dtype=np.float32),
        tracking,
        fps=16,
        min_similarity=0.3,
        ambiguity_margin=0.05,
    )

    assert track.coverage == pytest.approx(1.0)
    assert track.ambiguous_ratio == pytest.approx(0.0)
    assert [len(frame.all_instances()) for frame in track.frames] == [1, 2, 1]
    assert track.as_dict()["max_target_faces"] == 2


def test_target_aware_compositor_builds_union_mask_for_mirror_reflection():
    original = np.full((120, 180, 3), 30, dtype=np.uint8)
    generated = np.full_like(original, 220)
    direct = TargetFaceInstance(bbox=(20.0, 25.0, 70.0, 95.0))
    reflection = TargetFaceInstance(bbox=(110.0, 30.0, 155.0, 92.0))
    track = TargetTrackFrame(
        index=0,
        bbox=direct.bbox,
        instances=(direct, reflection),
    )
    compositor = TargetAwareCompositor(
        temporal_smoothing=0.0,
        occlusion_strength=0.0,
        color_match=False,
    )

    output, alpha, _ = compositor.compose(original, generated, track)

    assert alpha[58, 45] > 0.5
    assert alpha[58, 132] > 0.5
    assert alpha[10, 90] == 0.0
    assert not np.array_equal(output[58, 45], original[58, 45])
    assert not np.array_equal(output[58, 132], original[58, 132])
