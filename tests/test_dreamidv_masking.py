from __future__ import annotations

import cv2
import numpy as np

from faceswap_pro.dreamidv_masking import refine_target_mask_frame
from faceswap_pro.temporal_video import TargetFaceInstance, TargetTrackFrame


def _instance(bbox):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return TargetFaceInstance(
        bbox=tuple(float(value) for value in bbox),
        kps=(
            (x1 + 0.32 * width, y1 + 0.38 * height),
            (x1 + 0.68 * width, y1 + 0.38 * height),
            (x1 + 0.50 * width, y1 + 0.54 * height),
            (x1 + 0.38 * width, y1 + 0.72 * height),
            (x1 + 0.62 * width, y1 + 0.72 * height),
        ),
        similarity=0.9,
    )


def test_target_aware_mask_preserves_dwpose_primary_and_adds_mirror():
    raw = np.zeros((100, 160, 3), dtype=np.uint8)
    cv2.ellipse(raw, (38, 52), (20, 29), 0, 0, 360, (255, 255, 255), -1)
    direct = _instance((16, 18, 60, 86))
    reflection = _instance((104, 24, 142, 84))
    track = TargetTrackFrame(
        index=0,
        bbox=direct.bbox,
        instances=(direct, reflection),
    )

    refined, counters = refine_target_mask_frame(raw, track)
    gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)

    assert gray[52, 38] > 200
    assert gray[54, 123] > 200
    assert gray[8, 80] == 0
    assert counters["kept_raw_components"] == 1
    assert counters["synthetic_instances_added"] == 1


def test_target_aware_mask_drops_unrelated_larger_face_component():
    raw = np.zeros((120, 180, 3), dtype=np.uint8)
    cv2.ellipse(raw, (90, 62), (34, 42), 0, 0, 360, (255, 255, 255), -1)
    target = _instance((15, 25, 55, 90))
    track = TargetTrackFrame(index=0, bbox=target.bbox, instances=(target,))

    refined, counters = refine_target_mask_frame(raw, track)
    gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)

    assert gray[58, 35] > 200
    assert gray[62, 90] == 0
    assert counters["raw_components"] == 1
    assert counters["kept_raw_components"] == 0
    assert counters["synthetic_instances_added"] == 1


def test_target_aware_mask_supports_multiple_reflections():
    raw = np.zeros((120, 220, 3), dtype=np.uint8)
    instances = (
        _instance((12, 28, 52, 98)),
        _instance((88, 30, 126, 94)),
        _instance((164, 26, 204, 96)),
    )
    track = TargetTrackFrame(index=0, bbox=instances[0].bbox, instances=instances)

    refined, counters = refine_target_mask_frame(raw, track)
    gray = cv2.cvtColor(refined, cv2.COLOR_BGR2GRAY)

    assert gray[60, 32] > 200
    assert gray[60, 107] > 200
    assert gray[60, 184] > 200
    assert counters["target_instances"] == 3
    assert counters["synthetic_instances_added"] == 3


def test_target_aware_mask_is_empty_when_target_is_ambiguous():
    raw = np.full((64, 96, 3), 255, dtype=np.uint8)
    target = _instance((20, 12, 62, 58))
    track = TargetTrackFrame(
        index=0,
        bbox=target.bbox,
        ambiguous=True,
        instances=(target,),
    )

    refined, counters = refine_target_mask_frame(raw, track)

    assert not np.any(refined)
    assert counters["target_instances"] == 0
