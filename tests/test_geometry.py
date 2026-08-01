from __future__ import annotations

import numpy as np

from faceswap_pro.geometry import (
    GeometryAwareSwapper,
    MeshMaskBuilder,
    PiecewiseAffineMeshWarper,
    PoseConfidencePolicy,
)
from faceswap_pro.modeling import FaceData, FaceGeometry, SwapResult


def _circular_geometry(size: int = 64, *, pitch: float = 0.0) -> FaceGeometry:
    center = size / 2.0
    radius = size * 0.32
    angles = np.linspace(0.0, 2.0 * np.pi, 478, endpoint=False)
    landmarks = np.column_stack(
        [
            center + np.cos(angles) * radius,
            center + np.sin(angles) * radius,
            np.zeros_like(angles),
        ]
    ).astype(np.float32)
    return FaceGeometry(
        landmarks=landmarks,
        pose=np.asarray([pitch, 0.0, 0.0], dtype=np.float32),
        confidence=0.9,
    )


def test_face_geometry_clone_detaches_arrays():
    geometry = _circular_geometry()
    clone = geometry.clone()
    clone.landmarks[0, 0] = 999.0
    clone.pose[0] = 45.0

    assert geometry.landmarks[0, 0] != 999.0
    assert geometry.pose[0] == 0.0


def test_mesh_mask_is_spatially_limited():
    mask = MeshMaskBuilder(erode_ratio=0.0, blur_ratio=0.0).build(
        _circular_geometry(),
        (64, 64),
    )

    assert mask is not None
    assert mask.shape == (64, 64, 1)
    assert mask[32, 32, 0] > 0.9
    assert mask[0, 0, 0] == 0.0


def test_piecewise_affine_identity_warp_preserves_image():
    image = np.arange(64 * 64 * 3, dtype=np.uint16).reshape(64, 64, 3)
    image = (image % 256).astype(np.uint8)
    geometry = _circular_geometry()

    warped = PiecewiseAffineMeshWarper(sample_step=12).warp(image, geometry, geometry)

    difference = np.abs(warped.astype(np.int16) - image.astype(np.int16))
    assert float(difference.mean()) < 1.0


class _BaseSwapper:
    def swap(self, frame, target_face, source_face):
        crop = np.full((64, 64, 3), 180, dtype=np.uint8)
        affine = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return SwapResult(crop=crop, affine=affine)


class _Estimator:
    def __init__(self, geometry: FaceGeometry):
        self.geometry = geometry

    def estimate(self, image, bbox=None):
        return self.geometry.clone()


def test_geometry_aware_swapper_returns_adaptive_mask_and_pose_opacity():
    geometry = _circular_geometry(pitch=45.0)
    swapper = GeometryAwareSwapper(
        _BaseSwapper(),
        _Estimator(geometry),
        warper=None,
        mask_builder=MeshMaskBuilder(erode_ratio=0.0, blur_ratio=0.0),
        pose_policy=PoseConfidencePolicy(
            pitch_fade_start=30.0,
            pitch_limit=60.0,
            minimum_opacity=0.25,
        ),
    )
    face = FaceData(
        bbox=np.asarray([0, 0, 64, 64], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
    )

    result = swapper.swap(np.zeros((64, 64, 3), dtype=np.uint8), face, face)

    assert result.mask is not None
    assert result.mask.shape == (64, 64, 1)
    assert 0.25 <= result.opacity < 1.0
    assert result.metadata["geometry_provider"] == "mediapipe_face_landmarker"
