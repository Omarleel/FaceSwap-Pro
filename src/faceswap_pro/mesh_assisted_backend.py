from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from .geometry import (
    MediaPipeFaceGeometryEstimator,
    MeshAssistedAnalyzer,
    MeshAssistedSwapper,
    MeshMaskBuilder,
    PiecewiseAffineMeshWarper,
    PoseConfidencePolicy,
)
from .modeling import ModelBundle, ModelCapabilities, create_model_bundle

BACKEND_NAME = "insightface_inswapper_mediapipe_mesh"
LEGACY_BACKEND_NAME = "mediapipe_3d_hybrid"


def _option(options: dict[str, Any], name: str, default: Any) -> Any:
    value = options.get(name, default)
    return default if value is None else value


class MediaPipeMeshAssistedBackendFactory:
    """Añade postproceso por malla a un generador existente.

    Este backend NO es 3D-aware en el sentido generativo: MediaPipe estima una
    malla después de que el generador produjo el rostro. La distinción queda
    reflejada en ``ModelCapabilities.geometry_postprocess``.
    """

    def __init__(self, *, registered_name: str = BACKEND_NAME, warn_legacy: bool = False):
        self._registered_name = registered_name
        self._warn_legacy = warn_legacy

    def create(self, config: Any, model_path: Path) -> ModelBundle:
        if self._warn_legacy:
            warnings.warn(
                "El backend 'mediapipe_3d_hybrid' fue renombrado a "
                "'insightface_inswapper_mediapipe_mesh' porque solo realiza "
                "postproceso por malla. Actualiza tu YAML.",
                DeprecationWarning,
                stacklevel=2,
            )

        options = dict(config.engine.options)
        base_backend = str(_option(options, "base_backend", "insightface_inswapper")).lower()
        if base_backend in {BACKEND_NAME, LEGACY_BACKEND_NAME}:
            raise ValueError(
                "engine.options.base_backend no puede apuntar al backend asistido por malla."
            )

        geometry_model_path = Path(
            str(_option(options, "geometry_model_path", "models/face_landmarker.task"))
        )
        base = create_model_bundle(base_backend, config, model_path)
        estimator = MediaPipeFaceGeometryEstimator(
            geometry_model_path,
            min_detection_confidence=float(
                _option(options, "geometry_min_detection_confidence", 0.50)
            ),
            min_presence_confidence=float(
                _option(options, "geometry_min_presence_confidence", 0.50)
            ),
            min_tracking_confidence=float(
                _option(options, "geometry_min_tracking_confidence", 0.50)
            ),
            crop_padding=float(_option(options, "geometry_crop_padding", 0.12)),
        )

        mesh_warp_enabled = bool(_option(options, "mesh_warp", True))
        warper = (
            PiecewiseAffineMeshWarper(
                sample_step=int(_option(options, "mesh_sample_step", 7))
            )
            if mesh_warp_enabled
            else None
        )
        swapper = MeshAssistedSwapper(
            base.swapper,
            estimator,
            warper=warper,
            mask_builder=MeshMaskBuilder(
                erode_ratio=float(_option(options, "mesh_mask_erode_ratio", 0.012)),
                blur_ratio=float(_option(options, "mesh_mask_blur_ratio", 0.035)),
            ),
            pose_policy=PoseConfidencePolicy(
                pitch_fade_start=float(_option(options, "pitch_fade_start", 28.0)),
                pitch_limit=float(_option(options, "pitch_limit", 58.0)),
                yaw_fade_start=float(_option(options, "yaw_fade_start", 50.0)),
                yaw_limit=float(_option(options, "yaw_limit", 78.0)),
                minimum_opacity=float(_option(options, "minimum_pose_opacity", 0.28)),
            ),
        )
        runtime = dict(base.runtime)
        runtime.update(
            {
                "base_backend": base.backend,
                "geometry_conditioning": base.capabilities.geometry_conditioning,
                "geometry_postprocess": "mediapipe_mesh_warp",
                "mesh_estimator": "mediapipe_face_landmarker",
                "mesh_model": str(geometry_model_path),
                "mesh_warp": mesh_warp_enabled,
                "mesh_sample_step": int(_option(options, "mesh_sample_step", 7)),
            }
        )
        return ModelBundle(
            backend=self._registered_name,
            analyzer=MeshAssistedAnalyzer(base.analyzer, estimator),
            swapper=swapper,
            providers=base.providers,
            runtime=runtime,
            capabilities=ModelCapabilities(
                generator=base.capabilities.generator,
                native_output_size=base.capabilities.native_output_size,
                geometry_conditioning=base.capabilities.geometry_conditioning,
                geometry_postprocess="mediapipe_mesh_warp",
                temporal_generation=base.capabilities.temporal_generation,
            ),
            model_artifacts=tuple((*base.model_artifacts, geometry_model_path)),
        )


class LegacyMediaPipe3DHybridBackendFactory(MediaPipeMeshAssistedBackendFactory):
    def __init__(self) -> None:
        super().__init__(registered_name=LEGACY_BACKEND_NAME, warn_legacy=True)
