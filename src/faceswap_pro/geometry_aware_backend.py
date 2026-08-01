"""Compatibilidad con el backend histórico ``mediapipe_3d_hybrid``.

El nombre sugería condicionamiento 3D dentro del generador, cuando en realidad solo
había postproceso por malla. Código nuevo debe importar ``mesh_assisted_backend``.
"""

from .mesh_assisted_backend import (
    BACKEND_NAME as CANONICAL_BACKEND_NAME,
    LEGACY_BACKEND_NAME,
    LegacyMediaPipe3DHybridBackendFactory,
    MediaPipeMeshAssistedBackendFactory,
)

# Compatibilidad para plugins que importaban estas dos variables desde este módulo.
BACKEND_NAME = LEGACY_BACKEND_NAME
MediaPipe3DHybridBackendFactory = LegacyMediaPipe3DHybridBackendFactory

__all__ = [
    "BACKEND_NAME",
    "CANONICAL_BACKEND_NAME",
    "LEGACY_BACKEND_NAME",
    "LegacyMediaPipe3DHybridBackendFactory",
    "MediaPipe3DHybridBackendFactory",
    "MediaPipeMeshAssistedBackendFactory",
]
