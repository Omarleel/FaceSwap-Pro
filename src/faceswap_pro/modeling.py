from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class FaceGeometry:
    """Geometría facial densa independiente del framework que la produjo."""

    landmarks: np.ndarray
    pose: np.ndarray | None = None
    transformation: np.ndarray | None = None
    blendshapes: Mapping[str, float] = field(default_factory=dict)
    confidence: float = 0.0

    def clone(self) -> FaceGeometry:
        return FaceGeometry(
            landmarks=np.asarray(self.landmarks, dtype=np.float32).copy(),
            pose=None if self.pose is None else np.asarray(self.pose, dtype=np.float32).copy(),
            transformation=(
                None
                if self.transformation is None
                else np.asarray(self.transformation, dtype=np.float32).copy()
            ),
            blendshapes=dict(self.blendshapes),
            confidence=float(self.confidence),
        )


@dataclass
class FaceData:
    """Representación neutral de un rostro usada por el dominio.

    ``reference_image`` se utiliza únicamente cuando un generador necesita la imagen
    de identidad completa (por ejemplo HifiFace). Los backends basados solo en
    embeddings pueden ignorarla.
    """

    bbox: np.ndarray
    kps: np.ndarray
    det_score: float = 0.0
    embedding: np.ndarray | None = None
    pose: np.ndarray | None = None
    geometry: FaceGeometry | None = None
    reference_image: np.ndarray | None = field(default=None, repr=False, compare=False)
    native: Any | None = field(default=None, repr=False, compare=False)

    def clone(self) -> FaceData:
        return FaceData(
            bbox=np.asarray(self.bbox, dtype=np.float32).copy(),
            kps=np.asarray(self.kps, dtype=np.float32).copy(),
            det_score=float(self.det_score),
            embedding=(
                None
                if self.embedding is None
                else np.asarray(self.embedding, dtype=np.float32).copy()
            ),
            pose=None if self.pose is None else np.asarray(self.pose, dtype=np.float32).copy(),
            geometry=None if self.geometry is None else self.geometry.clone(),
            reference_image=(
                None
                if self.reference_image is None
                else np.asarray(self.reference_image).copy()
            ),
            native=self.native,
        )


@dataclass(frozen=True)
class DetectionStats:
    detected: int
    recognized: int
    full_scan: bool


@dataclass(frozen=True)
class SwapResult:
    crop: np.ndarray
    affine: np.ndarray
    mask: np.ndarray | None = None
    opacity: float = 1.0
    mask_mode: str = "multiply"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelCapabilities:
    """Descripción verificable del modelo, evitando nombres de marketing ambiguos."""

    generator: str = "unknown"
    native_output_size: int | None = None
    geometry_conditioning: str = "none"
    geometry_postprocess: str = "none"
    temporal_generation: str = "frame_independent"

    @property
    def truly_3d_aware(self) -> bool:
        return self.geometry_conditioning in {
            "3dmm_internal",
            "flame_internal",
            "gaussian_3d_internal",
        }


@runtime_checkable
class FaceAnalyzer(Protocol):
    """Interfaz mínima para localizar y reconocer rostros."""

    def find_faces(self, image: np.ndarray) -> list[FaceData]:
        """Devuelve todos los rostros relevantes de una imagen de referencia."""

    def analyze(
        self,
        frame: np.ndarray,
        previous_bbox: np.ndarray | None,
        full_scan: bool,
    ) -> tuple[list[FaceData], DetectionStats]:
        """Analiza un frame del video aplicando la estrategia propia del backend."""


@runtime_checkable
class FaceGeometryEstimator(Protocol):
    """Obtiene malla 3D, pose y expresión sin imponer un framework concreto."""

    def estimate(
        self,
        image: np.ndarray,
        bbox: np.ndarray | None = None,
    ) -> FaceGeometry | None: ...


@runtime_checkable
class FaceSwapper(Protocol):
    """Interfaz independiente del modelo que genera el recorte reemplazado."""

    def swap(
        self,
        frame: np.ndarray,
        target_face: FaceData,
        source_face: FaceData,
    ) -> SwapResult:
        """Genera el rostro sintético y la transformación usada para componerlo."""


@runtime_checkable
class FaceRestorer(Protocol):
    """Interfaz pequeña para restauración opcional de un recorte facial."""

    @property
    def enabled(self) -> bool: ...

    def restore(self, bgr: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelBundle:
    """Servicios de un backend que procesa cada fotograma de forma independiente."""

    backend: str
    analyzer: FaceAnalyzer
    swapper: FaceSwapper
    providers: tuple[Any, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    model_artifacts: tuple[Path, ...] = ()


@dataclass(frozen=True)
class VideoReference:
    """Referencia alineada con metadatos de pose/calidad para selección por plano."""

    path: Path
    yaw: float = 0.0
    pitch: float = 0.0
    quality: float = 1.0


@dataclass(frozen=True)
class VideoSwapRequest:
    """Entradas neutrales para un generador que procesa clips completos."""

    input_video: Path
    source_reference: Path
    output_video: Path
    source_references: tuple[VideoReference, ...] = ()
    target_embedding: np.ndarray | None = field(default=None, repr=False, compare=False)
    source_embedding: np.ndarray | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class VideoSwapResult:
    """Resultado de un backend temporal nativo."""

    output_video: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class VideoSwapBackend(Protocol):
    """Contrato para modelos de vídeo que no deben ejecutarse frame a frame."""

    def process(self, request: VideoSwapRequest) -> VideoSwapResult: ...


@dataclass(frozen=True)
class VideoModelBundle:
    """Servicios de un backend temporal nativo, separado del pipeline por fotogramas."""

    backend: str
    analyzer: FaceAnalyzer
    processor: VideoSwapBackend
    providers: tuple[Any, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    model_artifacts: tuple[Path, ...] = ()


BackendBundle = ModelBundle | VideoModelBundle


class ModelBackendFactory(Protocol):
    def create(self, config: Any, model_path: Path) -> BackendBundle: ...


BackendFactory = ModelBackendFactory | Callable[[Any, Path], BackendBundle]
_BACKENDS: dict[str, BackendFactory] = {}


def register_model_backend(
    name: str,
    factory: BackendFactory,
    *,
    replace: bool = False,
) -> None:
    """Registra un backend sin modificar el pipeline (principio abierto/cerrado)."""

    key = name.strip().lower()
    if not key:
        raise ValueError("El nombre del backend no puede estar vacío.")
    if key in _BACKENDS and not replace:
        raise ValueError(f"El backend ya está registrado: {key}")
    _BACKENDS[key] = factory


def available_model_backends() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))


def create_model_bundle(name: str, config: Any, model_path: Path) -> BackendBundle:
    key = name.strip().lower()
    try:
        factory = _BACKENDS[key]
    except KeyError as exc:
        available = ", ".join(available_model_backends()) or "ninguno"
        raise ValueError(
            f"Backend de modelo desconocido: {name!r}. Disponibles: {available}."
        ) from exc

    creator = getattr(factory, "create", None)
    bundle = creator(config, model_path) if callable(creator) else factory(config, model_path)
    if bundle.backend.strip().lower() != key:
        raise ValueError(
            f"La fábrica {key!r} devolvió un bundle identificado como {bundle.backend!r}."
        )
    return bundle
