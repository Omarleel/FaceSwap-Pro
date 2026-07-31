from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class FaceData:
    """Representación neutral de un rostro usada por el dominio.

    Los adaptadores pueden conservar el objeto del framework en ``native``; el resto
    de la aplicación solo utiliza estos campos y no depende de InsightFace.
    """

    bbox: np.ndarray
    kps: np.ndarray
    det_score: float = 0.0
    embedding: np.ndarray | None = None
    pose: np.ndarray | None = None
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
    """Servicios de modelo construidos por un backend concreto."""

    backend: str
    analyzer: FaceAnalyzer
    swapper: FaceSwapper
    providers: tuple[Any, ...] = ()
    runtime: Mapping[str, Any] = field(default_factory=dict)


class ModelBackendFactory(Protocol):
    def create(self, config: Any, model_path: Path) -> ModelBundle: ...


BackendFactory = ModelBackendFactory | Callable[[Any, Path], ModelBundle]
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


def create_model_bundle(name: str, config: Any, model_path: Path) -> ModelBundle:
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
