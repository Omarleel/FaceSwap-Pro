from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

# Plantilla ArcFace de cinco puntos para 112×112.
_ARCFACE_112 = np.asarray(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# Plantilla usada por el pipeline de inferencia de xuehy/HiFiFace-pytorch.
_HIFIFACE_112 = np.asarray(
    [
        [39.730, 51.138],
        [72.270, 51.138],
        [56.000, 68.493],
        [42.463, 87.010],
        [69.537, 87.010],
    ],
    dtype=np.float32,
)


def _scaled_template(template: np.ndarray, size: int) -> np.ndarray:
    if size < 32:
        raise ValueError("El tamaño alineado debe ser al menos 32 píxeles.")
    return template * (float(size) / 112.0)


def arcface_template(size: int) -> np.ndarray:
    return _scaled_template(_ARCFACE_112, size)


def hififace_template(size: int) -> np.ndarray:
    return _scaled_template(_HIFIFACE_112, size)


_TEMPLATES: dict[str, Callable[[int], np.ndarray]] = {
    "arcface": arcface_template,
    "hififace": hififace_template,
}


def estimate_face_affine(
    kps: np.ndarray,
    size: int,
    *,
    template: str = "arcface",
) -> np.ndarray:
    points = np.asarray(kps, dtype=np.float32)
    if points.shape != (5, 2):
        raise ValueError(f"Se esperaban 5 landmarks 2D; se recibió {points.shape}.")
    try:
        destination = _TEMPLATES[template](size)
    except KeyError as exc:
        raise ValueError(f"Plantilla facial desconocida: {template!r}.") from exc
    matrix, _ = cv2.estimateAffinePartial2D(
        points,
        destination,
        method=cv2.LMEDS,
    )
    if matrix is None or not np.isfinite(matrix).all():
        raise RuntimeError("No se pudo estimar la transformación facial de cinco puntos.")
    return np.asarray(matrix, dtype=np.float32)


def align_face(
    image: np.ndarray,
    kps: np.ndarray,
    size: int,
    *,
    template: str = "arcface",
    interpolation: int = cv2.INTER_CUBIC,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = estimate_face_affine(kps, size, template=template)
    crop = cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return np.ascontiguousarray(crop), matrix
