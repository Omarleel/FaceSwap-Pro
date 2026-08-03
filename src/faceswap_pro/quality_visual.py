from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    scale = min(width / max(1, w), height / max(1, h))
    resized = cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4,
    )
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _read_fraction(capture: cv2.VideoCapture, fraction: float) -> np.ndarray | None:
    count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(np.clip(fraction, 0.0, 1.0) * (count - 1))))
    ok, frame = capture.read()
    return frame if ok else None


def write_visual_comparison_sheet(
    *,
    input_video: Path,
    output_video: Path,
    destination: Path,
    source_reference: Path | None = None,
    target_reference: Path | None = None,
    samples: int = 6,
) -> dict[str, Any]:
    """Crea una hoja visual reproducible de entrada/salida en puntos normalizados."""

    source_capture = cv2.VideoCapture(str(input_video))
    output_capture = cv2.VideoCapture(str(output_video))
    if not source_capture.isOpened() or not output_capture.isOpened():
        source_capture.release()
        output_capture.release()
        return {
            "status": "unavailable",
            "path": None,
            "error": "No se pudieron abrir ambos vídeos para la comparación visual.",
        }

    tile_w, tile_h = 480, 270
    rows: list[np.ndarray] = []
    try:
        if source_reference is not None or target_reference is not None:
            source_image = (
                cv2.imread(str(source_reference), cv2.IMREAD_COLOR)
                if source_reference is not None
                else None
            )
            target_image = (
                cv2.imread(str(target_reference), cv2.IMREAD_COLOR)
                if target_reference is not None
                else None
            )
            identity_row = np.hstack(
                [
                    _label(_fit(source_image, tile_w, tile_h), "Identidad de origen"),
                    _label(_fit(target_image, tile_w, tile_h), "Referencia del sujeto objetivo"),
                ]
            )
            rows.append(identity_row)

        fractions = np.linspace(0.05, 0.95, max(1, int(samples)))
        used = 0
        for fraction in fractions:
            before = _read_fraction(source_capture, float(fraction))
            after = _read_fraction(output_capture, float(fraction))
            if before is None or after is None:
                continue
            timestamp = f"{fraction * 100:.0f}% de la línea temporal"
            row = np.hstack(
                [
                    _label(_fit(before, tile_w, tile_h), f"Entrada · {timestamp}"),
                    _label(_fit(after, tile_w, tile_h), f"Salida · {timestamp}"),
                ]
            )
            rows.append(row)
            used += 1
    finally:
        source_capture.release()
        output_capture.release()

    if not rows:
        return {
            "status": "unavailable",
            "path": None,
            "error": "No se pudieron extraer fotogramas de comparación.",
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        return {
            "status": "error",
            "path": None,
            "error": f"No se pudo escribir la hoja visual: {destination}",
        }
    return {
        "status": "created",
        "path": str(destination),
        "sample_count": used,
        "resolution": [int(sheet.shape[1]), int(sheet.shape[0])],
    }
