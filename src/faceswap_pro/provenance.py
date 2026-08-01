from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import __version__
from .paths import build_manifest_path


DISCLOSURE_TEXT = "CONTENIDO SINTÉTICO · IA"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash reproducible de un archivo o de un árbol de modelos."""

    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        file_hash = bytes.fromhex(sha256_file(child))
        digest.update(file_hash)
    return digest.hexdigest()


def model_record(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
        "sha256": sha256_path(path),
    }


def add_disclosure(frame, text: str = DISCLOSURE_TEXT):
    """Añade una etiqueta visible de contenido sintético."""
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(w, h) / 1500.0)
    thickness = max(1, round(scale * 2))
    margin = max(12, round(min(w, h) * 0.018))
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = margin, margin
    x2, y2 = x1 + tw + margin, y1 + th + baseline + margin

    x2 = min(w, x2)
    y2 = min(h, y2)
    label_roi = frame[y1:y2, x1:x2]
    if label_roi.size:
        overlay = np.zeros_like(label_roi)
        cv2.addWeighted(overlay, 0.58, label_roi, 0.42, 0, label_roi)
    cv2.putText(
        frame,
        text,
        (x1 + margin // 2, max(th + baseline, y2 - baseline - margin // 2)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return frame


def write_manifest(
    *,
    output_video: Path,
    manifest_dir: Path,
    input_video: Path,
    source_images: list[Path],
    target_reference: Path,
    model_path: Path,
    runtime: dict[str, Any],
    additional_model_paths: list[Path] | None = None,
) -> Path:
    manifest_path = build_manifest_path(output_video, manifest_dir)
    payload = {
        "schema": "faceswap-pro-manifest-v1",
        "project": {"name": "FaceSwap-Pro", "version": __version__},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure": DISCLOSURE_TEXT,
        "output_video": {"path": str(output_video)},
        "input_video": {"path": str(input_video), "sha256": sha256_file(input_video)},
        "target_reference": {
            "path": str(target_reference),
            "sha256": sha256_file(target_reference),
        },
        "source_images": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_images
        ],
        "model": model_record(model_path),
        "additional_models": [
            model_record(path)
            for path in (additional_model_paths or [])
            if path.exists()
        ],
        "runtime": runtime,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest_path
