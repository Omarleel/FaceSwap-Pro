from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
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


def _path_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        stat = path.stat()
        digest.update(f"file:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def sha256_path_cached(path: Path, cache_file: Path) -> str:
    """Evita releer checkpoints si tamaño/mtime y estructura no cambiaron."""

    path = path.expanduser().resolve()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        cache = {}
    fingerprint = _path_fingerprint(path)
    key = os.path.normcase(str(path))
    record = cache.get(key)
    if isinstance(record, dict) and record.get("fingerprint") == fingerprint:
        cached_hash = record.get("sha256")
        if isinstance(cached_hash, str) and len(cached_hash) == 64:
            return cached_hash
    digest = sha256_path(path)
    cache[key] = {
        "fingerprint": fingerprint,
        "sha256": digest,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = cache_file.with_suffix(cache_file.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    temporary.replace(cache_file)
    return digest


def model_record(
    path: Path,
    *,
    hash_content: bool = True,
    hash_cache_file: Path | None = None,
) -> dict[str, str]:
    record = {
        "path": str(path),
        "kind": "directory" if path.is_dir() else "file",
    }
    if not hash_content:
        record["sha256"] = "skipped"
    elif hash_cache_file is not None:
        record["sha256"] = sha256_path_cached(path, hash_cache_file)
        record["hash_cache"] = str(hash_cache_file)
    else:
        record["sha256"] = sha256_path(path)
    return record


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


def _resolve_tool(value: str) -> str | None:
    candidate = Path(value).expanduser()
    if candidate.is_file():
        return str(candidate.resolve())
    return shutil.which(value)


def embed_c2pa_manifest(
    *,
    output_video: Path,
    input_video: Path,
    runtime: dict[str, Any],
    manifest_dir: Path,
    config: Any,
) -> dict[str, Any]:
    """Firma e incrusta Content Credentials con c2patool cuando está habilitado."""

    enabled = bool(getattr(config, "c2pa_enabled", False))
    required = bool(getattr(config, "c2pa_required", False))
    if not enabled:
        return {"enabled": False, "status": "disabled"}
    tool = _resolve_tool(str(getattr(config, "c2pa_tool", "c2patool")))
    if tool is None:
        message = "No se encontró c2patool."
        if required:
            raise RuntimeError(message)
        return {"enabled": True, "status": "skipped", "error": message}

    definition = {
        "alg": str(getattr(config, "c2pa_algorithm", "es256")),
        "claim_generator": f"FaceSwap-Pro/{__version__}",
        "title": output_video.name,
        "ingredient_paths": [str(input_video.resolve())],
        "assertions": [
            {
                "label": "c2pa.actions.v2",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.created",
                            "digitalSourceType": (
                                "http://cv.iptc.org/newscodes/digitalsourcetype/"
                                "compositeWithTrainedAlgorithmicMedia"
                            ),
                            "softwareAgent": {
                                "name": "FaceSwap-Pro",
                                "version": __version__,
                            },
                        }
                    ],
                    "allActionsIncluded": True,
                },
            },
            {
                "label": "org.faceswap-pro.provenance",
                "data": {
                    "synthetic_media": True,
                    "operation": "authorized_face_replacement",
                    "model_backend": runtime.get("model_backend"),
                    "model_capabilities": runtime.get("model_capabilities"),
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                },
            },
            {
                "label": "stds.schema-org.CreativeWork",
                "data": {
                    "@context": "https://schema.org",
                    "@type": "VideoObject",
                    "name": output_video.name,
                    "description": "Deep fake generado mediante reemplazo facial autorizado.",
                },
            },
        ],
    }
    cert = getattr(config, "c2pa_sign_cert", None)
    key = getattr(config, "c2pa_private_key", None)
    if cert and key:
        definition["sign_cert"] = str(Path(cert).expanduser().resolve())
        definition["private_key"] = str(Path(key).expanduser().resolve())
    timestamp = getattr(config, "c2pa_timestamp_url", None)
    if timestamp:
        definition["ta_url"] = str(timestamp)

    manifest_dir.mkdir(parents=True, exist_ok=True)
    definition_path = manifest_dir / f"{output_video.stem}.c2pa-definition.json"
    definition_path.write_text(
        json.dumps(definition, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with tempfile.TemporaryDirectory(prefix="faceswap_c2pa_", dir=output_video.parent) as temp:
        signed = Path(temp) / output_video.name
        completed = subprocess.run(
            [tool, str(output_video), "-m", str(definition_path), "-o", str(signed)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not signed.is_file():
            message = completed.stderr.strip() or completed.stdout.strip() or "Firma C2PA fallida."
            if required:
                raise RuntimeError(message)
            return {
                "enabled": True,
                "status": "error",
                "tool": tool,
                "definition": str(definition_path),
                "error": message,
            }
        signed.replace(output_video)
    return {
        "enabled": True,
        "status": "embedded",
        "tool": tool,
        "definition": str(definition_path),
        "certificate": str(cert) if cert else "c2patool-development-certificate",
    }


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
    visible_disclosure: bool = True,
    hash_models: bool = True,
    c2pa_status: dict[str, Any] | None = None,
    hash_cache_dir: Path | None = None,
) -> Path:
    manifest_path = build_manifest_path(output_video, manifest_dir)
    cache_file = (
        (hash_cache_dir or manifest_dir / ".hash-cache") / "model-sha256.json"
        if hash_models
        else None
    )
    payload = {
        "schema": "faceswap-pro-manifest-v2",
        "project": {"name": "FaceSwap-Pro", "version": __version__},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure": {
            "visible": bool(visible_disclosure),
            "text": DISCLOSURE_TEXT if visible_disclosure else None,
        },
        "c2pa": c2pa_status or {"enabled": False, "status": "not_requested"},
        "output_video": {
            "path": str(output_video),
            "sha256": sha256_file(output_video) if output_video.is_file() else None,
        },
        "input_video": {"path": str(input_video), "sha256": sha256_file(input_video)},
        "target_reference": {
            "path": str(target_reference),
            "sha256": sha256_file(target_reference),
        },
        "source_images": [
            {"path": str(path), "sha256": sha256_file(path)} for path in source_images
        ],
        "model": model_record(
            model_path,
            hash_content=hash_models,
            hash_cache_file=cache_file,
        ),
        "additional_models": [
            model_record(
                path,
                hash_content=hash_models,
                hash_cache_file=cache_file,
            )
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
