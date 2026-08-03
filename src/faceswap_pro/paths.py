from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT_INPUT_VIDEO = Path("inputs/videos/input.mp4")
DEFAULT_SOURCE_DIR = Path("inputs/source_faces")
DEFAULT_TARGET_REFERENCE = Path("inputs/target_faces/target.jpg")
DEFAULT_SWAPPER_MODEL = Path("models/inswapper_128.onnx")
DEFAULT_CONFIG = Path("config/max_speed.yaml")
DEFAULT_OUTPUT_DIR = Path("outputs/videos")
DEFAULT_MANIFEST_DIR = Path("outputs/manifests")

PROJECT_DIRECTORIES = (
    Path("inputs/videos"),
    Path("inputs/source_faces"),
    Path("inputs/target_faces"),
    Path("models"),
    Path("models/hififace/standard_model"),
    Path("models/hififace/aux"),
    Path("models/dreamidv"),
    Path("third_party"),
    DEFAULT_OUTPUT_DIR,
    DEFAULT_MANIFEST_DIR,
)


def create_project_directories(root: Path = Path.cwd()) -> list[Path]:
    """Crea la estructura estándar de FaceSwap-Pro y devuelve las rutas creadas."""
    created: list[Path] = []
    for relative in PROJECT_DIRECTORIES:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        created.append(directory)
    return created


def build_output_path(
    input_video: Path,
    output_video: Path | None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    now: datetime | None = None,
) -> Path:
    """Resuelve una salida explícita o genera un nombre fechado sin sobrescribir."""
    if output_video is not None:
        resolved = output_video
        if not resolved.suffix:
            resolved = resolved.with_suffix(".mp4")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = f"{input_video.stem}_faceswap_{stamp}"
    candidate = output_dir / f"{base}.mp4"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{base}_{counter}.mp4"
        counter += 1
    return candidate


def build_manifest_path(output_video: Path, manifest_dir: Path) -> Path:
    manifest_dir.mkdir(parents=True, exist_ok=True)
    return manifest_dir / f"{output_video.stem}.manifest.json"
