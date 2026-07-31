from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

FFMPEG_ENV_VAR = "FACESWAP_PRO_FFMPEG"


def _unique_existing_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            candidate = path.expanduser()
        except RuntimeError:
            continue
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        result.append(candidate)
    return result


def ffmpeg_candidates() -> list[Path]:
    """Devuelve ejecutables FFmpeg en orden de preferencia.

    La variable FACESWAP_PRO_FFMPEG tiene prioridad. En Windows se
    prefiere después el alias de WinGet/Gyan porque las builds de Conda suelen
    carecer de NVENC. Finalmente se recorren PATH y shutil.which().
    """

    raw: list[Path] = []

    override = os.environ.get(FFMPEG_ENV_VAR)
    if override:
        raw.append(Path(override))

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            raw.append(Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe")

    resolved = shutil.which("ffmpeg")
    if resolved:
        raw.append(Path(resolved))

    executable_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry:
            raw.append(Path(entry) / executable_name)

    return _unique_existing_paths(raw)


def ffmpeg_has_encoder(ffmpeg: Path | str, encoder: str) -> bool:
    try:
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    combined = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and encoder in combined


def ffmpeg_has_hwaccel(ffmpeg: Path | str, hwaccel: str) -> bool:
    try:
        result = subprocess.run(
            [str(ffmpeg), "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    combined = f"{result.stdout}\n{result.stderr}"
    return result.returncode == 0 and any(
        line.strip() == hwaccel for line in combined.splitlines()
    )


def select_ffmpeg(preferred_encoder: str | None = None) -> Path | None:
    """Selecciona FFmpeg, prefiriendo una build que tenga el encoder pedido."""

    candidates = ffmpeg_candidates()
    if preferred_encoder:
        for candidate in candidates:
            if ffmpeg_has_encoder(candidate, preferred_encoder):
                return candidate
    return candidates[0] if candidates else None
