from __future__ import annotations

"""Máscaras de condicionamiento DreamID-V conscientes de identidad.

DWPose upstream conserva únicamente el hull facial de mayor área. Eso funciona para
un único rostro, pero omite reflejos o varias apariciones simultáneas del mismo
actor. Este módulo cruza la máscara DWPose con el tracking de identidad y añade una
máscara sintética solo para las instancias objetivo que DWPose no cubrió.
"""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .temporal_video import TargetFaceInstance, TargetTrack, TargetTrackFrame


@dataclass
class TargetAwareMaskStats:
    frames_total: int = 0
    frames_with_target: int = 0
    frames_with_multiple_targets: int = 0
    frames_without_target: int = 0
    target_instances: int = 0
    max_target_instances: int = 0
    raw_components: int = 0
    kept_raw_components: int = 0
    dropped_raw_components: int = 0
    synthetic_instances_added: int = 0
    source_mask_shortfall: int = 0

    def as_dict(self) -> dict[str, int | float]:
        return {
            "frames_total": self.frames_total,
            "frames_with_target": self.frames_with_target,
            "frames_with_multiple_targets": self.frames_with_multiple_targets,
            "frames_without_target": self.frames_without_target,
            "target_instances": self.target_instances,
            "max_target_instances": self.max_target_instances,
            "average_target_instances": (
                float(self.target_instances / self.frames_with_target)
                if self.frames_with_target
                else 0.0
            ),
            "raw_components": self.raw_components,
            "kept_raw_components": self.kept_raw_components,
            "dropped_raw_components": self.dropped_raw_components,
            "synthetic_instances_added": self.synthetic_instances_added,
            "source_mask_shortfall": self.source_mask_shortfall,
        }


def _scaled_instance(
    instance: TargetFaceInstance,
    *,
    scale_x: float,
    scale_y: float,
) -> TargetFaceInstance:
    x1, y1, x2, y2 = instance.bbox
    return TargetFaceInstance(
        bbox=(x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
        kps=tuple((x * scale_x, y * scale_y) for x, y in instance.kps),
        yaw=instance.yaw,
        pitch=instance.pitch,
        similarity=instance.similarity,
    )


def _instance_conditioning_mask(
    shape: tuple[int, int],
    instance: TargetFaceInstance,
    *,
    expand: float,
) -> np.ndarray:
    """Construye una elipse facial estable a partir de bbox y cinco landmarks."""

    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = (float(value) for value in instance.bbox)
    box_width = max(2.0, x2 - x1)
    box_height = max(2.0, y2 - y1)
    center_x = 0.5 * (x1 + x2)
    center_y = y1 + 0.52 * box_height
    angle = 0.0
    if len(instance.kps) >= 2:
        left_eye = np.asarray(instance.kps[0], dtype=np.float32)
        right_eye = np.asarray(instance.kps[1], dtype=np.float32)
        delta = right_eye - left_eye
        if float(np.linalg.norm(delta)) > 1e-3:
            angle = float(np.degrees(np.arctan2(delta[1], delta[0])))

    axes = (
        max(2, int(round(0.51 * box_width * max(0.75, expand)))),
        max(2, int(round(0.58 * box_height * max(0.75, expand)))),
    )
    cv2.ellipse(
        mask,
        (int(round(center_x)), int(round(center_y))),
        axes,
        angle,
        0.0,
        360.0,
        255,
        -1,
        cv2.LINE_AA,
    )
    return mask


def _component_masks(binary: np.ndarray) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    minimum_area = max(12, int(binary.size * 0.00002))
    result: list[np.ndarray] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        result.append(labels == label)
    return result


def refine_target_mask_frame(
    raw_mask: np.ndarray,
    track_frame: TargetTrackFrame | None,
    *,
    track_size: tuple[int, int] | None = None,
    expand: float = 1.08,
    min_raw_coverage: float = 0.18,
) -> tuple[np.ndarray, dict[str, int]]:
    """Retiene componentes DWPose del actor y añade reflejos no cubiertos.

    La máscara primaria de DWPose se conserva cuando coincide con una instancia
    objetivo. Solo se usa la elipse sintética cuando esa instancia no alcanza la
    cobertura mínima, de modo que la calidad actual del rostro principal no cambia.
    """

    if raw_mask.ndim == 3:
        raw_gray = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
    else:
        raw_gray = np.asarray(raw_mask, dtype=np.uint8)
    height, width = raw_gray.shape[:2]
    empty = np.zeros((height, width), dtype=np.uint8)
    counters = {
        "target_instances": 0,
        "raw_components": 0,
        "kept_raw_components": 0,
        "synthetic_instances_added": 0,
    }
    if track_frame is None or track_frame.ambiguous:
        return cv2.cvtColor(empty, cv2.COLOR_GRAY2BGR), counters

    instances = list(track_frame.all_instances())
    if not instances:
        return cv2.cvtColor(empty, cv2.COLOR_GRAY2BGR), counters

    if track_size is None:
        track_width, track_height = width, height
    else:
        track_width, track_height = track_size
    scale_x = width / max(1.0, float(track_width))
    scale_y = height / max(1.0, float(track_height))
    scaled = [
        _scaled_instance(instance, scale_x=scale_x, scale_y=scale_y)
        for instance in instances
    ]
    instance_masks = [
        _instance_conditioning_mask((height, width), instance, expand=expand)
        for instance in scaled
    ]
    counters["target_instances"] = len(instance_masks)

    binary = (raw_gray >= 24).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    components = _component_masks(binary)
    counters["raw_components"] = len(components)

    selected_raw = np.zeros_like(raw_gray)
    covered = np.zeros(len(instance_masks), dtype=np.float32)
    for component in components:
        area = max(1, int(component.sum()))
        matches: list[int] = []
        for index, instance_mask in enumerate(instance_masks):
            instance_binary = instance_mask > 16
            overlap = int(np.count_nonzero(component & instance_binary))
            instance_area = max(1, int(np.count_nonzero(instance_binary)))
            if overlap / area >= 0.05 or overlap / instance_area >= 0.08:
                matches.append(index)
                covered[index] = max(covered[index], overlap / instance_area)
        if not matches:
            continue
        selected_raw[component] = np.maximum(selected_raw[component], raw_gray[component])
        counters["kept_raw_components"] += 1

    result = selected_raw
    threshold = float(np.clip(min_raw_coverage, 0.02, 0.95))
    for index, instance_mask in enumerate(instance_masks):
        if covered[index] >= threshold:
            continue
        result = np.maximum(result, instance_mask)
        counters["synthetic_instances_added"] += 1

    # El codec de la máscara trabaja mejor con bordes compactos y sin huecos de
    # compresión. No se aplica feather: DreamID-V upstream recibe una máscara dura.
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR), counters


def build_target_aware_mask_video(
    *,
    ffmpeg: Path,
    source_mask: Path,
    destination: Path,
    track: TargetTrack,
    requested_frames: int,
    fps: float,
    start_frame: int = 0,
    valid_frames: int | None = None,
    expand: float = 1.08,
    min_raw_coverage: float = 0.18,
) -> TargetAwareMaskStats:
    """Genera un vídeo lossless de máscara condicionado por identidad."""

    capture = cv2.VideoCapture(str(source_mask))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir la máscara DWPose: {source_mask}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("La máscara DWPose no reporta una resolución válida.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(ffmpeg),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{float(fps):.8f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "0",
        "-pix_fmt",
        "yuv444p",
        str(destination),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    stats = TargetAwareMaskStats()
    error: BaseException | None = None
    try:
        if process.stdin is None:
            raise RuntimeError("FFmpeg no abrió stdin para la máscara objetivo.")
        for index in range(max(0, int(requested_frames))):
            ok, raw_frame = capture.read()
            if not ok or raw_frame is None:
                stats.source_mask_shortfall += max(0, requested_frames - index)
                raise RuntimeError(
                    "La máscara DWPose terminó antes que el vídeo objetivo "
                    f"(frame {index}/{requested_frames})."
                )
            if valid_frames is not None and valid_frames > 0:
                relative_index = min(index, valid_frames - 1)
            else:
                relative_index = index
            track_frame = track.frame(start_frame + relative_index)
            refined, counters = refine_target_mask_frame(
                raw_frame,
                track_frame,
                track_size=(track.width, track.height),
                expand=expand,
                min_raw_coverage=min_raw_coverage,
            )
            instances = 0 if track_frame is None else len(track_frame.all_instances())
            stats.frames_total += 1
            stats.target_instances += instances
            stats.max_target_instances = max(stats.max_target_instances, instances)
            if instances:
                stats.frames_with_target += 1
            else:
                stats.frames_without_target += 1
            if instances > 1:
                stats.frames_with_multiple_targets += 1
            stats.raw_components += counters["raw_components"]
            stats.kept_raw_components += counters["kept_raw_components"]
            stats.synthetic_instances_added += counters["synthetic_instances_added"]
            process.stdin.write(np.ascontiguousarray(refined).tobytes())
    except BaseException as exc:  # noqa: BLE001 - se relanza después de cerrar FFmpeg
        error = exc
    finally:
        capture.release()
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
            process.stderr.close()
        return_code = process.wait()

    stats.dropped_raw_components = max(
        0, stats.raw_components - stats.kept_raw_components
    )
    if error is not None:
        destination.unlink(missing_ok=True)
        raise error
    if return_code != 0:
        destination.unlink(missing_ok=True)
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"FFmpeg no pudo escribir la máscara objetivo: {message}")
    if not destination.is_file() or destination.stat().st_size <= 0:
        raise RuntimeError("No se creó la máscara objetivo para DreamID-V.")
    return stats
