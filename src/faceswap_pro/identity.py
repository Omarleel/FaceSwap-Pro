from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from .math_utils import bbox_area, l2_normalize
from .modeling import FaceAnalyzer, FaceData

console = Console()
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class SourceReferenceSample:
    path: Path
    image: np.ndarray
    face: FaceData
    embedding: np.ndarray
    weight: float
    quality: float
    yaw: float
    pitch: float


def list_source_images(directory: Path, limit: int) -> list[Path]:
    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No se encontraron imágenes en {directory}")
    return images[:limit]


def _best_face(faces: list[FaceData], image_shape) -> FaceData:
    h, w = image_shape[:2]
    cx, cy = w / 2.0, h / 2.0

    def score(face: FaceData) -> float:
        x1, y1, x2, y2 = face.bbox
        fx, fy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        center_penalty = np.hypot((fx - cx) / max(w, 1), (fy - cy) / max(h, 1))
        return (
            bbox_area(face.bbox)
            * float(face.det_score or 0.0)
            * (1.0 - min(center_penalty, 0.8))
        )

    return max(faces, key=score)


def estimate_pose_from_five_points(kps: np.ndarray) -> tuple[float, float]:
    """Estimación estable y ligera de yaw/pitch para elegir referencias.

    No sustituye un estimador 3D; solo ordena vistas izquierda/frontal/derecha.
    """

    points = np.asarray(kps, dtype=np.float32)
    if points.shape[0] < 5:
        return 0.0, 0.0
    left_eye, right_eye, nose, left_mouth, right_mouth = points[:5]
    eye_mid = (left_eye + right_eye) * 0.5
    mouth_mid = (left_mouth + right_mouth) * 0.5
    eye_distance = max(1.0, float(np.linalg.norm(right_eye - left_eye)))
    face_height = max(1.0, float(np.linalg.norm(mouth_mid - eye_mid)))
    yaw = float(np.clip((nose[0] - eye_mid[0]) / eye_distance * 55.0, -60.0, 60.0))
    expected_nose_y = eye_mid[1] + 0.52 * face_height
    pitch = float(np.clip((nose[1] - expected_nose_y) / face_height * 55.0, -40.0, 40.0))
    return yaw, pitch


def _pose_weight(face: FaceData) -> float:
    if face.pose is not None and len(face.pose) >= 2:
        pitch, yaw = float(face.pose[0]), float(face.pose[1])
    else:
        yaw, pitch = estimate_pose_from_five_points(face.kps)
    return max(0.20, float(np.exp(-(abs(pitch) + abs(yaw)) / 70.0)))


def _image_quality(image: np.ndarray, face: FaceData) -> float:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float32)
    margin_x = 0.08 * max(1.0, x2 - x1)
    margin_y = 0.08 * max(1.0, y2 - y1)
    ix1 = max(0, int(np.floor(x1 - margin_x)))
    iy1 = max(0, int(np.floor(y1 - margin_y)))
    ix2 = min(w, int(np.ceil(x2 + margin_x)))
    iy2 = min(h, int(np.ceil(y2 + margin_y)))
    crop = image[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    sharpness_score = float(np.clip(np.log1p(sharpness) / np.log(1001.0), 0.0, 1.0))
    mean = float(gray.mean())
    exposure_score = float(np.exp(-((mean - 128.0) / 95.0) ** 2))
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    clipping_score = 1.0 - min(1.0, clipped * 4.0)
    return float(np.clip(0.55 * sharpness_score + 0.30 * exposure_score + 0.15 * clipping_score, 0.0, 1.0))


def collect_source_references(
    analyzer: FaceAnalyzer,
    source_dir: Path,
    min_score: float,
    limit: int,
) -> list[SourceReferenceSample]:
    samples: list[SourceReferenceSample] = []
    for path in list_source_images(source_dir, limit):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            console.print(f"[yellow]Omitida (no se pudo leer):[/yellow] {path}")
            continue
        faces = [
            face
            for face in analyzer.find_faces(image)
            if float(face.det_score or 0.0) >= min_score
        ]
        if not faces:
            console.print(f"[yellow]Sin rostro fiable:[/yellow] {path}")
            continue
        face = _best_face(faces, image.shape)
        if face.embedding is None:
            console.print(f"[yellow]Sin embedding facial:[/yellow] {path}")
            continue
        embedding = l2_normalize(face.embedding)
        area_ratio = bbox_area(face.bbox) / max(1.0, image.shape[0] * image.shape[1])
        quality = _image_quality(image, face)
        weight = max(
            1e-4,
            float(face.det_score)
            * np.sqrt(area_ratio)
            * _pose_weight(face)
            * (0.45 + 0.55 * quality),
        )
        if face.pose is not None and len(face.pose) >= 2:
            pitch, yaw = float(face.pose[0]), float(face.pose[1])
        else:
            yaw, pitch = estimate_pose_from_five_points(face.kps)
        samples.append(
            SourceReferenceSample(
                path=path,
                image=np.ascontiguousarray(image),
                face=face,
                embedding=embedding,
                weight=weight,
                quality=quality,
                yaw=yaw,
                pitch=pitch,
            )
        )
        console.print(
            f"[green]✓[/green] {path.name}  peso={weight:.3f} "
            f"calidad={quality:.2f} pose=({yaw:+.1f},{pitch:+.1f})"
        )
    if not samples:
        raise ValueError("Ninguna foto de origen produjo un rostro utilizable.")
    return samples


def _identity_from_samples(samples: list[SourceReferenceSample]) -> FaceData:
    matrix = np.stack([sample.embedding for sample in samples], axis=0)
    weights = np.asarray([sample.weight for sample in samples], dtype=np.float32)
    mean_embedding = l2_normalize(np.average(matrix, axis=0, weights=weights))
    best = max(samples, key=lambda sample: sample.weight)
    source_face = best.face.clone()
    source_face.embedding = mean_embedding.astype(np.float32)
    source_face.reference_image = best.image.copy()
    return source_face


def select_diverse_source_references(
    samples: list[SourceReferenceSample],
    limit: int = 6,
) -> list[SourceReferenceSample]:
    """Selecciona frontal y vistas laterales antes de completar por calidad."""

    if limit <= 0:
        return []
    ordered = sorted(samples, key=lambda sample: (sample.quality, sample.weight), reverse=True)
    bins = [(-90.0, -18.0), (-18.0, 18.0), (18.0, 90.0)]
    selected: list[SourceReferenceSample] = []
    for low, high in bins:
        candidates = [sample for sample in ordered if low <= sample.yaw < high]
        if candidates:
            selected.append(candidates[0])
    selected_paths = {sample.path for sample in selected}
    for sample in ordered:
        if sample.path not in selected_paths:
            selected.append(sample)
            selected_paths.add(sample.path)
        if len(selected) >= limit:
            break
    return selected[:limit]


def build_source_identity(
    analyzer: FaceAnalyzer,
    source_dir: Path,
    min_score: float,
    limit: int,
) -> tuple[FaceData, list[Path]]:
    samples = collect_source_references(analyzer, source_dir, min_score, limit)
    return _identity_from_samples(samples), [sample.path for sample in samples]


def build_source_identity_and_bank(
    analyzer: FaceAnalyzer,
    source_dir: Path,
    min_score: float,
    limit: int,
    bank_size: int = 6,
) -> tuple[FaceData, list[SourceReferenceSample], list[Path]]:
    samples = collect_source_references(analyzer, source_dir, min_score, limit)
    return (
        _identity_from_samples(samples),
        select_diverse_source_references(samples, bank_size),
        [sample.path for sample in samples],
    )


def load_reference_embedding(
    analyzer: FaceAnalyzer,
    path: Path,
    min_score: float,
) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"No se pudo leer la referencia del sujeto objetivo: {path}")
    faces = [
        face
        for face in analyzer.find_faces(image)
        if float(face.det_score or 0.0) >= min_score
    ]
    if not faces:
        raise ValueError("No se detectó un rostro fiable en la referencia del sujeto objetivo.")
    face = _best_face(faces, image.shape)
    if face.embedding is None:
        raise ValueError("La referencia objetivo no produjo embedding de reconocimiento.")
    return l2_normalize(face.embedding)
