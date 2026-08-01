from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from rich.console import Console

from .math_utils import bbox_area, l2_normalize
from .modeling import FaceAnalyzer, FaceData

console = Console()
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def _pose_weight(face: FaceData) -> float:
    pose = face.pose
    if pose is None or len(pose) < 2:
        return 1.0
    pitch, yaw = float(pose[0]), float(pose[1])
    return max(0.20, float(np.exp(-(abs(pitch) + abs(yaw)) / 70.0)))


def build_source_identity(
    analyzer: FaceAnalyzer,
    source_dir: Path,
    min_score: float,
    limit: int,
) -> tuple[FaceData, list[Path]]:
    paths = list_source_images(source_dir, limit)
    selected_faces: list[FaceData] = []
    selected_images: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    weights: list[float] = []
    valid_paths: list[Path] = []

    for path in paths:
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
        weight = max(
            1e-4,
            float(face.det_score) * np.sqrt(area_ratio) * _pose_weight(face),
        )
        selected_faces.append(face)
        selected_images.append(np.ascontiguousarray(image))
        embeddings.append(embedding)
        weights.append(weight)
        valid_paths.append(path)
        console.print(f"[green]✓[/green] {path.name}  peso={weight:.3f}")

    if not embeddings:
        raise ValueError("Ninguna foto de origen produjo un rostro utilizable.")

    matrix = np.stack(embeddings, axis=0)
    mean_embedding = l2_normalize(
        np.average(
            matrix,
            axis=0,
            weights=np.asarray(weights, dtype=np.float32),
        )
    )

    best_index = int(np.argmax(weights))
    source_face = selected_faces[best_index].clone()
    source_face.embedding = mean_embedding.astype(np.float32)
    source_face.reference_image = selected_images[best_index].copy()

    return source_face, valid_paths


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
