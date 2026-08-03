from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

from .modeling import (
    FaceAnalyzer,
    FaceData,
    FaceGeometry,
    FaceGeometryEstimator,
    FaceSwapper,
    SwapResult,
)

# Contorno facial oficial de la topología MediaPipe Face Mesh.
FACE_OVAL_INDICES = np.asarray(
    [
        10,
        338,
        297,
        332,
        284,
        251,
        389,
        356,
        454,
        323,
        361,
        288,
        397,
        365,
        379,
        378,
        400,
        377,
        152,
        148,
        176,
        149,
        150,
        136,
        172,
        58,
        132,
        93,
        234,
        127,
        162,
        21,
        54,
        103,
        67,
        109,
    ],
    dtype=np.int32,
)


def _rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    """Convierte una matriz 3×3 en pitch/yaw/roll en grados.

    MediaPipe entrega la transformación de la cara canónica a la cara detectada.
    Se usa una descomposición XYZ estable para control de confianza, no como medida
    biométrica absoluta.
    """

    r = np.asarray(rotation, dtype=np.float64)[:3, :3]
    sy = math.sqrt(float(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0]))
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(float(r[2, 1]), float(r[2, 2]))
        y = math.atan2(float(-r[2, 0]), sy)
        z = math.atan2(float(r[1, 0]), float(r[0, 0]))
    else:
        x = math.atan2(float(-r[1, 2]), float(r[1, 1]))
        y = math.atan2(float(-r[2, 0]), sy)
        z = 0.0
    # Dominio histórico del proyecto: pose[0]=pitch, pose[1]=yaw, pose[2]=roll.
    return np.degrees(np.asarray([x, y, z], dtype=np.float32))


def _landmark_confidence(landmarks: list[Any]) -> float:
    values: list[float] = []
    for landmark in landmarks:
        for attribute in ("presence", "visibility"):
            value = getattr(landmark, attribute, None)
            if value is not None and float(value) > 0.0:
                values.append(float(value))
    # FaceLandmarker ya filtra por sus umbrales. Algunas versiones no exponen
    # presencia/visibilidad en el objeto Python, por lo que 1.0 es el fallback sano.
    return float(np.clip(np.mean(values), 0.0, 1.0)) if values else 1.0


class MediaPipeFaceGeometryEstimator(FaceGeometryEstimator):
    """Adaptador de MediaPipe Face Landmarker a ``FaceGeometry``.

    La dependencia se importa de forma diferida. El objeto nativo se protege con un
    bloqueo porque el analizador y el swapper viven en hilos diferentes.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        crop_padding: float = 0.12,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(
                f"No existe el modelo de malla MediaPipe: {model_path}. "
                "Descarga face_landmarker.task desde la fuente oficial."
            )
        try:
            import mediapipe as mp
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "El postproceso por malla requiere MediaPipe. Instala el extra: "
                "python -m pip install -e \".[mesh]\""
            ) from exc

        self._mp = mp
        self._crop_padding = max(0.0, float(crop_padding))
        self._lock = threading.Lock()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=float(min_detection_confidence),
            min_face_presence_confidence=float(min_presence_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    @staticmethod
    def _crop_bounds(
        image_shape: tuple[int, ...],
        bbox: np.ndarray | None,
        padding: float,
    ) -> tuple[int, int, int, int]:
        h, w = image_shape[:2]
        if bbox is None:
            return 0, 0, w, h
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
        side = max(float(x2 - x1), float(y2 - y1), 1.0)
        margin = side * padding
        return (
            max(0, int(math.floor(float(x1) - margin))),
            max(0, int(math.floor(float(y1) - margin))),
            min(w, int(math.ceil(float(x2) + margin))),
            min(h, int(math.ceil(float(y2) + margin))),
        )

    def estimate(
        self,
        image: np.ndarray,
        bbox: np.ndarray | None = None,
    ) -> FaceGeometry | None:
        x1, y1, x2, y2 = self._crop_bounds(image.shape, bbox, self._crop_padding)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = np.ascontiguousarray(image[y1:y2, x1:x2])
        if crop.size == 0:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return None

        native_landmarks = result.face_landmarks[0]
        crop_h, crop_w = crop.shape[:2]
        landmarks = np.asarray(
            [
                [
                    x1 + float(point.x) * crop_w,
                    y1 + float(point.y) * crop_h,
                    float(point.z) * crop_w,
                ]
                for point in native_landmarks
            ],
            dtype=np.float32,
        )

        transformation = None
        pose = None
        matrices = getattr(result, "facial_transformation_matrixes", None)
        if matrices:
            transformation = np.asarray(matrices[0], dtype=np.float32)
            pose = _rotation_matrix_to_euler(transformation)

        blendshapes: dict[str, float] = {}
        native_blendshapes = getattr(result, "face_blendshapes", None)
        if native_blendshapes:
            for category in native_blendshapes[0]:
                name = getattr(category, "category_name", None) or getattr(
                    category, "display_name", ""
                )
                if name:
                    blendshapes[str(name)] = float(getattr(category, "score", 0.0))

        return FaceGeometry(
            landmarks=landmarks,
            pose=pose,
            transformation=transformation,
            blendshapes=blendshapes,
            confidence=_landmark_confidence(native_landmarks),
        )


class MeshAssistedAnalyzer(FaceAnalyzer):
    """Añade una malla estimada como metadato; no condiciona al generador."""

    def __init__(self, base: FaceAnalyzer, estimator: FaceGeometryEstimator) -> None:
        self._base = base
        self._estimator = estimator

    @property
    def supports_multiple_previous_bboxes(self) -> bool:
        return bool(
            getattr(self._base, "supports_multiple_previous_bboxes", False)
        )

    def _enrich(self, image: np.ndarray, faces: list[FaceData]) -> list[FaceData]:
        enriched: list[FaceData] = []
        for face in faces:
            result = face.clone()
            geometry = self._estimator.estimate(image, result.bbox)
            if geometry is not None:
                result.geometry = geometry
                if geometry.pose is not None:
                    result.pose = geometry.pose.copy()
            enriched.append(result)
        return enriched

    def find_faces(self, image: np.ndarray) -> list[FaceData]:
        return self._enrich(image, self._base.find_faces(image))

    def analyze(
        self,
        frame: np.ndarray,
        previous_bbox: np.ndarray | None,
        full_scan: bool,
    ):
        faces, stats = self._base.analyze(frame, previous_bbox, full_scan)
        return self._enrich(frame, faces), stats


@dataclass(frozen=True)
class PoseConfidencePolicy:
    pitch_fade_start: float = 28.0
    pitch_limit: float = 58.0
    yaw_fade_start: float = 50.0
    yaw_limit: float = 78.0
    minimum_opacity: float = 0.28

    @staticmethod
    def _axis_confidence(angle: float, start: float, limit: float) -> float:
        angle = abs(float(angle))
        if angle <= start:
            return 1.0
        if angle >= limit:
            return 0.0
        return 1.0 - (angle - start) / max(limit - start, 1e-6)

    def opacity(self, geometry: FaceGeometry | None) -> float:
        if geometry is None:
            return 1.0
        confidence = float(np.clip(geometry.confidence, 0.0, 1.0))
        if geometry.pose is not None and len(geometry.pose) >= 2:
            pitch, yaw = float(geometry.pose[0]), float(geometry.pose[1])
            confidence *= self._axis_confidence(
                pitch, self.pitch_fade_start, self.pitch_limit
            )
            confidence *= self._axis_confidence(yaw, self.yaw_fade_start, self.yaw_limit)
        if confidence <= 0.0:
            return 0.0
        return float(np.clip(max(self.minimum_opacity, confidence), 0.0, 1.0))


class MeshMaskBuilder:
    """Construye una máscara dependiente del contorno real de la malla objetivo."""

    def __init__(self, erode_ratio: float = 0.012, blur_ratio: float = 0.035) -> None:
        self.erode_ratio = max(0.0, float(erode_ratio))
        self.blur_ratio = max(0.0, float(blur_ratio))

    def build(self, geometry: FaceGeometry, size: tuple[int, int]) -> np.ndarray | None:
        height, width = size
        if geometry.landmarks.shape[0] <= int(FACE_OVAL_INDICES.max()):
            return None
        polygon = geometry.landmarks[FACE_OVAL_INDICES, :2]
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        polygon = np.rint(polygon).astype(np.int32)
        mask = np.zeros((height, width), dtype=np.float32)
        cv2.fillPoly(mask, [polygon], 1.0, cv2.LINE_AA)

        erode = int(round(min(width, height) * self.erode_ratio))
        if erode > 0:
            kernel_size = erode * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            mask = cv2.erode(mask, kernel, iterations=1)

        blur = int(round(min(width, height) * self.blur_ratio))
        if blur > 0:
            blur = blur | 1
            mask = cv2.GaussianBlur(mask, (blur, blur), 0)
        return np.clip(mask, 0.0, 1.0)[..., None]


class PiecewiseAffineMeshWarper:
    """Ajusta el rostro generado a la malla objetivo mediante triángulos afines."""

    def __init__(self, sample_step: int = 7) -> None:
        self.sample_step = max(3, int(sample_step))

    def _indices(self, count: int) -> np.ndarray:
        sampled = np.arange(0, min(count, 468), self.sample_step, dtype=np.int32)
        oval = FACE_OVAL_INDICES[FACE_OVAL_INDICES < count]
        return np.unique(np.concatenate([sampled, oval]))

    @staticmethod
    def _warp_triangle(
        source: np.ndarray,
        destination: np.ndarray,
        weight: np.ndarray,
        source_triangle: np.ndarray,
        target_triangle: np.ndarray,
    ) -> None:
        src_rect = cv2.boundingRect(source_triangle.astype(np.float32))
        dst_rect = cv2.boundingRect(target_triangle.astype(np.float32))
        sx, sy, sw, sh = src_rect
        dx, dy, dw, dh = dst_rect
        if sw <= 1 or sh <= 1 or dw <= 1 or dh <= 1:
            return
        src_h, src_w = source.shape[:2]
        dst_h, dst_w = destination.shape[:2]
        if sx < 0 or sy < 0 or sx + sw > src_w or sy + sh > src_h:
            return
        if dx < 0 or dy < 0 or dx + dw > dst_w or dy + dh > dst_h:
            return

        src_local = source_triangle - np.asarray([sx, sy], dtype=np.float32)
        dst_local = target_triangle - np.asarray([dx, dy], dtype=np.float32)
        transform = cv2.getAffineTransform(
            src_local.astype(np.float32),
            dst_local.astype(np.float32),
        )
        patch = source[sy : sy + sh, sx : sx + sw]
        warped = cv2.warpAffine(
            patch,
            transform,
            (dw, dh),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        ).astype(np.float32)
        triangle_mask = np.zeros((dh, dw), dtype=np.float32)
        cv2.fillConvexPoly(
            triangle_mask,
            np.rint(dst_local).astype(np.int32),
            1.0,
            cv2.LINE_AA,
        )
        triangle_mask = triangle_mask[..., None]
        destination[dy : dy + dh, dx : dx + dw] += warped * triangle_mask
        weight[dy : dy + dh, dx : dx + dw] += triangle_mask

    def warp(
        self,
        image: np.ndarray,
        source_geometry: FaceGeometry,
        target_geometry: FaceGeometry,
    ) -> np.ndarray:
        count = min(source_geometry.landmarks.shape[0], target_geometry.landmarks.shape[0])
        indices = self._indices(count)
        if len(indices) < 8:
            return image
        source_points = source_geometry.landmarks[indices, :2].astype(np.float32)
        target_points = target_geometry.landmarks[indices, :2].astype(np.float32)
        height, width = image.shape[:2]
        target_points[:, 0] = np.clip(target_points[:, 0], 0, width - 1)
        target_points[:, 1] = np.clip(target_points[:, 1], 0, height - 1)
        source_points[:, 0] = np.clip(source_points[:, 0], 0, width - 1)
        source_points[:, 1] = np.clip(source_points[:, 1], 0, height - 1)
        try:
            triangles = Delaunay(target_points).simplices
        except QhullError:
            return image

        accumulated = np.zeros_like(image, dtype=np.float32)
        weights = np.zeros((height, width, 1), dtype=np.float32)
        for triangle in triangles:
            self._warp_triangle(
                image,
                accumulated,
                weights,
                source_points[triangle],
                target_points[triangle],
            )
        valid = weights[..., 0] > 1e-5
        if not np.any(valid):
            return image
        result = image.astype(np.float32)
        result[valid] = accumulated[valid] / weights[valid]
        return np.clip(result, 0, 255).astype(np.uint8)


class MeshAssistedSwapper(FaceSwapper):
    """Postprocesa un resultado 2D con malla; no convierte el modelo en 3D-aware."""

    def __init__(
        self,
        base: FaceSwapper,
        estimator: FaceGeometryEstimator,
        *,
        warper: PiecewiseAffineMeshWarper | None,
        mask_builder: MeshMaskBuilder,
        pose_policy: PoseConfidencePolicy,
    ) -> None:
        self._base = base
        self._estimator = estimator
        self._warper = warper
        self._mask_builder = mask_builder
        self._pose_policy = pose_policy

    def swap(
        self,
        frame: np.ndarray,
        target_face: FaceData,
        source_face: FaceData,
    ) -> SwapResult:
        base_result = self._base.swap(frame, target_face, source_face)
        crop = np.ascontiguousarray(base_result.crop)
        affine = np.asarray(base_result.affine, dtype=np.float32)
        crop_h, crop_w = crop.shape[:2]
        aligned_target = cv2.warpAffine(
            frame,
            affine,
            (crop_w, crop_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        target_geometry = self._estimator.estimate(aligned_target)
        source_geometry = self._estimator.estimate(crop)
        mesh_warped = False
        if self._warper is not None and target_geometry is not None and source_geometry is not None:
            crop = self._warper.warp(crop, source_geometry, target_geometry)
            mesh_warped = True

        mask = None
        if target_geometry is not None:
            mask = self._mask_builder.build(target_geometry, (crop_h, crop_w))
        if base_result.mask is not None:
            base_mask = np.asarray(base_result.mask, dtype=np.float32)
            if base_mask.ndim == 2:
                base_mask = base_mask[..., None]
            if base_mask.shape[:2] != (crop_h, crop_w):
                base_mask = cv2.resize(base_mask, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                if base_mask.ndim == 2:
                    base_mask = base_mask[..., None]
            mask = base_mask if mask is None else np.clip(mask * base_mask, 0.0, 1.0)

        opacity = float(base_result.opacity) * self._pose_policy.opacity(target_geometry)
        pose = (
            None
            if target_geometry is None or target_geometry.pose is None
            else target_geometry.pose.tolist()
        )
        metadata = dict(base_result.metadata)
        metadata.update(
            {
                "geometry_provider": "mediapipe_face_landmarker",
                "mesh_warped": mesh_warped,
                "geometry_confidence": (
                    None if target_geometry is None else float(target_geometry.confidence)
                ),
                "pose_degrees": pose,
                "pose_opacity": opacity,
            }
        )
        return SwapResult(
            crop=crop,
            affine=affine,
            mask=mask,
            opacity=float(np.clip(opacity, 0.0, 1.0)),
            metadata=metadata,
        )


# Alias de compatibilidad. Los nombres históricos sugerían condicionamiento 3D
# dentro del generador, cuando en realidad solo había postproceso por malla.
GeometryAwareAnalyzer = MeshAssistedAnalyzer
GeometryAwareSwapper = MeshAssistedSwapper
