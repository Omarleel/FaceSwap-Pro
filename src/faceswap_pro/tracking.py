from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .math_utils import bbox_area, bbox_iou, cosine_similarity
from .modeling import FaceData


@dataclass
class TrackState:
    bbox: np.ndarray
    kps: np.ndarray
    embedding: np.ndarray
    face_template: FaceData
    missing: int = 0


class TemporalFaceTracker:
    def __init__(
        self,
        reference_embedding: np.ndarray,
        min_similarity: float,
        smoothing: float,
        max_missing_frames: int,
        scene_cut_threshold: float,
        optical_flow: bool = True,
        flow_win_size: int = 31,
        flow_max_level: int = 3,
        flow_max_error: float = 25.0,
    ) -> None:
        self.reference_embedding = reference_embedding
        self.min_similarity = min_similarity
        self.smoothing = float(np.clip(smoothing, 0.0, 0.98))
        self.max_missing_frames = max_missing_frames
        self.scene_cut_threshold = scene_cut_threshold
        self.optical_flow = bool(optical_flow)
        self.flow_win_size = max(5, int(flow_win_size) | 1)
        self.flow_max_level = max(0, int(flow_max_level))
        self.flow_max_error = max(0.1, float(flow_max_error))
        self.state: TrackState | None = None
        self.previous_hist: np.ndarray | None = None
        self.previous_gray: np.ndarray | None = None

    @property
    def current_bbox(self) -> np.ndarray | None:
        return None if self.state is None else self.state.bbox

    @property
    def needs_redetect(self) -> bool:
        return self.state is None or self.state.missing > 0

    def reset(self, *, clear_flow: bool = True) -> None:
        self.state = None
        if clear_flow:
            self.previous_gray = None

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def observe(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Calcula una sola imagen gris y detecta cortes antes de analizar el frame."""
        gray = self._gray(frame)
        small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self.previous_hist is not None:
            distance = cv2.compareHist(self.previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            cut = bool(distance >= self.scene_cut_threshold)
        self.previous_hist = hist
        if cut:
            self.reset(clear_flow=True)
        return gray, cut

    def mark_missing(self, gray: np.ndarray) -> None:
        if self.state is not None:
            self.state.missing += 1
            if self.state.missing > self.max_missing_frames:
                self.reset(clear_flow=False)
        self.previous_gray = gray

    def _mark_missing(self, gray: np.ndarray) -> None:
        """Alias privado conservado para extensiones antiguas."""
        self.mark_missing(gray)

    def select_detected(self, frame: np.ndarray, gray: np.ndarray, faces):
        if not faces:
            self.mark_missing(gray)
            return None

        scored = []
        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            identity = cosine_similarity(embedding, self.reference_embedding)
            continuity = bbox_iou(face.bbox, self.state.bbox) if self.state is not None else 0.0
            score = 0.78 * identity + 0.22 * continuity
            scored.append((score, identity, face))

        if not scored:
            self.mark_missing(gray)
            return None

        _, identity, best = max(scored, key=lambda item: item[0])
        if identity < self.min_similarity:
            self.mark_missing(gray)
            return None

        best_bbox = np.asarray(best.bbox, dtype=np.float32)
        best_kps = np.asarray(best.kps, dtype=np.float32)
        if self.state is None:
            smooth_bbox = best_bbox
            smooth_kps = best_kps
        else:
            alpha = self.smoothing
            smooth_bbox = alpha * self.state.bbox + (1.0 - alpha) * best_bbox
            smooth_kps = alpha * self.state.kps + (1.0 - alpha) * best_kps

        result = best.clone()
        result.bbox = smooth_bbox.astype(np.float32)
        result.kps = smooth_kps.astype(np.float32)
        self.state = TrackState(
            bbox=result.bbox.copy(),
            kps=result.kps.copy(),
            embedding=np.asarray(best.embedding, dtype=np.float32).copy(),
            face_template=result.clone(),
            missing=0,
        )
        self.previous_gray = gray
        return result

    @staticmethod
    def _transform_bbox(bbox: np.ndarray, transform: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
        corners = np.array(
            [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]], dtype=np.float32
        )
        moved = cv2.transform(corners, transform)[0]
        return np.array(
            [moved[:, 0].min(), moved[:, 1].min(), moved[:, 0].max(), moved[:, 1].max()],
            dtype=np.float32,
        )

    def _valid_transform(self, transform: np.ndarray, frame_shape) -> bool:
        linear = transform[:, :2]
        scale_x = float(np.linalg.norm(linear[:, 0]))
        scale_y = float(np.linalg.norm(linear[:, 1]))
        if not (0.70 <= scale_x <= 1.40 and 0.70 <= scale_y <= 1.40):
            return False
        h, w = frame_shape[:2]
        translation = float(np.hypot(transform[0, 2], transform[1, 2]))
        return translation <= 0.35 * max(w, h)

    def propagate(self, frame: np.ndarray, gray: np.ndarray):
        """Propaga cinco landmarks con Lucas-Kanade entre detecciones completas."""
        if not self.optical_flow or self.state is None or self.previous_gray is None:
            self.mark_missing(gray)
            return None

        previous_points = self.state.kps.astype(np.float32).reshape(-1, 1, 2)
        next_points, status, errors = cv2.calcOpticalFlowPyrLK(
            self.previous_gray,
            gray,
            previous_points,
            None,
            winSize=(self.flow_win_size, self.flow_win_size),
            maxLevel=self.flow_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
            flags=0,
            minEigThreshold=1e-4,
        )
        if next_points is None or status is None:
            self.mark_missing(gray)
            return None

        valid = status.reshape(-1).astype(bool)
        if errors is not None:
            valid &= errors.reshape(-1) <= self.flow_max_error
        if int(valid.sum()) < 3:
            self.mark_missing(gray)
            return None

        old = previous_points.reshape(-1, 2)[valid]
        new = next_points.reshape(-1, 2)[valid]
        transform, _ = cv2.estimateAffinePartial2D(old, new, method=cv2.LMEDS)
        if transform is None:
            delta = np.median(new - old, axis=0)
            transform = np.array(
                [[1.0, 0.0, float(delta[0])], [0.0, 1.0, float(delta[1])]],
                dtype=np.float32,
            )
        transform = np.asarray(transform, dtype=np.float32)
        if not self._valid_transform(transform, frame.shape):
            self.mark_missing(gray)
            return None

        moved_kps = cv2.transform(self.state.kps[None, ...], transform)[0]
        moved_bbox = self._transform_bbox(self.state.bbox, transform)
        h, w = frame.shape[:2]
        if (
            moved_bbox[2] <= 0
            or moved_bbox[3] <= 0
            or moved_bbox[0] >= w
            or moved_bbox[1] >= h
        ):
            self.mark_missing(gray)
            return None

        self.state.bbox = moved_bbox.astype(np.float32)
        self.state.kps = moved_kps.astype(np.float32)
        self.state.missing = 0
        self.previous_gray = gray

        result = self.state.face_template.clone()
        result.bbox = self.state.bbox.copy()
        result.kps = self.state.kps.copy()
        if result.geometry is not None and result.geometry.landmarks.size:
            moved_dense = cv2.transform(
                result.geometry.landmarks[None, :, :2].astype(np.float32),
                transform,
            )[0]
            result.geometry.landmarks[:, :2] = moved_dense
        return result

    def select(self, frame, faces):
        """API compatible con la versión anterior y con las pruebas existentes."""
        gray, _ = self.observe(frame)
        return self.select_detected(frame, gray, faces)


class MultiFaceTracker:
    """Rastrea varias apariciones simultáneas de la misma identidad.

    Un actor frente a un espejo produce dos detecciones independientes con la misma
    identidad. El rastreador histórico conservaba únicamente la mejor coincidencia,
    por lo que la reflexión quedaba sin procesar. Esta clase mantiene un tracker
    temporal por instancia y realiza una asociación uno-a-uno entre detecciones y
    trayectorias para evitar mezclar/suavizar el rostro real con su reflejo.
    """

    def __init__(
        self,
        reference_embedding: np.ndarray,
        min_similarity: float,
        smoothing: float,
        max_missing_frames: int,
        scene_cut_threshold: float,
        *,
        max_faces: int = 2,
        optical_flow: bool = True,
        flow_win_size: int = 31,
        flow_max_level: int = 3,
        flow_max_error: float = 25.0,
        duplicate_iou_threshold: float = 0.72,
    ) -> None:
        self.reference_embedding = np.asarray(reference_embedding, dtype=np.float32)
        self.min_similarity = float(min_similarity)
        self.smoothing = float(smoothing)
        self.max_missing_frames = int(max_missing_frames)
        self.scene_cut_threshold = float(scene_cut_threshold)
        self.max_faces = max(1, int(max_faces))
        self.optical_flow = bool(optical_flow)
        self.flow_win_size = int(flow_win_size)
        self.flow_max_level = int(flow_max_level)
        self.flow_max_error = float(flow_max_error)
        self.duplicate_iou_threshold = float(
            np.clip(duplicate_iou_threshold, 0.25, 0.95)
        )
        self._tracks: dict[int, TemporalFaceTracker] = {}
        self._next_track_id = 0
        self.previous_hist: np.ndarray | None = None

    @property
    def current_bboxes(self) -> tuple[np.ndarray, ...]:
        active = [
            tracker.current_bbox.copy()
            for tracker in self._tracks.values()
            if tracker.current_bbox is not None
        ]
        return tuple(active)

    @property
    def current_bbox(self) -> np.ndarray | None:
        """Compatibilidad con analizadores antiguos que aceptan una sola ROI."""
        boxes = self.current_bboxes
        if not boxes:
            return None
        return max(boxes, key=bbox_area).copy()

    @property
    def needs_redetect(self) -> bool:
        return not self._tracks or any(
            tracker.needs_redetect for tracker in self._tracks.values()
        )

    def _new_tracker(self) -> TemporalFaceTracker:
        return TemporalFaceTracker(
            self.reference_embedding,
            self.min_similarity,
            self.smoothing,
            self.max_missing_frames,
            self.scene_cut_threshold,
            optical_flow=self.optical_flow,
            flow_win_size=self.flow_win_size,
            flow_max_level=self.flow_max_level,
            flow_max_error=self.flow_max_error,
        )

    def reset(self, *, clear_flow: bool = True) -> None:
        for tracker in self._tracks.values():
            tracker.reset(clear_flow=clear_flow)
        self._tracks.clear()

    @staticmethod
    def _gray(frame: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def observe(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        gray = self._gray(frame)
        small = cv2.resize(gray, (96, 54), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([small], [0], None, [32], [0, 256])
        cv2.normalize(hist, hist)
        cut = False
        if self.previous_hist is not None:
            distance = cv2.compareHist(
                self.previous_hist,
                hist,
                cv2.HISTCMP_BHATTACHARYYA,
            )
            cut = bool(distance >= self.scene_cut_threshold)
        self.previous_hist = hist
        if cut:
            self.reset(clear_flow=True)
        return gray, cut

    @staticmethod
    def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = np.asarray(a, dtype=np.float32)
        bx1, by1, bx2, by2 = np.asarray(b, dtype=np.float32)
        return float(
            np.hypot(
                (ax1 + ax2 - bx1 - bx2) * 0.5,
                (ay1 + ay2 - by1 - by2) * 0.5,
            )
        )

    @staticmethod
    def _bbox_diagonal(bbox: np.ndarray) -> float:
        x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
        return max(1.0, float(np.hypot(x2 - x1, y2 - y1)))

    def _deduplicate(self, faces) -> list[tuple[float, FaceData]]:
        scored: list[tuple[float, FaceData]] = []
        for face in faces:
            embedding = getattr(face, "embedding", None)
            if embedding is None:
                continue
            similarity = cosine_similarity(embedding, self.reference_embedding)
            if similarity < self.min_similarity:
                continue
            scored.append((float(similarity), face))

        scored.sort(
            key=lambda item: (
                item[0],
                float(getattr(item[1], "det_score", 0.0) or 0.0),
                bbox_area(item[1].bbox),
            ),
            reverse=True,
        )
        unique: list[tuple[float, FaceData]] = []
        for similarity, face in scored:
            if any(
                bbox_iou(face.bbox, kept.bbox) >= self.duplicate_iou_threshold
                for _, kept in unique
            ):
                continue
            unique.append((similarity, face))
        return unique

    def _association_score(
        self,
        tracker: TemporalFaceTracker,
        face: FaceData,
        similarity: float,
    ) -> float | None:
        previous = tracker.current_bbox
        if previous is None:
            return None
        iou = bbox_iou(previous, face.bbox)
        distance = self._center_distance(previous, face.bbox)
        scale = max(self._bbox_diagonal(previous), self._bbox_diagonal(face.bbox))
        normalized_distance = distance / scale
        # La compuerta espacial impide que dos instancias lejanas (actor/reflejo)
        # se promedien entre sí cuando cambia el orden de las detecciones.
        if iou <= 0.0 and normalized_distance > 2.4:
            return None
        proximity = max(0.0, 1.0 - normalized_distance / 2.4)
        continuity = 0.72 * iou + 0.28 * proximity
        return 0.55 * similarity + 0.45 * continuity

    def _prune_expired(self) -> None:
        expired = [
            track_id
            for track_id, tracker in self._tracks.items()
            if tracker.state is None
        ]
        for track_id in expired:
            del self._tracks[track_id]

    @staticmethod
    def _composition_order(faces: list[FaceData]) -> list[FaceData]:
        # Primero las caras pequeñas/lejana; las grandes/frontal quedan por encima
        # si dos máscaras se solapan parcialmente.
        return sorted(faces, key=lambda face: bbox_area(face.bbox))

    def select_all_detected(
        self,
        frame: np.ndarray,
        gray: np.ndarray,
        faces,
    ) -> list[FaceData]:
        candidates = self._deduplicate(faces)
        if not candidates:
            for tracker in self._tracks.values():
                tracker.mark_missing(gray)
            self._prune_expired()
            return []

        track_ids = list(self._tracks)
        pairs: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            tracker = self._tracks[track_id]
            for candidate_index, (similarity, face) in enumerate(candidates):
                score = self._association_score(tracker, face, similarity)
                if score is not None:
                    pairs.append((score, track_id, candidate_index))
        pairs.sort(reverse=True)

        assigned_tracks: set[int] = set()
        assigned_candidates: set[int] = set()
        selected: list[FaceData] = []
        for _, track_id, candidate_index in pairs:
            if track_id in assigned_tracks or candidate_index in assigned_candidates:
                continue
            tracker = self._tracks[track_id]
            result = tracker.select_detected(
                frame,
                gray,
                [candidates[candidate_index][1]],
            )
            if result is None:
                continue
            assigned_tracks.add(track_id)
            assigned_candidates.add(candidate_index)
            selected.append(result)

        for track_id, tracker in list(self._tracks.items()):
            if track_id not in assigned_tracks:
                tracker.mark_missing(gray)

        self._prune_expired()

        for candidate_index, (_, face) in enumerate(candidates):
            if candidate_index in assigned_candidates:
                continue
            if len(self._tracks) >= self.max_faces:
                break
            tracker = self._new_tracker()
            result = tracker.select_detected(frame, gray, [face])
            if result is None:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = tracker
            selected.append(result)

        return self._composition_order(selected)

    def propagate_all(self, frame: np.ndarray, gray: np.ndarray) -> list[FaceData]:
        propagated: list[FaceData] = []
        for tracker in self._tracks.values():
            result = tracker.propagate(frame, gray)
            if result is not None:
                propagated.append(result)
        self._prune_expired()
        return self._composition_order(propagated)

    def select_detected(self, frame: np.ndarray, gray: np.ndarray, faces):
        """Devuelve el rostro principal para integraciones que aún esperan uno."""
        selected = self.select_all_detected(frame, gray, faces)
        return max(selected, key=lambda face: bbox_area(face.bbox), default=None)

    def propagate(self, frame: np.ndarray, gray: np.ndarray):
        propagated = self.propagate_all(frame, gray)
        return max(propagated, key=lambda face: bbox_area(face.bbox), default=None)

    def select(self, frame: np.ndarray, faces):
        gray, _ = self.observe(frame)
        return self.select_all_detected(frame, gray, faces)
