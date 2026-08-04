from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

from .identity import estimate_pose_from_five_points
from .math_utils import bbox_area, bbox_iou, cosine_similarity
from .tracking import MultiFaceTracker


@dataclass(frozen=True)
class TargetFaceInstance:
    bbox: tuple[float, float, float, float]
    kps: tuple[tuple[float, float], ...] = ()
    yaw: float = 0.0
    pitch: float = 0.0
    similarity: float = 0.0


@dataclass(frozen=True)
class TargetTrackFrame:
    index: int
    bbox: tuple[float, float, float, float] | None
    kps: tuple[tuple[float, float], ...] = ()
    yaw: float = 0.0
    pitch: float = 0.0
    similarity: float = 0.0
    ambiguous: bool = False
    scene_cut: bool = False
    instances: tuple[TargetFaceInstance, ...] = ()

    def all_instances(self) -> tuple[TargetFaceInstance, ...]:
        if self.instances:
            return self.instances
        if self.bbox is None:
            return ()
        return (
            TargetFaceInstance(
                bbox=self.bbox,
                kps=self.kps,
                yaw=self.yaw,
                pitch=self.pitch,
                similarity=self.similarity,
            ),
        )


@dataclass(frozen=True)
class TargetTrack:
    fps: float
    width: int
    height: int
    frames: tuple[TargetTrackFrame, ...]
    scene_cuts: tuple[int, ...]
    coverage: float
    ambiguous_ratio: float

    def frame(self, index: int) -> TargetTrackFrame | None:
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None

    def pose_for_range(self, start: int, end: int) -> tuple[float, float]:
        subset = [
            frame
            for frame in self.frames[max(0, start) : min(len(self.frames), end)]
            if frame.bbox is not None and not frame.ambiguous
        ]
        if not subset:
            return 0.0, 0.0
        return (
            float(np.median([frame.yaw for frame in subset])),
            float(np.median([frame.pitch for frame in subset])),
        )

    def as_dict(self) -> dict[str, Any]:
        instance_counts = [len(frame.all_instances()) for frame in self.frames]
        return {
            "fps": self.fps,
            "resolution": [self.width, self.height],
            "frames": len(self.frames),
            "scene_cuts": list(self.scene_cuts),
            "coverage": self.coverage,
            "ambiguous_ratio": self.ambiguous_ratio,
            "average_target_faces": float(np.mean(instance_counts)) if instance_counts else 0.0,
            "max_target_faces": max(instance_counts, default=0),
            "frames_with_multiple_target_faces": sum(
                1 for count in instance_counts if count > 1
            ),
        }


def analyze_target_track(
    video_path,
    analyzer,
    target_embedding: np.ndarray,
    tracking,
    *,
    fps: float,
    min_similarity: float = 0.30,
    ambiguity_margin: float = 0.05,
) -> TargetTrack:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"No se pudo abrir el proxy para tracking: {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_target_faces = max(1, int(getattr(tracking, "max_target_faces", 1)))
    tracker = MultiFaceTracker(
        target_embedding,
        min_similarity,
        tracking.smoothing,
        tracking.max_missing_frames,
        tracking.scene_cut_threshold,
        max_faces=max_target_faces,
        optical_flow=tracking.optical_flow,
        flow_win_size=tracking.flow_win_size,
        flow_max_level=tracking.flow_max_level,
        flow_max_error=tracking.flow_max_error,
    )

    result: list[TargetTrackFrame] = []
    scene_cuts: list[int] = []
    ambiguous_count = 0
    selected_count = 0
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            gray, scene_cut = tracker.observe(frame)
            if scene_cut:
                scene_cuts.append(index)
            detect = (
                tracker.needs_redetect
                or scene_cut
                or index % max(1, int(tracking.detection_interval)) == 0
            )
            selected: list = []
            similarity = 0.0
            ambiguous = False
            if detect:
                full_scan = (
                    not tracker.current_bboxes
                    or scene_cut
                    or index % max(1, int(tracking.full_scan_interval)) == 0
                )
                previous_regions = tracker.current_bbox
                if bool(
                    getattr(analyzer, "supports_multiple_previous_bboxes", False)
                ) and tracker.current_bboxes:
                    previous_regions = tracker.current_bboxes
                faces, _ = analyzer.analyze(frame, previous_regions, full_scan)
                scored = sorted(
                    (
                        (cosine_similarity(face.embedding, target_embedding), face)
                        for face in faces
                        if getattr(face, "embedding", None) is not None
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )
                if scored:
                    similarity = float(scored[0][0])
                    if max_target_faces == 1 and len(scored) > 1:
                        second = float(scored[1][0])
                        ambiguous = (
                            similarity >= tracker.min_similarity
                            and second >= tracker.min_similarity
                            and similarity - second < ambiguity_margin
                        )
                    if not ambiguous:
                        selected = tracker.select_all_detected(
                            frame,
                            gray,
                            [item[1] for item in scored],
                        )
                    else:
                        tracker.select_all_detected(frame, gray, [])
                else:
                    tracker.select_all_detected(frame, gray, [])
            else:
                selected = tracker.propagate_all(frame, gray)
                if tracker.needs_redetect:
                    previous_regions = tracker.current_bbox
                    if bool(
                        getattr(analyzer, "supports_multiple_previous_bboxes", False)
                    ) and tracker.current_bboxes:
                        previous_regions = tracker.current_bboxes
                    faces, _ = analyzer.analyze(frame, previous_regions, True)
                    selected = tracker.select_all_detected(frame, gray, faces)
                similarities = [
                    cosine_similarity(face.embedding, target_embedding)
                    for face in selected
                    if getattr(face, "embedding", None) is not None
                ]
                if similarities:
                    similarity = float(max(similarities))

            if ambiguous:
                ambiguous_count += 1
            if not selected:
                result.append(
                    TargetTrackFrame(
                        index=index,
                        bbox=None,
                        similarity=similarity,
                        ambiguous=ambiguous,
                        scene_cut=scene_cut,
                    )
                )
            else:
                selected_count += 1
                instances = []
                for face in selected:
                    yaw, pitch = estimate_pose_from_five_points(face.kps)
                    face_similarity = (
                        float(cosine_similarity(face.embedding, target_embedding))
                        if getattr(face, "embedding", None) is not None
                        else 0.0
                    )
                    instances.append(
                        TargetFaceInstance(
                            bbox=tuple(float(value) for value in face.bbox),
                            kps=tuple(
                                tuple(float(v) for v in point) for point in face.kps
                            ),
                            yaw=yaw,
                            pitch=pitch,
                            similarity=face_similarity,
                        )
                    )
                primary = max(
                    instances,
                    key=lambda item: bbox_area(np.asarray(item.bbox, dtype=np.float32)),
                )
                result.append(
                    TargetTrackFrame(
                        index=index,
                        bbox=primary.bbox,
                        kps=primary.kps,
                        yaw=primary.yaw,
                        pitch=primary.pitch,
                        similarity=max(item.similarity for item in instances),
                        ambiguous=False,
                        scene_cut=scene_cut,
                        instances=tuple(instances),
                    )
                )
            index += 1
    finally:
        capture.release()

    total = max(1, len(result))
    return TargetTrack(
        fps=float(fps),
        width=width,
        height=height,
        frames=tuple(result),
        scene_cuts=tuple(scene_cuts),
        coverage=float(selected_count / total),
        ambiguous_ratio=float(ambiguous_count / total),
    )


def _flow_warp_mask(previous_gray: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    scale = min(1.0, 320.0 / max(h, w))
    size = (max(32, int(round(w * scale))), max(32, int(round(h * scale))))
    prev_small = cv2.resize(previous_gray, size, interpolation=cv2.INTER_AREA)
    gray_small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        prev_small,
        gray_small,
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.1,
        0,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(size[0], dtype=np.float32),
        np.arange(size[1], dtype=np.float32),
    )
    # Flujo prev->actual. Para llevar la máscara previa al actual se usa el mapa inverso aproximado.
    warped_small = cv2.remap(
        cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR),
        grid_x - flow[..., 0],
        grid_y - flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return cv2.resize(warped_small, (w, h), interpolation=cv2.INTER_LINEAR)


def _face_mask(
    shape: tuple[int, int],
    track: TargetTrackFrame | TargetFaceInstance,
    expand: float,
) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    if track.bbox is None:
        return mask
    x1, y1, x2, y2 = track.bbox
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    center = (int(round((x1 + x2) * 0.5)), int(round(y1 + 0.53 * bh)))
    axes = (
        max(2, int(round(0.50 * bw * expand))),
        max(2, int(round(0.56 * bh * expand))),
    )
    cv2.ellipse(mask, center, axes, 0.0, 0.0, 360.0, 1.0, -1, cv2.LINE_AA)
    return mask


def _adaptive_occlusion_mask(
    original: np.ndarray,
    generated: np.ndarray,
    face_mask: np.ndarray,
) -> np.ndarray:
    if not np.any(face_mask > 0.05):
        return np.zeros_like(face_mask)
    ycrcb = cv2.cvtColor(original, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    core = face_mask > 0.75
    if int(core.sum()) < 64:
        return np.zeros_like(face_mask)
    chroma = ycrcb[..., 1:3]
    median = np.median(chroma[core], axis=0)
    distance = np.linalg.norm(chroma - median, axis=2)
    non_skin = distance > 34.0

    gray_o = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_g = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
    edge_o = cv2.Canny(gray_o, 55, 135)
    edge_g = cv2.Canny(gray_g, 55, 135)
    edge_residual = (edge_o > 0) & (cv2.dilate(edge_g, np.ones((3, 3), np.uint8)) == 0)
    candidate = (non_skin & edge_residual & (face_mask > 0.08)).astype(np.uint8)
    candidate = cv2.dilate(candidate, np.ones((5, 5), np.uint8), iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros_like(candidate)
    min_area = max(12, int(core.sum() * 0.0015))
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == label] = 1
    return cv2.GaussianBlur(keep.astype(np.float32), (0, 0), sigmaX=2.0)


def _color_match(generated: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    region = mask > 0.35
    if int(region.sum()) < 128:
        return generated
    gen_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
    org_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    g = gen_lab[region]
    o = org_lab[region]
    g_mean, g_std = g.mean(axis=0), np.maximum(g.std(axis=0), 3.0)
    o_mean, o_std = o.mean(axis=0), np.maximum(o.std(axis=0), 3.0)
    matched = (gen_lab - g_mean) * np.clip(o_std / g_std, 0.72, 1.38) + o_mean
    return cv2.cvtColor(np.clip(matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


class TargetAwareCompositor:
    def __init__(
        self,
        *,
        mask_expand: float = 1.08,
        feather_ratio: float = 0.055,
        temporal_smoothing: float = 0.62,
        occlusion_strength: float = 0.92,
        color_match: bool = True,
    ) -> None:
        self.mask_expand = float(max(0.7, mask_expand))
        self.feather_ratio = float(max(0.005, feather_ratio))
        self.temporal_smoothing = float(np.clip(temporal_smoothing, 0.0, 0.95))
        self.occlusion_strength = float(np.clip(occlusion_strength, 0.0, 1.0))
        self.color_match = bool(color_match)
        self.previous_gray: np.ndarray | None = None
        self.previous_mask: np.ndarray | None = None

    def compose(
        self,
        original: np.ndarray,
        generated: np.ndarray,
        track: TargetTrackFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if generated.shape[:2] != original.shape[:2]:
            generated = cv2.resize(
                generated,
                (original.shape[1], original.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        if track is not None and track.scene_cut:
            self.previous_gray = None
            self.previous_mask = None
        if track is None or track.bbox is None or track.ambiguous:
            self.previous_gray = gray
            self.previous_mask = np.zeros(original.shape[:2], dtype=np.float32)
            return original.copy(), self.previous_mask.copy(), self.previous_mask.copy()

        current = np.zeros(original.shape[:2], dtype=np.float32)
        for instance in track.all_instances():
            current = np.maximum(
                current,
                _face_mask(original.shape[:2], instance, self.mask_expand),
            )
        if self.previous_gray is not None and self.previous_mask is not None:
            warped = _flow_warp_mask(self.previous_gray, gray, self.previous_mask)
            current = (
                (1.0 - self.temporal_smoothing) * current
                + self.temporal_smoothing * warped
            )
        current = np.clip(current, 0.0, 1.0)
        sigma = max(1.0, min(original.shape[:2]) * self.feather_ratio)
        current = cv2.GaussianBlur(current, (0, 0), sigmaX=sigma)
        current = np.clip(current / max(1e-6, float(current.max())), 0.0, 1.0)

        corrected = _color_match(generated, original, current) if self.color_match else generated
        occlusion = _adaptive_occlusion_mask(original, corrected, current)
        alpha = np.clip(current * (1.0 - self.occlusion_strength * occlusion), 0.0, 1.0)
        alpha[alpha < 0.01] = 0.0
        output = (
            corrected.astype(np.float32) * alpha[..., None]
            + original.astype(np.float32) * (1.0 - alpha[..., None])
        )
        self.previous_gray = gray
        self.previous_mask = current
        return np.clip(output, 0, 255).astype(np.uint8), alpha, occlusion


def align_generated_to_source(generated: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Alinea una hipótesis generada al movimiento/estructura del frame fuente."""

    if generated.shape[:2] != source.shape[:2]:
        source = cv2.resize(source, (generated.shape[1], generated.shape[0]), cv2.INTER_AREA)
    h, w = generated.shape[:2]
    scale = min(1.0, 384.0 / max(h, w))
    size = (max(64, int(round(w * scale))), max(64, int(round(h * scale))))
    src_small = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
    gen_small = cv2.resize(generated, size, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(src_small, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(gen_small, cv2.COLOR_BGR2GRAY),
        None,
        0.5,
        3,
        21,
        3,
        5,
        1.1,
        0,
    )
    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
    flow[..., 0] *= w / size[0]
    flow[..., 1] *= h / size[1]
    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    return cv2.remap(
        generated,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT101,
    )


@dataclass
class QualityAccumulator:
    analyzer: Any | None = None
    source_embedding: np.ndarray | None = None
    sample_interval: int = 8
    identity_scores: list[float] = field(default_factory=list)
    pose_errors: list[float] = field(default_factory=list)
    outside_mae: list[float] = field(default_factory=list)
    temporal_errors: list[float] = field(default_factory=list)
    boundary_ratios: list[float] = field(default_factory=list)
    _previous_face: np.ndarray | None = field(default=None, repr=False)
    _recent_deltas: deque[float] = field(default_factory=lambda: deque(maxlen=20), repr=False)

    def observe(
        self,
        index: int,
        original: np.ndarray,
        output: np.ndarray,
        alpha: np.ndarray,
        track: TargetTrackFrame | None,
        *,
        chunk_boundary: bool = False,
    ) -> None:
        outside = alpha < 0.01
        if np.any(outside):
            self.outside_mae.append(
                float(np.mean(np.abs(output[outside].astype(np.float32) - original[outside])))
            )
        if track is None or track.bbox is None:
            self._previous_face = None
            return
        x1, y1, x2, y2 = [int(round(v)) for v in track.bbox]
        h, w = output.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        face = cv2.resize(output[y1:y2, x1:x2], (128, 128), interpolation=cv2.INTER_AREA)
        if self._previous_face is not None:
            delta = float(np.mean(np.abs(face.astype(np.float32) - self._previous_face)))
            self.temporal_errors.append(delta)
            baseline = float(np.median(self._recent_deltas)) if self._recent_deltas else max(delta, 1.0)
            if chunk_boundary:
                self.boundary_ratios.append(delta / max(1.0, baseline))
            self._recent_deltas.append(delta)
        self._previous_face = face

        if (
            self.analyzer is None
            or self.source_embedding is None
            or index % max(1, self.sample_interval) != 0
        ):
            return
        faces = self.analyzer.find_faces(output)
        candidates = [face for face in faces if getattr(face, "embedding", None) is not None]
        if not candidates:
            return
        chosen = max(candidates, key=lambda face: bbox_iou(face.bbox, np.asarray(track.bbox)))
        self.identity_scores.append(float(cosine_similarity(chosen.embedding, self.source_embedding)))
        if len(chosen.kps) >= 5 and track.kps:
            expected = np.asarray(track.kps, dtype=np.float32)
            actual = np.asarray(chosen.kps, dtype=np.float32)
            scale = max(1.0, float(np.linalg.norm(expected[1] - expected[0])))
            self.pose_errors.append(float(np.mean(np.linalg.norm(actual - expected, axis=1)) / scale))

    @staticmethod
    def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
        data = np.asarray(list(values), dtype=np.float32)
        if data.size == 0:
            return {"count": 0, "mean": None, "std": None, "p95": None}
        return {
            "count": int(data.size),
            "mean": round(float(data.mean()), 6),
            "std": round(float(data.std()), 6),
            "p95": round(float(np.percentile(data, 95)), 6),
        }

    def report(self) -> dict[str, Any]:
        return {
            "identity_similarity": self._summary(self.identity_scores),
            "normalized_landmark_error": self._summary(self.pose_errors),
            "outside_mask_mae": self._summary(self.outside_mae),
            "temporal_frame_delta": self._summary(self.temporal_errors),
            "chunk_boundary_jump_ratio": self._summary(self.boundary_ratios),
        }
