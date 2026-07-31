from __future__ import annotations

from dataclasses import dataclass

import numpy as np
try:
    from insightface.app.common import Face
except ModuleNotFoundError:  # Permite pruebas unitarias sin descargar modelos pesados.
    class Face(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

        def __init__(self, data=None, **kwargs):
            super().__init__(data or {})
            self.update(kwargs)

from .math_utils import bbox_area, bbox_iou


@dataclass(frozen=True)
class DetectionStats:
    detected: int
    recognized: int
    full_scan: bool


class SelectiveFaceAnalyzer:
    """Usa SCRFD en todos los intervalos de detección y ArcFace solo en candidatos.

    ``FaceAnalysis.get`` ejecuta cada módulo cargado sobre cada rostro. Este adaptador
    llama directamente al detector y limita el reconocimiento a los candidatos con
    mayor continuidad espacial, manteniendo un escaneo completo periódico para
    recuperar al sujeto después de cortes o movimientos bruscos.
    """

    def __init__(self, face_app, max_faces: int, max_recognition_candidates: int) -> None:
        self.detector = face_app.det_model
        self.recognizer = face_app.models.get("recognition")
        if self.recognizer is None:
            raise RuntimeError("El paquete de modelos no contiene un modelo de reconocimiento.")
        self.max_faces = max(1, int(max_faces))
        self.max_recognition_candidates = max(1, int(max_recognition_candidates))

    @staticmethod
    def _continuity_score(face, previous_bbox: np.ndarray, frame_shape) -> float:
        h, w = frame_shape[:2]
        previous = np.asarray(previous_bbox, dtype=np.float32)
        current = np.asarray(face.bbox, dtype=np.float32)
        iou = bbox_iou(current, previous)

        pcx = float(previous[0] + previous[2]) * 0.5
        pcy = float(previous[1] + previous[3]) * 0.5
        ccx = float(current[0] + current[2]) * 0.5
        ccy = float(current[1] + current[3]) * 0.5
        diagonal = max(1.0, float(np.hypot(w, h)))
        center_similarity = 1.0 - min(1.0, float(np.hypot(ccx - pcx, ccy - pcy)) / diagonal)
        return 0.78 * iou + 0.22 * center_similarity

    @staticmethod
    def _quality_score(face) -> float:
        return bbox_area(face.bbox) * float(face.det_score or 0.0)

    def analyze(
        self,
        frame: np.ndarray,
        previous_bbox: np.ndarray | None,
        full_scan: bool,
    ) -> tuple[list[Face], DetectionStats]:
        bboxes, keypoints = self.detector.detect(frame, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return [], DetectionStats(0, 0, full_scan)

        detected_faces: list[Face] = []
        for index in range(int(bboxes.shape[0])):
            kps = keypoints[index] if keypoints is not None else None
            detected_faces.append(
                Face(
                    bbox=bboxes[index, 0:4],
                    kps=kps,
                    det_score=float(bboxes[index, 4]),
                )
            )

        if full_scan or previous_bbox is None:
            faces = sorted(detected_faces, key=self._quality_score, reverse=True)[
                : self.max_faces
            ]
            candidates = faces
        else:
            faces = sorted(
                detected_faces,
                key=lambda face: self._continuity_score(face, previous_bbox, frame.shape),
                reverse=True,
            )[: self.max_faces]
            candidates = faces[: self.max_recognition_candidates]

        for face in candidates:
            self.recognizer.get(frame, face)

        return candidates, DetectionStats(len(detected_faces), len(candidates), full_scan)
