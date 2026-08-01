from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .engine import build_providers, preload_gpu_runtime
from .math_utils import bbox_area, bbox_iou
from .modeling import (
    DetectionStats,
    FaceData,
    ModelBundle,
    ModelCapabilities,
    SwapResult,
)

BACKEND_NAME = "insightface_inswapper"


class _FallbackFace(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


def _new_native_face(**payload):
    try:
        from insightface.app.common import Face
    except ModuleNotFoundError:
        return _FallbackFace(payload)
    return Face(payload)


class SelectiveFaceAnalyzer:
    """Optimización específica de InsightFace para reconocer pocos candidatos."""

    def __init__(self, face_app: Any, max_faces: int, max_recognition_candidates: int) -> None:
        self.detector = face_app.det_model
        self.recognizer = face_app.models.get("recognition")
        if self.recognizer is None:
            raise RuntimeError("El paquete de modelos no contiene reconocimiento.")
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
        center_similarity = 1.0 - min(
            1.0,
            float(np.hypot(ccx - pcx, ccy - pcy)) / diagonal,
        )
        return 0.78 * iou + 0.22 * center_similarity

    @staticmethod
    def _quality_score(face) -> float:
        return bbox_area(face.bbox) * float(face.det_score or 0.0)

    def analyze(
        self,
        frame: np.ndarray,
        previous_bbox: np.ndarray | None,
        full_scan: bool,
    ) -> tuple[list[Any], DetectionStats]:
        bboxes, keypoints = self.detector.detect(frame, max_num=0, metric="default")
        if bboxes.shape[0] == 0:
            return [], DetectionStats(0, 0, full_scan)

        detected_faces = []
        for index in range(int(bboxes.shape[0])):
            kps = keypoints[index] if keypoints is not None else None
            detected_faces.append(
                _new_native_face(
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
                key=lambda face: self._continuity_score(
                    face, previous_bbox, frame.shape
                ),
                reverse=True,
            )[: self.max_faces]
            candidates = faces[: self.max_recognition_candidates]

        for face in candidates:
            self.recognizer.get(frame, face)

        return candidates, DetectionStats(
            len(detected_faces),
            len(candidates),
            full_scan,
        )


def _optional_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).copy()


def _to_face_data(face: Any) -> FaceData:
    kps = getattr(face, "kps", None)
    if kps is None:
        kps = np.empty((0, 2), dtype=np.float32)
    return FaceData(
        bbox=np.asarray(face.bbox, dtype=np.float32).copy(),
        kps=np.asarray(kps, dtype=np.float32).copy(),
        det_score=float(getattr(face, "det_score", 0.0) or 0.0),
        embedding=_optional_array(getattr(face, "embedding", None)),
        pose=_optional_array(getattr(face, "pose", None)),
        native=face,
    )


def _to_insightface_face(face: FaceData):
    from insightface.app.common import Face

    payload: dict[str, Any] = {}
    native = face.native
    if native is not None and hasattr(native, "items"):
        payload.update(
            {
                key: value.copy() if isinstance(value, np.ndarray) else value
                for key, value in native.items()
            }
        )
    payload.update(
        {
            "bbox": np.asarray(face.bbox, dtype=np.float32).copy(),
            "kps": np.asarray(face.kps, dtype=np.float32).copy(),
            "det_score": float(face.det_score),
        }
    )
    if face.embedding is not None:
        payload["embedding"] = np.asarray(face.embedding, dtype=np.float32).copy()
    if face.pose is not None:
        payload["pose"] = np.asarray(face.pose, dtype=np.float32).copy()
    return Face(payload)


class InsightFaceAnalyzer:
    """Adaptador que oculta FaceAnalysis y la optimización selectiva."""

    def __init__(self, face_app: Any, max_faces: int, max_recognition_candidates: int) -> None:
        self._face_app = face_app
        self._selective = SelectiveFaceAnalyzer(
            face_app,
            max_faces=max_faces,
            max_recognition_candidates=max_recognition_candidates,
        )

    def find_faces(self, image: np.ndarray) -> list[FaceData]:
        return [_to_face_data(face) for face in self._face_app.get(image)]

    def analyze(
        self,
        frame: np.ndarray,
        previous_bbox: np.ndarray | None,
        full_scan: bool,
    ) -> tuple[list[FaceData], DetectionStats]:
        faces, stats = self._selective.analyze(frame, previous_bbox, full_scan)
        return [_to_face_data(face) for face in faces], stats


class InsightFaceSwapper:
    """Adaptador del contrato de INSwapper al contrato neutral FaceSwapper."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def swap(
        self,
        frame: np.ndarray,
        target_face: FaceData,
        source_face: FaceData,
    ) -> SwapResult:
        crop, affine = self._model.get(
            frame,
            _to_insightface_face(target_face),
            _to_insightface_face(source_face),
            paste_back=False,
        )
        if crop is None or affine is None:
            raise RuntimeError("El modelo INSwapper no devolvió recorte y transformación.")
        return SwapResult(
            crop=np.asarray(crop),
            affine=np.asarray(affine, dtype=np.float32),
        )


class InsightFaceAnalysisServices:
    def __init__(self, analyzer, providers, runtime) -> None:
        self.analyzer = analyzer
        self.providers = providers
        self.runtime = runtime


def create_insightface_analysis_services(config: Any) -> InsightFaceAnalysisServices:
    """Crea detección/reconocimiento sin imponer un generador concreto."""

    engine_config = config.engine
    import insightface
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    preload_gpu_runtime()
    providers = build_providers(engine_config)
    face_app = FaceAnalysis(
        name=engine_config.model_pack,
        allowed_modules=list(engine_config.allowed_modules),
        providers=providers,
    )
    face_app.prepare(
        ctx_id=int(engine_config.cuda.get("device_id", 0)),
        det_thresh=engine_config.det_thresh,
        det_size=engine_config.det_size,
    )
    analyzer = InsightFaceAnalyzer(
        face_app,
        max_faces=engine_config.max_faces,
        max_recognition_candidates=config.tracking.max_recognition_candidates,
    )
    return InsightFaceAnalysisServices(
        analyzer=analyzer,
        providers=tuple(providers),
        runtime={
            "insightface": getattr(insightface, "__version__", "unknown"),
            "onnxruntime": ort.__version__,
            "providers_available": ort.get_available_providers(),
            "model_pack": engine_config.model_pack,
        },
    )


class InsightFaceBackendFactory:
    def create(self, config: Any, model_path: Path) -> ModelBundle:
        engine_config = config.engine
        import insightface

        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if "inswapper_128" in model_path.name.lower() and engine_config.model_pack != "buffalo_l":
            raise ValueError(
                "inswapper_128 requiere embeddings del paquete buffalo_l; "
                "restaura engine.model_pack: buffalo_l."
            )

        services = create_insightface_analysis_services(config)
        model = insightface.model_zoo.get_model(
            str(model_path),
            download=False,
            providers=list(services.providers),
        )
        if model is None:
            raise RuntimeError(f"No se pudo cargar el modelo swapper: {model_path}")

        runtime = dict(services.runtime)
        runtime.update({"generator_model": str(model_path)})
        return ModelBundle(
            backend=BACKEND_NAME,
            analyzer=services.analyzer,
            swapper=InsightFaceSwapper(model),
            providers=services.providers,
            runtime=runtime,
            capabilities=ModelCapabilities(
                generator="inswapper_128",
                native_output_size=128,
                geometry_conditioning="none",
                geometry_postprocess="none",
                temporal_generation="frame_independent",
            ),
            model_artifacts=(model_path,),
        )
