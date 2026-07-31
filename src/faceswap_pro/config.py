from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class EngineConfig:
    model_pack: str
    det_size: tuple[int, int]
    det_thresh: float
    providers: list[str]
    cuda: dict[str, Any]
    allowed_modules: tuple[str, ...]
    max_faces: int


@dataclass(frozen=True)
class IdentityConfig:
    source_min_score: float
    target_min_similarity: float
    max_source_images: int


@dataclass(frozen=True)
class TrackingConfig:
    smoothing: float
    max_missing_frames: int
    scene_cut_threshold: float
    detection_interval: int
    full_scan_interval: int
    max_recognition_candidates: int
    optical_flow: bool
    flow_win_size: int
    flow_max_level: int
    flow_max_error: float


@dataclass(frozen=True)
class BlendConfig:
    aligned_size: int
    mask_shrink: float
    mask_blur_ratio: float
    color_match_strength: float
    detail_strength: float
    roi_enabled: bool
    roi_margin: float
    interpolation: str


@dataclass(frozen=True)
class RestorerConfig:
    enabled: bool
    model_path: Path
    input_size: int
    output_range: str


@dataclass(frozen=True)
class EncodingConfig:
    codec: str
    preset: str
    cq: int
    audio_bitrate: str
    fallback_codec: str
    fallback_preset: str
    fallback_crf: int


@dataclass(frozen=True)
class PerformanceConfig:
    decoder: str
    hardware_decode: bool
    reader_queue: int
    analysis_queue: int
    writer_queue: int
    max_inflight: int
    postprocess_workers: int
    opencv_threads: int


@dataclass(frozen=True)
class WatermarkConfig:
    text: str


@dataclass(frozen=True)
class AppConfig:
    engine: EngineConfig
    identity: IdentityConfig
    tracking: TrackingConfig
    blend: BlendConfig
    restorer: RestorerConfig
    encoding: EncodingConfig
    performance: PerformanceConfig
    watermark: WatermarkConfig


def _positive_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(value))
    except (TypeError, ValueError):
        return default


def load_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    engine = raw["engine"]
    identity = raw["identity"]
    tracking = raw["tracking"]
    blend = raw["blend"]
    restorer = raw["restorer"]
    encoding = raw["encoding"]
    performance = raw.get("performance", {})

    allowed_modules = tuple(
        str(value) for value in engine.get("allowed_modules", ["detection", "recognition"])
    )
    if "detection" not in allowed_modules or "recognition" not in allowed_modules:
        raise ValueError(
            "engine.allowed_modules debe incluir detection y recognition para INSwapper."
        )

    interpolation = str(blend.get("interpolation", "cubic")).lower()
    if interpolation not in {"linear", "cubic", "lanczos"}:
        raise ValueError("blend.interpolation debe ser linear, cubic o lanczos.")

    decoder = str(performance.get("decoder", "auto")).lower()
    if decoder not in {"auto", "opencv", "ffmpeg", "ffmpeg_cuda"}:
        raise ValueError(
            "performance.decoder debe ser auto, opencv, ffmpeg o ffmpeg_cuda."
        )

    return AppConfig(
        engine=EngineConfig(
            model_pack=str(engine["model_pack"]),
            det_size=tuple(int(x) for x in engine["det_size"]),
            det_thresh=float(engine["det_thresh"]),
            providers=list(engine["providers"]),
            cuda=dict(engine.get("cuda", {})),
            allowed_modules=allowed_modules,
            max_faces=_positive_int(engine.get("max_faces", 10), 10),
        ),
        identity=IdentityConfig(
            source_min_score=float(identity["source_min_score"]),
            target_min_similarity=float(identity["target_min_similarity"]),
            max_source_images=_positive_int(identity["max_source_images"], 40),
        ),
        tracking=TrackingConfig(
            smoothing=float(tracking["smoothing"]),
            max_missing_frames=_positive_int(tracking["max_missing_frames"], 6),
            scene_cut_threshold=float(tracking["scene_cut_threshold"]),
            detection_interval=_positive_int(tracking.get("detection_interval", 3), 3),
            full_scan_interval=_positive_int(tracking.get("full_scan_interval", 90), 90),
            max_recognition_candidates=_positive_int(
                tracking.get("max_recognition_candidates", 2), 2
            ),
            optical_flow=bool(tracking.get("optical_flow", True)),
            flow_win_size=_positive_int(tracking.get("flow_win_size", 31), 31, 5),
            flow_max_level=_positive_int(tracking.get("flow_max_level", 3), 3, 0),
            flow_max_error=float(tracking.get("flow_max_error", 25.0)),
        ),
        blend=BlendConfig(
            aligned_size=_positive_int(blend["aligned_size"], 384, 128),
            mask_shrink=float(blend["mask_shrink"]),
            mask_blur_ratio=float(blend["mask_blur_ratio"]),
            color_match_strength=float(blend["color_match_strength"]),
            detail_strength=float(blend["detail_strength"]),
            roi_enabled=bool(blend.get("roi_enabled", True)),
            roi_margin=float(blend.get("roi_margin", 0.15)),
            interpolation=interpolation,
        ),
        restorer=RestorerConfig(
            enabled=bool(restorer["enabled"]),
            model_path=Path(restorer["model_path"]),
            input_size=int(restorer["input_size"]),
            output_range=str(restorer.get("output_range", "auto")),
        ),
        encoding=EncodingConfig(**encoding),
        performance=PerformanceConfig(
            decoder=decoder,
            hardware_decode=bool(performance.get("hardware_decode", True)),
            reader_queue=_positive_int(performance.get("reader_queue", 6), 6),
            analysis_queue=_positive_int(performance.get("analysis_queue", 3), 3),
            writer_queue=_positive_int(performance.get("writer_queue", 6), 6),
            max_inflight=_positive_int(performance.get("max_inflight", 6), 6),
            postprocess_workers=max(0, int(performance.get("postprocess_workers", 0))),
            opencv_threads=max(0, int(performance.get("opencv_threads", 0))),
        ),
        watermark=WatermarkConfig(**raw["watermark"]),
    )
