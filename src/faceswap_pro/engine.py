from __future__ import annotations

from pathlib import Path

import insightface
import onnxruntime as ort
from insightface.app import FaceAnalysis


def preload_gpu_runtime() -> None:
    try:
        ort.preload_dlls(directory="")
    except Exception:
        # En Linux o con CUDA instalada en sistema puede no ser necesario.
        pass


def build_providers(config) -> list:
    available = set(ort.get_available_providers())
    providers = []
    for name in config.providers:
        if name not in available:
            continue
        if name == "CUDAExecutionProvider":
            providers.append((name, {"device_id": 0, **{k: str(v) for k, v in config.cuda.items()}}))
        else:
            providers.append(name)
    if not providers:
        raise RuntimeError(f"No hay proveedores ONNX Runtime utilizables. Disponibles: {sorted(available)}")
    return providers


def initialize_models(engine_config, swapper_model: Path):
    if "inswapper_128" in swapper_model.name.lower() and engine_config.model_pack != "buffalo_l":
        raise ValueError(
            "inswapper_128 requiere embeddings del paquete buffalo_l; "
            "restaura engine.model_pack: buffalo_l."
        )
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
    swapper = insightface.model_zoo.get_model(
        str(swapper_model), download=False, providers=providers
    )
    if swapper is None:
        raise RuntimeError(f"No se pudo cargar el modelo swapper: {swapper_model}")
    return face_app, swapper, providers
