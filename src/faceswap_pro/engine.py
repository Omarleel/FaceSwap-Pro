from __future__ import annotations

from importlib import import_module
from pathlib import Path

from .modeling import (
    ModelBundle,
    available_model_backends,
    create_model_bundle,
    register_model_backend,
)

_BUILTINS_REGISTERED = False


def _load_onnxruntime():
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX Runtime no está instalado. Ejecuta el instalador del proyecto antes de procesar."
        ) from exc
    return ort


def preload_gpu_runtime() -> None:
    # ONNX Runtime recomienda importar PyTorch primero cuando ambos frameworks
    # comparten CUDA/cuDNN. Así ORT reutiliza las DLL de la build de PyTorch en
    # lugar de precargar otra familia de CUDA desde site-packages.
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    ort = _load_onnxruntime()
    try:
        ort.preload_dlls()
    except Exception:
        # En Linux o con CUDA instalada en el sistema puede no ser necesario.
        pass


def build_providers(config) -> list:
    ort = _load_onnxruntime()
    available = set(ort.get_available_providers())
    providers = []
    for name in config.providers:
        if name not in available:
            continue
        if name == "CUDAExecutionProvider":
            providers.append(
                (name, {"device_id": 0, **{k: str(v) for k, v in config.cuda.items()}})
            )
        else:
            providers.append(name)
    if not providers:
        raise RuntimeError(
            f"No hay proveedores ONNX Runtime utilizables. Disponibles: {sorted(available)}"
        )
    return providers


def register_builtin_model_backends() -> None:
    """Registra adaptadores incluidos sin importar frameworks hasta que se usan."""

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from .hififace_backend import BACKEND_NAME as HIFIFACE_BACKEND_NAME
    from .hififace_backend import HifiFace3DMMBackendFactory
    from .insightface_backend import BACKEND_NAME as INSWAPPER_BACKEND_NAME
    from .insightface_backend import InsightFaceBackendFactory
    from .mesh_assisted_backend import (
        BACKEND_NAME as MESH_BACKEND_NAME,
        LEGACY_BACKEND_NAME,
        LegacyMediaPipe3DHybridBackendFactory,
        MediaPipeMeshAssistedBackendFactory,
    )

    register_model_backend(INSWAPPER_BACKEND_NAME, InsightFaceBackendFactory())
    register_model_backend(MESH_BACKEND_NAME, MediaPipeMeshAssistedBackendFactory())
    register_model_backend(LEGACY_BACKEND_NAME, LegacyMediaPipe3DHybridBackendFactory())
    register_model_backend(HIFIFACE_BACKEND_NAME, HifiFace3DMMBackendFactory())
    _BUILTINS_REGISTERED = True


def load_configured_model_plugins(config) -> None:
    """Importa módulos que registran backends adicionales desde la configuración."""

    for module_name in config.engine.plugins:
        name = module_name.strip()
        if not name:
            continue
        try:
            import_module(name)
        except Exception as exc:
            raise RuntimeError(
                f"No se pudo cargar el plugin de modelo {name!r}."
            ) from exc


def initialize_models(config, model_path: Path) -> ModelBundle:
    """Punto de composición: selecciona el backend configurado y crea sus servicios."""

    register_builtin_model_backends()
    load_configured_model_plugins(config)
    return create_model_bundle(config.engine.backend, config, model_path)


__all__ = [
    "ModelBundle",
    "available_model_backends",
    "build_providers",
    "initialize_models",
    "load_configured_model_plugins",
    "preload_gpu_runtime",
    "register_builtin_model_backends",
    "register_model_backend",
]
