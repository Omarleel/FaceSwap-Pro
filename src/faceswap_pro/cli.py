from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from .config import load_config
from .engine import (
    available_model_backends,
    preload_gpu_runtime,
    register_builtin_model_backends,
)
from .observability import (
    configure_observability,
    log_exception,
    log_problem,
    profile_span,
)
from .paths import (
    DEFAULT_CONFIG,
    DEFAULT_INPUT_VIDEO,
    DEFAULT_LOG_DIR,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SOURCE_DIR,
    DEFAULT_SWAPPER_MODEL,
    DEFAULT_TARGET_REFERENCE,
    build_output_path,
    create_project_directories,
)
from .pipeline import run_pipeline
from .runtime import (
    ffmpeg_candidates,
    ffmpeg_has_encoder,
    ffmpeg_has_hwaccel,
    select_ffmpeg,
)

app = typer.Typer(
    no_args_is_help=True,
    help="FaceSwap-Pro: reemplazo facial local optimizado para GPU NVIDIA.",
)
console = Console()


@app.callback()
def configure_runtime(
    ctx: typer.Context,
    log_dir: Path = typer.Option(
        DEFAULT_LOG_DIR,
        "--log-dir",
        file_okay=False,
        help="Carpeta para logs estructurados y métricas de perfilado.",
    ),
) -> None:
    """Inicializa observabilidad para todos los comandos."""
    session = configure_observability(
        log_dir,
        command=ctx.invoked_subcommand or "cli",
    )
    ctx.call_on_close(session.close)


def _resolve_model_path(config, override: Path | None = None) -> Path:
    configured = config.engine.options.get("model_path")
    if override is not None:
        return override
    if configured not in (None, ""):
        return Path(str(configured))
    return DEFAULT_SWAPPER_MODEL


def _optional_path(options: dict[str, object], name: str) -> Path | None:
    value = options.get(name)
    if value in (None, ""):
        return None
    return Path(str(value))


def _hififace_readiness(config, model_path: Path) -> dict[str, object]:
    options = config.engine.options
    iteration = options.get("checkpoint_iteration")
    generator_name = (
        "generator.pth" if iteration in (None, "") else f"generator_{int(iteration)}.pth"
    )
    repository = _optional_path(options, "repository_path")
    f3d = _optional_path(options, "f_3d_checkpoint_path")
    fid = _optional_path(options, "f_id_checkpoint_path")
    bfm = _optional_path(options, "bfm_folder")
    device = str(options.get("device", "cuda:0"))
    bfm_required = (
        "01_MorphableModel.mat",
        "BFM_exp_idx.mat",
        "BFM_front_idx.mat",
        "BFM_model_front.mat",
        "Exp_Pca.bin",
        "facemodel_info.mat",
        "select_vertex_id.mat",
        "similarity_Lm3D_all.mat",
        "std_exp.txt",
    )

    checkpoint = model_path / generator_name
    repository_ready = bool(
        repository is not None
        and repository.is_dir()
        and (repository / "models" / "model.py").is_file()
        and (repository / "models" / "generator.py").is_file()
    )
    bfm_ready = bool(
        bfm is not None
        and bfm.is_dir()
        and all((bfm / name).is_file() for name in bfm_required)
    )
    checks = {
        "repository": repository_ready,
        "checkpoint_directory": model_path.is_dir(),
        "generator_checkpoint": checkpoint.is_file(),
        "f_3d_checkpoint": bool(f3d is not None and f3d.is_file()),
        "f_id_checkpoint": bool(fid is not None and fid.is_file()),
        "bfm_folder": bool(bfm is not None and bfm.is_dir()),
        "bfm_files": bfm_ready,
        "torch_installed": importlib.util.find_spec("torch") is not None,
        "torchvision_installed": importlib.util.find_spec("torchvision") is not None,
        "kornia_installed": importlib.util.find_spec("kornia") is not None,
        "loguru_installed": importlib.util.find_spec("loguru") is not None,
    }
    cuda_available = False
    torch_error = None
    torch_version = None
    torch_cuda_runtime = None
    torch_gpu = None
    if checks["torch_installed"]:
        try:
            import torch

            torch_version = torch.__version__
            torch_cuda_runtime = torch.version.cuda
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                torch_gpu = torch.cuda.get_device_name(0)
        except Exception as exc:
            cuda_available = False
            torch_error = f"{type(exc).__name__}: {exc}"
    checks["torch_device"] = not device.startswith("cuda") or cuda_available
    return {
        "model_path": str(model_path),
        "expected_generator": str(checkpoint),
        "repository_path": None if repository is None else str(repository),
        "device": device,
        "torch_version": torch_version,
        "torch_cuda_runtime": torch_cuda_runtime,
        "torch_cuda_available": cuda_available,
        "torch_gpu": torch_gpu,
        "torch_error": torch_error,
        "checks": checks,
        "ready": all(checks.values()),
    }


@app.command("init")
def init_project() -> None:
    """Crea la estructura estándar de entradas, modelos y salidas."""
    directories = create_project_directories()
    console.print("[bold green]Estructura preparada:[/bold green]")
    for directory in directories:
        console.print(f"  {directory}")


@app.command()
def run(
    input_video: Path = typer.Option(
        DEFAULT_INPUT_VIDEO,
        "--input",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Video de entrada.",
    ),
    source_dir: Path = typer.Option(
        DEFAULT_SOURCE_DIR,
        "--source-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Carpeta con fotografías del rostro de origen.",
    ),
    target_reference: Path = typer.Option(
        DEFAULT_TARGET_REFERENCE,
        "--target-ref",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Referencia del sujeto que será reemplazado en el video.",
    ),
    model_path: Path | None = typer.Option(
        None,
        "--model",
        "--model-path",
        "--swapper-model",
        exists=False,
        file_okay=True,
        dir_okay=True,
        help=(
            "Archivo o directorio del modelo principal. Si se omite, se usa "
            "engine.options.model_path o models/inswapper_128.onnx."
        ),
    ),
    output_video: Path | None = typer.Option(
        None,
        "--output",
        help="Ruta completa opcional. Si se omite, se genera un nombre fechado.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        file_okay=False,
        help="Carpeta predeterminada para videos generados.",
    ),
    manifest_dir: Path = typer.Option(
        DEFAULT_MANIFEST_DIR,
        "--manifest-dir",
        file_okay=False,
        help="Carpeta para manifiestos y métricas del procesamiento.",
    ),
    config_file: Path = typer.Option(
        DEFAULT_CONFIG,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Perfil YAML de rendimiento/calidad.",
    ),
) -> None:
    """Procesa un sujeto objetivo identificado por una imagen de referencia."""
    with profile_span("cli.resolve_run_configuration"):
        resolved_output = build_output_path(input_video, output_video, output_dir)
        config = load_config(config_file)
        effective_model = _resolve_model_path(config, model_path)
    if not effective_model.exists():
        log_problem(
            "No existe el modelo principal configurado",
            model_path=str(effective_model),
        )
        raise typer.BadParameter(
            f"No existe el modelo principal: {effective_model}",
            param_hint="--model-path / engine.options.model_path",
        )
    console.print(f"[cyan]Entrada:[/cyan] {input_video}")
    console.print(f"[cyan]Salida:[/cyan] {resolved_output}")
    console.print(f"[cyan]Modelo principal:[/cyan] {effective_model}")
    with profile_span("cli.run_pipeline", input_video=str(input_video)):
        run_pipeline(
            input_video=input_video,
            source_dir=source_dir,
            target_reference=target_reference,
            model_path=effective_model,
            output_video=resolved_output,
            manifest_dir=manifest_dir,
            config=config,
        )


@app.command()
def doctor(
    config_file: Path | None = typer.Option(
        None,
        "--config",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Perfil opcional para validar también los artefactos del backend.",
    ),
) -> None:
    """Comprueba GPU, video y los artefactos del backend configurado."""
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        log_exception("ONNX Runtime no está instalado", exc, command="doctor")
        console.print("[red]ONNX Runtime no está instalado en este entorno.[/red]")
        raise typer.Exit(code=2) from exc
    preload_gpu_runtime()
    register_builtin_model_backends()
    ffmpeg_path = select_ffmpeg("h264_nvenc")
    ffmpeg = str(ffmpeg_path) if ffmpeg_path else None
    nvenc = bool(ffmpeg_path and ffmpeg_has_encoder(ffmpeg_path, "h264_nvenc"))
    geometry_model = Path("models/face_landmarker.task")
    mediapipe_installed = importlib.util.find_spec("mediapipe") is not None
    report = {
        "project": "FaceSwap-Pro",
        "model_backends": list(available_model_backends()),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "cuda_provider_ok": "CUDAExecutionProvider" in ort.get_available_providers(),
        "ffmpeg": ffmpeg,
        "ffmpeg_candidates": [str(path) for path in ffmpeg_candidates()],
        "h264_nvenc": nvenc,
        "cuda_hw_decode": bool(ffmpeg_path and ffmpeg_has_hwaccel(ffmpeg_path, "cuda")),
        "logical_cpu_count": os.cpu_count(),
        "mediapipe_installed": mediapipe_installed,
        "face_landmarker_model": str(geometry_model),
        "face_landmarker_model_exists": geometry_model.is_file(),
        "mediapipe_mesh_assist_ready": mediapipe_installed and geometry_model.is_file(),
        "torch_installed": importlib.util.find_spec("torch") is not None,
    }
    configured_backend_ready = True
    configured_backend = None
    if config_file is not None:
        selected_config = load_config(config_file)
        configured_model = _resolve_model_path(selected_config)
        configured_backend = selected_config.engine.backend
        report["configured_backend"] = configured_backend
        report["configured_model"] = str(configured_model)
        c2pa_tool = str(getattr(selected_config.provenance, "c2pa_tool", "c2patool"))
        c2pa_candidate = Path(c2pa_tool).expanduser()
        c2pa_resolved = (
            str(c2pa_candidate.resolve()) if c2pa_candidate.is_file() else shutil.which(c2pa_tool)
        )
        report["c2pa"] = {
            "enabled": bool(getattr(selected_config.provenance, "c2pa_enabled", False)),
            "required": bool(getattr(selected_config.provenance, "c2pa_required", False)),
            "tool": c2pa_resolved,
            "ready": bool(c2pa_resolved),
            "production_credentials": bool(
                getattr(selected_config.provenance, "c2pa_sign_cert", None)
                and getattr(selected_config.provenance, "c2pa_private_key", None)
            ),
        }
        if configured_backend == "hififace_3dmm":
            readiness = _hififace_readiness(selected_config, configured_model)
            report["hififace_3dmm"] = readiness
            configured_backend_ready = bool(readiness["ready"])
        elif configured_backend == "dreamid_v":
            from .dreamidv_backend import dreamidv_readiness

            readiness = dreamidv_readiness(
                selected_config,
                configured_model,
                probe_environment=True,
            )
            report["dreamid_v"] = readiness
            configured_backend_ready = bool(readiness["ready"])
        elif configured_backend in {
            "insightface_inswapper_mediapipe_mesh",
            "mediapipe_3d_hybrid",
        }:
            configured_backend_ready = bool(report["mediapipe_mesh_assist_ready"])
            report["backend_note"] = (
                "Postproceso por malla; geometry_conditioning=none."
            )
        else:
            configured_backend_ready = configured_model.is_file()
        if bool(getattr(selected_config.provenance, "c2pa_required", False)):
            configured_backend_ready = configured_backend_ready and bool(report["c2pa"]["ready"])
        report["configured_backend_ready"] = configured_backend_ready
    console.print(Panel.fit(json.dumps(report, indent=2, ensure_ascii=False), title="Diagnóstico"))
    if not report["cuda_provider_ok"]:
        raise typer.Exit(code=2)
    if not ffmpeg or not nvenc:
        log_problem(
            "GPU ONNX disponible, pero falta FFmpeg con h264_nvenc",
            ffmpeg=ffmpeg,
            h264_nvenc=nvenc,
        )
        console.print("[yellow]GPU ONNX disponible, pero falta FFmpeg con h264_nvenc.[/yellow]")
    if configured_backend in {
        "insightface_inswapper_mediapipe_mesh",
        "mediapipe_3d_hybrid",
    } and not report["mediapipe_mesh_assist_ready"]:
        log_problem(
            "El backend asistido por malla no está listo",
            mediapipe_installed=mediapipe_installed,
            face_landmarker_model_exists=geometry_model.is_file(),
        )
        console.print(
            "[yellow]El backend asistido por malla no está listo: instala MediaPipe "
            "y añade models/face_landmarker.task.[/yellow]"
        )
    if config_file is not None and not configured_backend_ready:
        log_problem(
            "El backend configurado no está listo",
            backend=configured_backend,
            config_file=str(config_file),
        )
        console.print(
            "[red]El backend configurado no está listo. Revisa el bloque de "
            "diagnóstico anterior.[/red]"
        )
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
