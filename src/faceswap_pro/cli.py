from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import onnxruntime as ort
import typer
from rich.console import Console
from rich.panel import Panel

from .config import load_config
from .engine import preload_gpu_runtime
from .paths import (
    DEFAULT_CONFIG,
    DEFAULT_INPUT_VIDEO,
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
    swapper_model: Path = typer.Option(
        DEFAULT_SWAPPER_MODEL,
        "--swapper-model",
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Modelo INSwapper ONNX.",
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
    resolved_output = build_output_path(input_video, output_video, output_dir)
    config = load_config(config_file)
    console.print(f"[cyan]Entrada:[/cyan] {input_video}")
    console.print(f"[cyan]Salida:[/cyan] {resolved_output}")
    run_pipeline(
        input_video=input_video,
        source_dir=source_dir,
        target_reference=target_reference,
        swapper_model=swapper_model,
        output_video=resolved_output,
        manifest_dir=manifest_dir,
        config=config,
    )


@app.command()
def doctor() -> None:
    """Comprueba CUDA EP, FFmpeg, NVDEC y NVENC antes de procesar."""
    preload_gpu_runtime()
    ffmpeg_path = select_ffmpeg("h264_nvenc")
    ffmpeg = str(ffmpeg_path) if ffmpeg_path else None
    nvenc = bool(ffmpeg_path and ffmpeg_has_encoder(ffmpeg_path, "h264_nvenc"))
    report = {
        "project": "FaceSwap-Pro",
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "cuda_provider_ok": "CUDAExecutionProvider" in ort.get_available_providers(),
        "ffmpeg": ffmpeg,
        "ffmpeg_candidates": [str(path) for path in ffmpeg_candidates()],
        "h264_nvenc": nvenc,
        "cuda_hw_decode": bool(ffmpeg_path and ffmpeg_has_hwaccel(ffmpeg_path, "cuda")),
        "logical_cpu_count": os.cpu_count(),
    }
    console.print(Panel.fit(json.dumps(report, indent=2, ensure_ascii=False), title="Diagnóstico"))
    if not report["cuda_provider_ok"]:
        raise typer.Exit(code=2)
    if not ffmpeg or not nvenc:
        console.print("[yellow]GPU ONNX disponible, pero falta FFmpeg con h264_nvenc.[/yellow]")


if __name__ == "__main__":
    app()
