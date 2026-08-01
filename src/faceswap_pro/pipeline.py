from __future__ import annotations

import platform
import tempfile
import time
from pathlib import Path

from rich.console import Console

from .blend import ProfessionalBlender
from .engine import initialize_models
from .identity import build_source_identity, load_reference_embedding
from .modeling import ModelBundle
from .parallel_pipeline import run_parallel_frames
from .provenance import write_manifest
from .restorer import build_face_restorer
from .tracking import TemporalFaceTracker
from .videoio import RawFFmpegWriter, mux_original_audio, open_video_reader

console = Console()


def run_pipeline(
    input_video: Path,
    source_dir: Path,
    target_reference: Path,
    model_path: Path,
    output_video: Path,
    manifest_dir: Path,
    config,
    *,
    model_bundle: ModelBundle | None = None,
) -> tuple[Path, Path]:
    """Orquesta el caso de uso dependiendo solo de contratos de alto nivel.

    ``model_bundle`` permite inyectar otro backend o dobles de prueba. La CLI usa la
    fábrica registrada según ``engine.backend``.
    """

    for required in (input_video, target_reference):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    models = model_bundle or initialize_models(config, model_path)
    analyzer = models.analyzer
    source_face, source_paths = build_source_identity(
        analyzer,
        source_dir,
        config.identity.source_min_score,
        config.identity.max_source_images,
    )
    target_embedding = load_reference_embedding(
        analyzer,
        target_reference,
        config.identity.source_min_score,
    )
    tracker = TemporalFaceTracker(
        target_embedding,
        config.identity.target_min_similarity,
        config.tracking.smoothing,
        config.tracking.max_missing_frames,
        config.tracking.scene_cut_threshold,
        optical_flow=config.tracking.optical_flow,
        flow_win_size=config.tracking.flow_win_size,
        flow_max_level=config.tracking.flow_max_level,
        flow_max_error=config.tracking.flow_max_error,
    )
    blender = ProfessionalBlender(**config.blend.__dict__)
    restorer = build_face_restorer(config.restorer, models.providers)

    reader = open_video_reader(input_video, config.performance)
    metadata = reader.metadata
    started = time.perf_counter()
    stats = None
    pipeline_settings = None

    console.print(
        "[cyan]Pipeline:[/cyan] "
        f"backend={models.backend}, generador={models.capabilities.generator}, "
        f"3D condicionado={'sí' if models.capabilities.truly_3d_aware else 'no'}, "
        f"postproceso={models.capabilities.geometry_postprocess}, "
        f"decoder={reader.backend}, "
        f"detección cada {config.tracking.detection_interval} frames, "
        f"blend ROI={'sí' if config.blend.roi_enabled else 'no'}"
    )

    with tempfile.TemporaryDirectory(prefix="faceswap_pro_") as temp_dir:
        silent_path = Path(temp_dir) / "video_sin_audio.mp4"
        try:
            writer = RawFFmpegWriter(
                silent_path,
                metadata.width,
                metadata.height,
                metadata.fps,
                config.encoding,
            )
        except Exception:
            reader.close()
            raise
        try:
            stats, pipeline_settings = run_parallel_frames(
                reader=reader,
                writer=writer,
                analyzer=models.analyzer,
                swapper=models.swapper,
                source_face=source_face,
                tracker=tracker,
                blender=blender,
                restorer=restorer,
                config=config,
            )
        finally:
            writer.close()
        mux_original_audio(
            silent_path,
            input_video,
            output_video,
            config.encoding.audio_bitrate,
        )

    elapsed = time.perf_counter() - started
    stats_dict = stats.as_dict() if stats is not None else {}
    effective_fps = stats_dict.get("written_frames", 0) / max(elapsed, 1e-9)
    runtime = {
        "python": platform.python_version(),
        "model_backend": models.backend,
        "model_runtime": dict(models.runtime),
        "model_capabilities": {
            "generator": models.capabilities.generator,
            "native_output_size": models.capabilities.native_output_size,
            "geometry_conditioning": models.capabilities.geometry_conditioning,
            "geometry_postprocess": models.capabilities.geometry_postprocess,
            "temporal_generation": models.capabilities.temporal_generation,
            "truly_3d_aware": models.capabilities.truly_3d_aware,
        },
        "providers_requested": list(models.providers),
        "decoder_backend": reader.backend,
        "encoder_codec": writer.used_codec,
        "fps": metadata.fps,
        "resolution": [metadata.width, metadata.height],
        "frames_reported": metadata.frame_count,
        "elapsed_seconds": round(elapsed, 3),
        "effective_fps": round(effective_fps, 3),
        "pipeline_settings": pipeline_settings,
        "pipeline_stats": stats_dict,
    }
    additional_model_paths = [
        path for path in models.model_artifacts if path.resolve() != model_path.resolve()
    ]

    manifest = write_manifest(
        output_video=output_video,
        manifest_dir=manifest_dir,
        input_video=input_video,
        source_images=source_paths,
        target_reference=target_reference,
        model_path=model_path,
        runtime=runtime,
        additional_model_paths=additional_model_paths,
    )
    console.print(f"[bold green]Video creado:[/bold green] {output_video}")
    console.print(f"[bold green]Manifiesto:[/bold green] {manifest}")
    console.print(
        f"[bold cyan]Rendimiento total:[/bold cyan] {effective_fps:.2f} FPS "
        f"({stats_dict.get('written_frames', 0)} frames)"
    )
    return output_video, manifest
