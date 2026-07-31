from __future__ import annotations

import platform
import tempfile
import time
from pathlib import Path

import insightface
import onnxruntime as ort
from rich.console import Console

from .blend import ProfessionalBlender
from .engine import initialize_models
from .identity import build_source_identity, load_reference_embedding
from .parallel_pipeline import run_parallel_frames
from .provenance import write_manifest
from .restorer import OptionalFaceRestorer
from .tracking import TemporalFaceTracker
from .videoio import RawFFmpegWriter, mux_original_audio, open_video_reader

console = Console()


def run_pipeline(
    input_video: Path,
    source_dir: Path,
    target_reference: Path,
    swapper_model: Path,
    output_video: Path,
    manifest_dir: Path,
    config,
) -> tuple[Path, Path]:
    for required in (input_video, target_reference, swapper_model):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not source_dir.is_dir():
        raise NotADirectoryError(source_dir)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    face_app, swapper, providers = initialize_models(config.engine, swapper_model)
    source_face, source_paths = build_source_identity(
        face_app,
        source_dir,
        config.identity.source_min_score,
        config.identity.max_source_images,
    )
    target_embedding = load_reference_embedding(
        face_app, target_reference, config.identity.source_min_score
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
    restorer = OptionalFaceRestorer(
        config.restorer.enabled,
        config.restorer.model_path,
        config.restorer.input_size,
        config.restorer.output_range,
        providers,
    )

    reader = open_video_reader(input_video, config.performance)
    metadata = reader.metadata
    started = time.perf_counter()
    stats = None
    pipeline_settings = None

    console.print(
        "[cyan]Pipeline:[/cyan] "
        f"decoder={reader.backend}, detección cada {config.tracking.detection_interval} frames, "
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
                face_app=face_app,
                swapper=swapper,
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
        "insightface": getattr(insightface, "__version__", "unknown"),
        "onnxruntime": ort.__version__,
        "providers_requested": providers,
        "providers_available": ort.get_available_providers(),
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
    manifest = write_manifest(
        output_video=output_video,
        manifest_dir=manifest_dir,
        input_video=input_video,
        source_images=source_paths,
        target_reference=target_reference,
        model_path=swapper_model,
        runtime=runtime,
    )
    console.print(f"[bold green]Video creado:[/bold green] {output_video}")
    console.print(f"[bold green]Manifiesto:[/bold green] {manifest}")
    console.print(
        f"[bold cyan]Rendimiento total:[/bold cyan] {effective_fps:.2f} FPS "
        f"({stats_dict.get('written_frames', 0)} frames)"
    )
    return output_video, manifest
