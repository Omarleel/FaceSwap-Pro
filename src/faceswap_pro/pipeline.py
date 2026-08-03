from __future__ import annotations

import json
import platform
import tempfile
import time
from pathlib import Path

import cv2
from rich.console import Console

from .alignment import align_face
from .blend import ProfessionalBlender
from .engine import initialize_models
from .identity import (
    build_source_identity,
    build_source_identity_and_bank,
    load_reference_embedding,
)
from .modeling import ModelBundle, VideoModelBundle, VideoReference, VideoSwapRequest
from .parallel_pipeline import run_parallel_frames
from .provenance import embed_c2pa_manifest, write_manifest
from .quality_visual import write_visual_comparison_sheet
from .restorer import build_face_restorer
from .tracking import TemporalFaceTracker
from .videoio import (
    RawFFmpegWriter,
    mux_original_audio,
    open_video_reader,
    probe_video,
)

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
    model_bundle: ModelBundle | VideoModelBundle | None = None,
) -> tuple[Path, Path]:
    """Selecciona un pipeline por fotogramas o temporal sin acoplar la CLI al modelo."""

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
    if isinstance(models, VideoModelBundle):
        return _run_video_pipeline(
            input_video=input_video,
            source_dir=source_dir,
            target_reference=target_reference,
            model_path=model_path,
            output_video=output_video,
            manifest_dir=manifest_dir,
            config=config,
            models=models,
        )
    return _run_frame_pipeline(
        input_video=input_video,
        source_dir=source_dir,
        target_reference=target_reference,
        model_path=model_path,
        output_video=output_video,
        manifest_dir=manifest_dir,
        config=config,
        models=models,
    )


def _run_video_pipeline(
    *,
    input_video: Path,
    source_dir: Path,
    target_reference: Path,
    model_path: Path,
    output_video: Path,
    manifest_dir: Path,
    config,
    models: VideoModelBundle,
) -> tuple[Path, Path]:
    started = time.perf_counter()
    console.print(
        "[cyan]Pipeline temporal:[/cyan] "
        f"backend={models.backend}, generador={models.capabilities.generator}, "
        f"generación={models.capabilities.temporal_generation}, "
        f"marca visible={'sí' if config.provenance.visible_disclosure else 'no'}"
    )

    with tempfile.TemporaryDirectory(prefix="faceswap_pro_video_") as temp:
        temp_dir = Path(temp)
        bank_size = int(
            getattr(getattr(config, "engine", None), "options", {}).get(
                "reference_bank_size", 6
            )
        )
        source_face, references, source_paths = _prepare_source_references(
            analyzer=models.analyzer,
            source_dir=source_dir,
            min_score=config.identity.source_min_score,
            limit=config.identity.max_source_images,
            destination_dir=temp_dir / "source_references",
            bank_size=bank_size,
        )
        target_embedding = load_reference_embedding(
            models.analyzer,
            target_reference,
            config.identity.source_min_score,
        )

        silent_video = temp_dir / "dreamidv_silent.mp4"
        result = models.processor.process(
            VideoSwapRequest(
                input_video=input_video,
                source_reference=references[0].path,
                source_references=tuple(references),
                target_embedding=target_embedding,
                source_embedding=source_face.embedding,
                output_video=silent_video,
            )
        )
        if result.output_video != silent_video or not silent_video.is_file():
            raise RuntimeError("El backend temporal no produjo el archivo de salida esperado.")
        mux_original_audio(
            silent_video,
            input_video,
            output_video,
            config.encoding.audio_bitrate,
        )
        visual_report = write_visual_comparison_sheet(
            input_video=input_video,
            output_video=output_video,
            destination=manifest_dir / f"{output_video.stem}.quality-contact-sheet.jpg",
            source_reference=references[0].path,
            target_reference=target_reference,
            samples=6,
        )

    elapsed = time.perf_counter() - started
    output_metadata = probe_video(output_video)
    quality_metrics = dict(result.metadata.get("quality_metrics", {}))
    quality_report = manifest_dir / f"{output_video.stem}.quality.json"
    quality_report.write_text(
        json.dumps(
            {
                "schema": "faceswap-pro-quality-v1",
                "output_video": str(output_video),
                "target_track": result.metadata.get("target_track"),
                "clip_plan": result.metadata.get("clip_plan"),
                "metrics": quality_metrics,
                "visual_comparison": visual_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    runtime = {
        "python": platform.python_version(),
        "model_backend": models.backend,
        "model_runtime": dict(models.runtime),
        "model_capabilities": _capabilities_record(models),
        "providers_requested": list(models.providers),
        "execution_scope": "video",
        "encoder_codec": quality_metrics.get("encoder_codec", "dreamidv_final_encoder"),
        "fps": output_metadata.fps,
        "resolution": [output_metadata.width, output_metadata.height],
        "frames_reported": output_metadata.frame_count,
        "elapsed_seconds": round(elapsed, 3),
        "video_backend": dict(result.metadata),
        "quality_report": str(quality_report),
        "visual_quality_report": visual_report,
        "visible_disclosure": bool(config.provenance.visible_disclosure),
    }
    manifest = _write_pipeline_manifest(
        output_video=output_video,
        manifest_dir=manifest_dir,
        input_video=input_video,
        source_paths=source_paths,
        target_reference=target_reference,
        model_path=model_path,
        models=models,
        runtime=runtime,
        provenance_config=config.provenance,
    )
    console.print(f"[bold green]Video creado:[/bold green] {output_video}")
    console.print(f"[bold green]Manifiesto:[/bold green] {manifest}")
    console.print(f"[bold green]Informe de calidad:[/bold green] {quality_report}")
    console.print(f"[bold cyan]Tiempo total:[/bold cyan] {elapsed:.2f} s")
    return output_video, manifest


def _prepare_source_references(
    *,
    analyzer,
    source_dir: Path,
    min_score: float,
    limit: int,
    destination_dir: Path,
    bank_size: int,
):
    source_face, samples, source_paths = build_source_identity_and_bank(
        analyzer,
        source_dir,
        min_score,
        limit,
        bank_size=bank_size,
    )
    destination_dir.mkdir(parents=True, exist_ok=True)
    samples = sorted(samples, key=lambda sample: sample.weight, reverse=True)
    references: list[VideoReference] = []
    for index, sample in enumerate(samples):
        crop, _ = align_face(
            sample.image,
            sample.face.kps,
            512,
            template="arcface",
            interpolation=cv2.INTER_LANCZOS4,
        )
        destination = destination_dir / f"reference_{index:02d}.png"
        if not cv2.imwrite(str(destination), crop):
            raise RuntimeError(f"No se pudo escribir la referencia DreamID-V: {destination}")
        references.append(
            VideoReference(
                path=destination,
                yaw=sample.yaw,
                pitch=sample.pitch,
                quality=sample.quality,
            )
        )
    if not references:
        raise RuntimeError("No se pudo construir el banco de referencias DreamID-V.")
    return source_face, references, source_paths


def _prepare_source_reference(
    *,
    analyzer,
    source_dir: Path,
    min_score: float,
    limit: int,
    destination: Path,
) -> list[Path]:
    """Compatibilidad con integraciones anteriores: escribe la mejor referencia."""
    source_face, references, source_paths = _prepare_source_references(
        analyzer=analyzer,
        source_dir=source_dir,
        min_score=min_score,
        limit=limit,
        destination_dir=destination.parent,
        bank_size=1,
    )
    del source_face
    if references[0].path != destination:
        destination.write_bytes(references[0].path.read_bytes())
    return source_paths

def _run_frame_pipeline(
    *,
    input_video: Path,
    source_dir: Path,
    target_reference: Path,
    model_path: Path,
    output_video: Path,
    manifest_dir: Path,
    config,
    models: ModelBundle,
) -> tuple[Path, Path]:
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
        f"blend ROI={'sí' if config.blend.roi_enabled else 'no'}, "
        f"marca visible={'sí' if config.provenance.visible_disclosure else 'no'}"
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
        "model_capabilities": _capabilities_record(models),
        "providers_requested": list(models.providers),
        "execution_scope": "frame",
        "decoder_backend": reader.backend,
        "encoder_codec": writer.used_codec,
        "fps": metadata.fps,
        "resolution": [metadata.width, metadata.height],
        "frames_reported": metadata.frame_count,
        "elapsed_seconds": round(elapsed, 3),
        "effective_fps": round(effective_fps, 3),
        "pipeline_settings": pipeline_settings,
        "pipeline_stats": stats_dict,
        "visible_disclosure": bool(config.provenance.visible_disclosure),
    }
    manifest = _write_pipeline_manifest(
        output_video=output_video,
        manifest_dir=manifest_dir,
        input_video=input_video,
        source_paths=source_paths,
        target_reference=target_reference,
        model_path=model_path,
        models=models,
        runtime=runtime,
        provenance_config=config.provenance,
    )
    console.print(f"[bold green]Video creado:[/bold green] {output_video}")
    console.print(f"[bold green]Manifiesto:[/bold green] {manifest}")
    console.print(
        f"[bold cyan]Rendimiento total:[/bold cyan] {effective_fps:.2f} FPS "
        f"({stats_dict.get('written_frames', 0)} frames)"
    )
    return output_video, manifest


def _capabilities_record(models: ModelBundle | VideoModelBundle) -> dict[str, object]:
    return {
        "generator": models.capabilities.generator,
        "native_output_size": models.capabilities.native_output_size,
        "geometry_conditioning": models.capabilities.geometry_conditioning,
        "geometry_postprocess": models.capabilities.geometry_postprocess,
        "temporal_generation": models.capabilities.temporal_generation,
        "truly_3d_aware": models.capabilities.truly_3d_aware,
    }


def _write_pipeline_manifest(
    *,
    output_video: Path,
    manifest_dir: Path,
    input_video: Path,
    source_paths: list[Path],
    target_reference: Path,
    model_path: Path,
    models: ModelBundle | VideoModelBundle,
    runtime: dict,
    provenance_config,
) -> Path:
    additional_model_paths = [
        path for path in models.model_artifacts if path.resolve() != model_path.resolve()
    ]
    c2pa_status = embed_c2pa_manifest(
        output_video=output_video,
        input_video=input_video,
        runtime=runtime,
        manifest_dir=manifest_dir,
        config=provenance_config,
    )
    runtime["c2pa"] = c2pa_status
    return write_manifest(
        output_video=output_video,
        manifest_dir=manifest_dir,
        input_video=input_video,
        source_images=source_paths,
        target_reference=target_reference,
        model_path=model_path,
        runtime=runtime,
        additional_model_paths=additional_model_paths,
        visible_disclosure=bool(provenance_config.visible_disclosure),
        hash_models=bool(getattr(provenance_config, "hash_models", True)),
        c2pa_status=c2pa_status,
        hash_cache_dir=manifest_dir / ".hash-cache",
    )
