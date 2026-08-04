from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np
from rich.console import Console

from .insightface_backend import create_insightface_analysis_services
from .modeling import (
    ModelCapabilities,
    VideoModelBundle,
    VideoReference,
    VideoSwapRequest,
    VideoSwapResult,
)
from .observability import log_external, profile_event, profile_span
from .runtime import select_ffmpeg
from .temporal_video import (
    QualityAccumulator,
    TargetAwareCompositor,
    TargetTrack,
    align_generated_to_source,
    analyze_target_track,
)
from .videoio import RawFFmpegWriter, probe_video, probe_video_color

BACKEND_NAME = "dreamid_v"
_SUPPORTED_VARIANTS = {"faster", "dwpose", "mediapipe"}
_SUPPORTED_SIZES = {"832*480", "1280*720"}
_SUPPORTED_SEED_MODES = {"fixed", "absolute_frame"}

console = Console()
CommandRunner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class DreamIDVOptions:
    repository_path: Path
    wan_checkpoint_dir: Path
    python_executable: str
    variant: str
    size: str
    frame_num: int
    sample_fps: int
    sample_steps: int
    sample_shift: float
    sample_guide_scale_img: float
    sample_solver: str
    seed: int
    seed_mode: str
    offload_model: bool
    t5_cpu: bool
    device_id: int
    chunking: bool
    chunk_overlap_frames: int
    scene_aware_chunking: bool
    scene_cut_window_frames: int
    chunk_codec: str
    chunk_crf: int
    chunk_pix_fmt: str
    chunk_container: str
    persistent_worker: bool
    precompute_pose: bool
    precompute_pose_global: bool
    worker_restart_attempts: int
    worker_fallback: bool
    release_analysis_gpu: bool
    profile_worker: bool
    profile_dit_forwards: bool
    profile_worker_cprofile: bool
    profile_worker_cprofile_all: bool
    profile_detailed_clips: int
    worker_cprofile_top: int
    worker_heartbeat_seconds: float
    cache_context: bool
    cache_reference_latents: bool
    reference_latent_cache_size: int
    cuda_cleanup_mode: str
    cuda_cleanup_reserved_ratio: float
    offload_fallback: bool
    staged_offload: bool
    offload_vae_during_dit: bool
    vae_dtype: str
    stream_video_write: bool
    sdpa_backend_priority: tuple[str, ...]
    sdpa_allow_math_fallback: bool
    sdpa_padding_mode: str
    sdpa_diagnostics: bool
    reference_bank_size: int
    target_ambiguity_margin: float
    target_min_coverage: float
    target_max_ambiguous_ratio: float
    mask_expand: float
    mask_feather_ratio: float
    mask_temporal_smoothing: float
    occlusion_strength: float
    color_match: bool
    benchmark_enabled: bool
    benchmark_interval: int
    hdr_policy: str

    @property
    def script_name(self) -> str:
        return {
            "faster": "generate_dreamidv_faster.py",
            "dwpose": "generate_dreamidv_dwpose.py",
            "mediapipe": "generate_dreamidv.py",
        }[self.variant]

    @property
    def package_name(self) -> str:
        return "dreamidv_wan_faster" if self.variant == "faster" else "dreamidv_wan"

    @classmethod
    def from_config(cls, config: Any) -> DreamIDVOptions:
        options = config.engine.options
        repository_path = _required_path(options, "repository_path")
        wan_checkpoint_dir = _required_path(options, "wan_checkpoint_dir")
        variant = str(options.get("variant", "faster")).strip().lower()
        if variant not in _SUPPORTED_VARIANTS:
            raise ValueError(
                f"engine.options.variant debe ser uno de {sorted(_SUPPORTED_VARIANTS)}."
            )

        size = str(options.get("size", "832*480")).strip()
        if size not in _SUPPORTED_SIZES:
            raise ValueError(f"engine.options.size debe ser uno de {sorted(_SUPPORTED_SIZES)}.")

        frame_num = int(options.get("frame_num", 49))
        if frame_num < 5 or frame_num % 4 != 1:
            raise ValueError("engine.options.frame_num debe tener forma 4n+1 y ser >= 5.")

        sample_fps = int(options.get("sample_fps", 16))
        sample_steps = int(options.get("sample_steps", 16 if variant == "faster" else 20))
        if sample_fps <= 0 or sample_steps <= 0:
            raise ValueError("sample_fps y sample_steps deben ser positivos.")

        sample_solver = str(options.get("sample_solver", "unipc")).strip().lower()
        if sample_solver not in {"unipc", "dpm++"}:
            raise ValueError("sample_solver debe ser unipc o dpm++.")

        overlap = int(options.get("chunk_overlap_frames", 17))
        if overlap < 0 or overlap >= frame_num:
            raise ValueError("chunk_overlap_frames debe estar entre 0 y frame_num-1.")
        ambiguity_margin = float(options.get("target_ambiguity_margin", 0.05))
        min_coverage = float(options.get("target_min_coverage", 0.05))
        max_ambiguous = float(options.get("target_max_ambiguous_ratio", 0.15))
        if ambiguity_margin < 0.0:
            raise ValueError("target_ambiguity_margin no puede ser negativo.")
        if not 0.0 <= min_coverage <= 1.0:
            raise ValueError("target_min_coverage debe estar entre 0 y 1.")
        if not 0.0 <= max_ambiguous <= 1.0:
            raise ValueError("target_max_ambiguous_ratio debe estar entre 0 y 1.")
        seed_mode = str(options.get("seed_mode", "absolute_frame")).strip().lower()
        if seed_mode not in _SUPPORTED_SEED_MODES:
            raise ValueError(f"seed_mode debe ser uno de {sorted(_SUPPORTED_SEED_MODES)}.")
        hdr_policy = str(options.get("hdr_policy", "tonemap")).strip().lower()
        if hdr_policy not in {"tonemap", "reject", "passthrough"}:
            raise ValueError("hdr_policy debe ser tonemap, reject o passthrough.")
        cuda_cleanup_mode = str(
            options.get("cuda_cleanup_mode", "adaptive")
        ).strip().lower()
        if cuda_cleanup_mode not in {"adaptive", "always", "never"}:
            raise ValueError("cuda_cleanup_mode debe ser adaptive, always o never.")
        cuda_cleanup_reserved_ratio = float(
            options.get("cuda_cleanup_reserved_ratio", 0.82)
        )
        if not 0.0 < cuda_cleanup_reserved_ratio <= 1.0:
            raise ValueError("cuda_cleanup_reserved_ratio debe estar entre 0 y 1.")
        vae_dtype = str(options.get("vae_dtype", "auto")).strip().lower()
        if vae_dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(
                "vae_dtype debe ser auto, bfloat16, float16 o float32."
            )

        python_executable = _normalize_executable(
            str(options.get("python_executable") or sys.executable)
        )
        return cls(
            repository_path=repository_path,
            wan_checkpoint_dir=wan_checkpoint_dir,
            python_executable=python_executable,
            variant=variant,
            size=size,
            frame_num=frame_num,
            sample_fps=sample_fps,
            sample_steps=sample_steps,
            sample_shift=float(options.get("sample_shift", 5.0)),
            sample_guide_scale_img=float(options.get("sample_guide_scale_img", 4.0)),
            sample_solver=sample_solver,
            seed=int(options.get("seed", 42)),
            seed_mode=seed_mode,
            offload_model=bool(options.get("offload_model", True)),
            t5_cpu=bool(options.get("t5_cpu", True)),
            device_id=int(options.get("device_id", 0)),
            chunking=bool(options.get("chunking", True)),
            chunk_overlap_frames=overlap,
            scene_aware_chunking=bool(options.get("scene_aware_chunking", True)),
            scene_cut_window_frames=max(0, int(options.get("scene_cut_window_frames", 12))),
            chunk_codec=str(options.get("chunk_codec", "libx264")),
            chunk_crf=int(options.get("chunk_crf", 0)),
            chunk_pix_fmt=str(options.get("chunk_pix_fmt", "yuv444p")),
            chunk_container=str(options.get("chunk_container", "mkv")).lstrip("."),
            persistent_worker=bool(options.get("persistent_worker", False)),
            precompute_pose=bool(options.get("precompute_pose", True)),
            precompute_pose_global=bool(options.get("precompute_pose_global", True)),
            worker_restart_attempts=max(0, int(options.get("worker_restart_attempts", 1))),
            worker_fallback=bool(options.get("worker_fallback", True)),
            release_analysis_gpu=bool(options.get("release_analysis_gpu", True)),
            profile_worker=bool(options.get("profile_worker", True)),
            profile_dit_forwards=bool(options.get("profile_dit_forwards", True)),
            profile_worker_cprofile=bool(options.get("profile_worker_cprofile", True)),
            profile_worker_cprofile_all=bool(
                options.get("profile_worker_cprofile_all", False)
            ),
            profile_detailed_clips=max(
                0, int(options.get("profile_detailed_clips", 1))
            ),
            worker_cprofile_top=max(1, int(options.get("worker_cprofile_top", 80))),
            worker_heartbeat_seconds=max(
                0.0, float(options.get("worker_heartbeat_seconds", 15.0))
            ),
            cache_context=bool(options.get("cache_context", True)),
            cache_reference_latents=bool(options.get("cache_reference_latents", True)),
            reference_latent_cache_size=max(
                1, int(options.get("reference_latent_cache_size", 8))
            ),
            cuda_cleanup_mode=cuda_cleanup_mode,
            cuda_cleanup_reserved_ratio=cuda_cleanup_reserved_ratio,
            offload_fallback=bool(options.get("offload_fallback", True)),
            staged_offload=bool(options.get("staged_offload", variant == "faster")),
            offload_vae_during_dit=bool(
                options.get("offload_vae_during_dit", variant == "faster")
            ),
            vae_dtype=vae_dtype,
            stream_video_write=bool(options.get("stream_video_write", True)),
            sdpa_backend_priority=_parse_sdpa_priority(
                options.get("sdpa_backend_priority", ["cudnn", "flash", "efficient", "math"])
            ),
            sdpa_allow_math_fallback=bool(options.get("sdpa_allow_math_fallback", False)),
            sdpa_padding_mode=_parse_sdpa_padding_mode(
                options.get("sdpa_padding_mode", "ragged")
            ),
            sdpa_diagnostics=bool(options.get("sdpa_diagnostics", True)),
            reference_bank_size=max(1, int(options.get("reference_bank_size", 6))),
            target_ambiguity_margin=ambiguity_margin,
            target_min_coverage=min_coverage,
            target_max_ambiguous_ratio=max_ambiguous,
            mask_expand=float(options.get("mask_expand", 1.08)),
            mask_feather_ratio=float(options.get("mask_feather_ratio", 0.055)),
            mask_temporal_smoothing=float(options.get("mask_temporal_smoothing", 0.62)),
            occlusion_strength=float(options.get("occlusion_strength", 0.92)),
            color_match=bool(options.get("color_match", True)),
            benchmark_enabled=bool(options.get("benchmark_enabled", True)),
            benchmark_interval=max(1, int(options.get("benchmark_interval", 8))),
            hdr_policy=hdr_policy,
        )


def _parse_sdpa_priority(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ValueError("sdpa_backend_priority debe ser una lista o cadena separada por comas.")
    aliases = {
        "flash_attention": "flash",
        "cudnn_attention": "cudnn",
        "memory_efficient": "efficient",
        "efficient_attention": "efficient",
    }
    result: list[str] = []
    for item in items:
        name = aliases.get(str(item).strip().lower(), str(item).strip().lower())
        if name not in {"cudnn", "flash", "efficient", "math"}:
            raise ValueError(
                "sdpa_backend_priority solo admite cudnn, flash, efficient y math."
            )
        if name not in result:
            result.append(name)
    if not result:
        raise ValueError("sdpa_backend_priority no puede estar vacío.")
    return tuple(result)


def _parse_sdpa_padding_mode(value: Any) -> str:
    mode = str(value).strip().lower()
    if mode not in {"ragged", "mask"}:
        raise ValueError("sdpa_padding_mode debe ser ragged o mask.")
    return mode


def _required_path(options: Mapping[str, Any], name: str) -> Path:
    value = options.get(name)
    if value in (None, ""):
        raise ValueError(f"Falta engine.options.{name} para el backend DreamID-V.")
    return Path(str(value)).expanduser().resolve()


def _normalize_executable(executable: str) -> str:
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        return str(candidate.resolve())
    return executable


def _python_exists(executable: str) -> bool:
    candidate = Path(executable)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.is_file()
    return shutil.which(executable) is not None


def build_dreamidv_command(
    options: DreamIDVOptions,
    *,
    checkpoint: Path,
    input_video: Path,
    source_reference: Path,
    output_video: Path,
    frame_num: int | None = None,
    seed: int | None = None,
) -> list[str]:
    """Construye la CLI oficial sin acoplar el dominio a PyTorch/Wan."""

    command = [
        options.python_executable,
        str((options.repository_path / options.script_name).resolve()),
        "--task",
        "swapface",
        "--size",
        options.size,
        "--frame_num",
        str(frame_num or options.frame_num),
        "--sample_fps",
        str(options.sample_fps),
        "--ckpt_dir",
        str(options.wan_checkpoint_dir.resolve()),
        "--dreamidv_ckpt",
        str(checkpoint.expanduser().resolve()),
        "--offload_model",
        "true" if options.offload_model else "false",
        "--save_file",
        str(output_video.expanduser().resolve()),
        "--base_seed",
        str(options.seed if seed is None else seed),
        "--ref_image",
        str(source_reference.expanduser().resolve()),
        "--ref_video",
        str(input_video.expanduser().resolve()),
        "--sample_solver",
        options.sample_solver,
        "--sample_steps",
        str(options.sample_steps),
        "--sample_shift",
        str(options.sample_shift),
        "--sample_guide_scale_img",
        str(options.sample_guide_scale_img),
    ]
    if options.t5_cpu:
        command.append("--t5_cpu")
    return command


@dataclass(frozen=True)
class DreamIDVClip:
    index: int
    start_frame: int
    valid_frames: int
    overlap_before: int
    hard_cut_before: bool
    seed: int

    @property
    def end_frame(self) -> int:
        return self.start_frame + self.valid_frames


@dataclass(frozen=True)
class DreamIDVPreparedClip:
    """Artefactos preparados antes de cargar el modelo Wan en la GPU."""

    clip: DreamIDVClip
    source_clip: Path
    generated_clip: Path
    source_reference: Path
    pose_video: Path | None = None
    mask_video: Path | None = None


class DreamIDVClipPlanner:
    """Planifica ventanas 4n+1 con solapamiento y cortes de escena."""

    def __init__(self, options: DreamIDVOptions) -> None:
        self.options = options

    def _duration_and_frames(self, input_video: Path) -> tuple[float, int]:
        metadata = probe_video(input_video)
        duration = metadata.frame_count / metadata.fps if metadata.frame_count > 0 else 0.0
        if duration <= 0:
            raise ValueError("No se pudo determinar la duraciÃ³n del vÃ­deo de entrada.")
        requested_frames = max(1, int(math.ceil(duration * self.options.sample_fps)))
        return duration, requested_frames

    def plan(self, input_video: Path) -> tuple[int, float, int]:
        duration, requested_frames = self._duration_and_frames(input_video)
        clips = self.plan_frames(requested_frames, ())
        return len(clips), duration, requested_frames

    def plan_frames(
        self,
        requested_frames: int,
        scene_cuts: Sequence[int],
    ) -> list[DreamIDVClip]:
        if requested_frames <= 0:
            raise ValueError("requested_frames debe ser positivo.")
        if not self.options.chunking and requested_frames > self.options.frame_num:
            raise ValueError(
                "El vÃ­deo excede frame_num y chunking estÃ¡ desactivado; "
                "activa chunking para no truncar la salida."
            )
        if requested_frames <= self.options.frame_num or not self.options.chunking:
            return [
                DreamIDVClip(
                    index=0,
                    start_frame=0,
                    valid_frames=requested_frames,
                    overlap_before=0,
                    hard_cut_before=False,
                    seed=self._seed_for_start(0),
                )
            ]

        stride = self.options.frame_num - self.options.chunk_overlap_frames
        starts: list[tuple[int, bool]] = [(0, False)]
        cuts = sorted(set(int(value) for value in scene_cuts if 0 < value < requested_frames))
        while starts[-1][0] + self.options.frame_num < requested_frames:
            previous = starts[-1][0]
            nominal = previous + stride
            next_start = nominal
            hard_cut = False
            if self.options.scene_aware_chunking and cuts:
                lower = max(previous + 1, nominal - self.options.scene_cut_window_frames)
                upper = min(
                    requested_frames - 1,
                    previous + self.options.frame_num - 1,
                    nominal + self.options.scene_cut_window_frames,
                )
                candidates = [cut for cut in cuts if lower <= cut <= upper]
                if candidates:
                    next_start = min(candidates, key=lambda cut: abs(cut - nominal))
                    hard_cut = True
            if next_start <= previous:
                next_start = previous + stride
                hard_cut = False
            starts.append((next_start, hard_cut))

        clips: list[DreamIDVClip] = []
        previous_end = 0
        for index, (start, hard_cut) in enumerate(starts):
            valid = min(self.options.frame_num, requested_frames - start)
            overlap = 0 if index == 0 else max(0, previous_end - start)
            clips.append(
                DreamIDVClip(
                    index=index,
                    start_frame=start,
                    valid_frames=valid,
                    overlap_before=overlap,
                    hard_cut_before=hard_cut,
                    seed=self._seed_for_start(start),
                )
            )
            previous_end = start + valid
        return clips

    def _seed_for_start(self, start: int) -> int:
        if self.options.seed_mode == "fixed":
            return self.options.seed
        return int((self.options.seed + start) % (2**31 - 1))


class DreamIDVPoseClient:
    """Precalcula DWPose en un proceso que se cierra antes de cargar Wan.

    DWPose y DreamID-V nunca comparten VRAM. El proceso mantiene las sesiones
    ONNX entre clips para no recargar los modelos, pero se termina por completo
    al finalizar la fase de preprocesamiento.
    """

    PREFIX = "FACESWAP_RESULT "

    def __init__(self, options: DreamIDVOptions) -> None:
        worker = Path(__file__).with_name("dreamidv_pose_worker.py").resolve()
        command = [
            options.python_executable,
            str(worker),
            "--repository",
            str(options.repository_path),
            "--device-id",
            str(options.device_id),
        ]
        self.process = subprocess.Popen(
            command,
            cwd=options.repository_path,
            env=_dreamidv_environment(options),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

    def generate(self, *, input_video: Path, pose_video: Path, mask_video: Path) -> None:
        self._request(
            {
                "input_video": str(input_video.resolve()),
                "pose_video": str(pose_video.resolve()),
                "mask_video": str(mask_video.resolve()),
            }
        )

    def _request(self, payload: Mapping[str, Any]) -> None:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("El worker DWPose no tiene pipes disponibles.")
        self.process.stdin.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        while True:
            line = self.process.stdout.readline()
            if not line:
                code = self.process.poll()
                raise RuntimeError(f"El worker DWPose terminó inesperadamente ({code}).")
            if line.startswith(self.PREFIX):
                response = json.loads(line[len(self.PREFIX) :])
                if not response.get("ok"):
                    raise RuntimeError(str(response.get("error", "Error desconocido de DWPose")))
                return
            console.print(f"[dim]{line.rstrip()}[/dim]")

    def close(self) -> None:
        _close_jsonl_process(self.process)

    def __enter__(self) -> DreamIDVPoseClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _ingest_worker_profile(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    event = str(payload.pop("event", "dreamidv.worker_event"))
    profile_event(event, **payload)


def _ingest_worker_log(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    level = str(payload.pop("level", "WARNING"))
    message = str(payload.pop("message", "Mensaje del worker DreamID-V"))
    source = str(payload.pop("source", "dreamidv_worker"))
    log_external(level, message, source=source, **payload)


class DreamIDVPersistentClient:
    """Cliente JSONL que mantiene DreamID-V cargado durante todos los clips.

    Además de esperar la respuesta, consume telemetría y heartbeats del worker y
    los incorpora a la sesión de observabilidad del proceso principal.
    """

    RESULT_PREFIX = "FACESWAP_RESULT "
    PROFILE_PREFIX = "FACESWAP_PROFILE "
    LOG_PREFIX = "FACESWAP_LOG "

    def __init__(self, options: DreamIDVOptions, checkpoint: Path) -> None:
        worker = Path(__file__).with_name("dreamidv_worker.py").resolve()
        command = [
            options.python_executable,
            str(worker),
            "--repository",
            str(options.repository_path),
            "--variant",
            options.variant,
            "--size",
            options.size,
            "--sample-fps",
            str(options.sample_fps),
            "--checkpoint-dir",
            str(options.wan_checkpoint_dir),
            "--dreamidv-checkpoint",
            str(checkpoint),
            "--device-id",
            str(options.device_id),
            "--reference-latent-cache-size",
            str(options.reference_latent_cache_size),
            "--cprofile-top",
            str(options.worker_cprofile_top),
            "--cuda-cleanup-mode",
            options.cuda_cleanup_mode,
            "--cuda-cleanup-reserved-ratio",
            str(options.cuda_cleanup_reserved_ratio),
            "--heartbeat-seconds",
            str(options.worker_heartbeat_seconds),
            "--profile-detailed-clips",
            str(options.profile_detailed_clips),
            "--vae-dtype",
            options.vae_dtype,
        ]
        if options.t5_cpu:
            command.append("--t5-cpu")
        if options.profile_worker:
            command.append("--profile-worker")
        if options.profile_worker and options.profile_dit_forwards:
            command.append("--profile-dit-forwards")
        if options.profile_worker and options.profile_worker_cprofile:
            command.append("--profile-cprofile")
        if options.profile_worker and options.profile_worker_cprofile_all:
            command.append("--profile-cprofile-all")
        if options.cache_context:
            command.append("--cache-context")
        if options.cache_reference_latents:
            command.append("--cache-reference-latents")
        if options.offload_fallback:
            command.append("--offload-fallback")
        if options.staged_offload:
            command.append("--staged-offload")
        if options.offload_vae_during_dit:
            command.append("--offload-vae-during-dit")
        if options.stream_video_write:
            command.append("--stream-video-write")
        with profile_span(
            "dreamidv.worker.spawn",
            variant=options.variant,
            profile_worker=options.profile_worker,
        ):
            self.process = subprocess.Popen(
                command,
                cwd=options.repository_path,
                env=_dreamidv_environment(options),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

    def _read_response(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise RuntimeError("El worker DreamID-V no tiene stdout disponible.")
        while True:
            line = self.process.stdout.readline()
            if not line:
                code = self.process.poll()
                raise RuntimeError(
                    f"El worker DreamID-V terminó inesperadamente ({code})."
                )
            stripped = line.rstrip("\r\n")
            if stripped.startswith(self.RESULT_PREFIX):
                response = json.loads(stripped[len(self.RESULT_PREFIX) :])
                if not response.get("ok"):
                    raise RuntimeError(
                        str(response.get("error", "Error desconocido del worker"))
                    )
                return response
            if stripped.startswith(self.PROFILE_PREFIX):
                try:
                    _ingest_worker_profile(
                        json.loads(stripped[len(self.PROFILE_PREFIX) :])
                    )
                except Exception as exc:  # noqa: BLE001 - telemetría no bloqueante
                    log_external(
                        "WARNING",
                        "No se pudo incorporar una métrica del worker DreamID-V",
                        source="dreamidv_parent",
                        error=str(exc),
                        raw_line=stripped[:2000],
                    )
                continue
            if stripped.startswith(self.LOG_PREFIX):
                try:
                    _ingest_worker_log(json.loads(stripped[len(self.LOG_PREFIX) :]))
                except Exception as exc:  # noqa: BLE001 - telemetría no bloqueante
                    log_external(
                        "WARNING",
                        "No se pudo incorporar un log del worker DreamID-V",
                        source="dreamidv_parent",
                        error=str(exc),
                        raw_line=stripped[:2000],
                    )
                continue
            console.print(f"[dim]{stripped}[/dim]")

    def generate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("El worker DreamID-V no tiene pipes disponibles.")
        request = dict(payload)
        request.setdefault("request_id", uuid.uuid4().hex[:16])
        with profile_span(
            "dreamidv.worker.roundtrip",
            worker_request_id=request["request_id"],
            clip_index=request.get("clip_index"),
            clip_start_frame=request.get("clip_start_frame"),
            sample_steps=request.get("sample_steps"),
        ):
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            return self._read_response()

    def close(self) -> None:
        # No espera una respuesta JSONL: si el usuario interrumpe un clip largo,
        # _close_jsonl_process aplica timeouts y termina el worker en vez de quedar
        # bloqueado hasta que finalice la difusión actual.
        _close_jsonl_process(self.process)

    def __enter__(self) -> DreamIDVPersistentClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

def _close_jsonl_process(process: subprocess.Popen[str]) -> None:
    if process.stdin is not None and process.poll() is None:
        try:
            process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
    if process.poll() is None:
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def _dreamidv_environment(options: DreamIDVOptions) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(options.device_id)
    # expandable_segments no está soportado por varias builds de PyTorch/Windows.
    # max_split_size y garbage_collection_threshold reducen fragmentación sin ese aviso.
    allocator = "max_split_size_mb:128,garbage_collection_threshold:0.80"
    if os.name != "nt":
        allocator = "expandable_segments:True," + allocator
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", allocator)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["FACESWAP_SDPA_BACKENDS"] = ",".join(options.sdpa_backend_priority)
    env["FACESWAP_SDPA_ALLOW_MATH"] = (
        "1" if options.sdpa_allow_math_fallback else "0"
    )
    env["FACESWAP_SDPA_PADDING_MODE"] = options.sdpa_padding_mode
    env["FACESWAP_SDPA_DIAGNOSTICS"] = "1" if options.sdpa_diagnostics else "0"
    return env


class DreamIDVSubprocessBackend:
    """Backend temporal con ventanas solapadas, tracking y recomposiciÃ³n selectiva."""

    def __init__(
        self,
        checkpoint: Path,
        options: DreamIDVOptions,
        *,
        analyzer: Any | None = None,
        identity_config: Any | None = None,
        tracking_config: Any | None = None,
        encoding_config: Any | None = None,
        runner: CommandRunner = subprocess.run,
    ) -> None:
        self.checkpoint = checkpoint
        self.options = options
        self.analyzer = analyzer
        self.identity_config = identity_config
        self.tracking_config = tracking_config
        self.encoding_config = encoding_config
        self.runner = runner
        self.planner = DreamIDVClipPlanner(options)

    def process(self, request: VideoSwapRequest) -> VideoSwapResult:
        request.output_video.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = select_ffmpeg()
        if ffmpeg is None:
            raise RuntimeError("FFmpeg es obligatorio para preparar clips DreamID-V.")
        input_meta = probe_video(request.input_video)
        duration = input_meta.frame_count / input_meta.fps
        requested_frames = max(1, int(math.ceil(duration * self.options.sample_fps)))
        color = probe_video_color(request.input_video)
        if color.hdr and self.options.hdr_policy == "reject":
            raise ValueError(
                "El vÃ­deo es HDR y hdr_policy=reject. Usa tonemap para una conversiÃ³n explÃ­cita."
            )

        with tempfile.TemporaryDirectory(prefix="faceswap_dreamidv_") as temp:
            temp_dir = Path(temp)
            proxy = temp_dir / "source_proxy.mkv"
            with profile_span(
                "dreamidv.proxy.prepare",
                requested_frames=requested_frames,
                sample_fps=self.options.sample_fps,
                hdr=bool(color.hdr),
            ):
                self._prepare_source_proxy(
                    ffmpeg=ffmpeg,
                    source=request.input_video,
                    destination=proxy,
                    requested_frames=requested_frames,
                    hdr=bool(color.hdr),
                )
            proxy_meta = probe_video(proxy)

            track: TargetTrack | None = None
            if (
                request.target_embedding is not None
                and self.analyzer is not None
                and self.tracking_config is not None
            ):
                min_similarity = float(
                    getattr(self.identity_config, "target_min_similarity", 0.30)
                )
                with profile_span(
                    "dreamidv.target_track",
                    requested_frames=requested_frames,
                ):
                    track = analyze_target_track(
                        proxy,
                        self.analyzer,
                        request.target_embedding,
                        self.tracking_config,
                        fps=self.options.sample_fps,
                        min_similarity=min_similarity,
                        ambiguity_margin=self.options.target_ambiguity_margin,
                    )
                if track.coverage < self.options.target_min_coverage:
                    raise RuntimeError(
                        "La identidad objetivo no pudo seguirse de forma fiable: "
                        f"cobertura={track.coverage:.1%}."
                    )
                if track.ambiguous_ratio > self.options.target_max_ambiguous_ratio:
                    raise RuntimeError(
                        "Demasiados frames ambiguos entre varias personas: "
                        f"ratio={track.ambiguous_ratio:.1%}."
                    )

            analysis_gpu_released = False
            if self.options.release_analysis_gpu and self.analyzer is not None:
                release = getattr(self.analyzer, "release_gpu_resources", None)
                if callable(release):
                    release()
                    self.analyzer = None
                    analysis_gpu_released = True
                    console.print(
                        "[green]InsightFace liberado de la GPU antes de DWPose/Wan.[/green]"
                    )

            clips = self.planner.plan_frames(
                requested_frames,
                track.scene_cuts if track is not None else (),
            )
            total_valid_window_frames = sum(clip.valid_frames for clip in clips)
            total_generated_frames = len(clips) * self.options.frame_num
            profile_event(
                "dreamidv.clip_plan",
                requested_frames=requested_frames,
                clip_count=len(clips),
                frame_num=self.options.frame_num,
                overlap_frames=self.options.chunk_overlap_frames,
                total_generated_frames=total_generated_frames,
                total_valid_window_frames=total_valid_window_frames,
                duplicated_overlap_frames=max(
                    0, total_valid_window_frames - requested_frames
                ),
                padded_window_frames=max(
                    0, total_generated_frames - total_valid_window_frames
                ),
            )
            references = request.source_references or (
                VideoReference(request.source_reference),
            )
            generated: list[tuple[DreamIDVClip, Path, Path]] = []
            prepared: list[DreamIDVPreparedClip] = []
            use_persistent_worker = (
                self.options.persistent_worker
                and self.options.variant in {"faster", "dwpose"}
            )

            # Fase 1: DWPose se ejecuta una sola vez sobre el proxy completo.
            # Después se recortan pose y máscara con los mismos índices de cada
            # clip, evitando recalcular los fotogramas incluidos en solapes.
            pose_worker: DreamIDVPoseClient | None = None
            if use_persistent_worker and self.options.precompute_pose:
                mode = "global" if self.options.precompute_pose_global else "por clip"
                console.print(
                    "[cyan]DreamID-V:[/cyan] precalculando pose y máscaras "
                    f"({mode}) antes de cargar Wan."
                )
                pose_worker = DreamIDVPoseClient(self.options)

            try:
                global_pose: Path | None = None
                global_mask: Path | None = None
                if pose_worker is not None and self.options.precompute_pose_global:
                    global_pose_dir = temp_dir / "pose_global"
                    global_pose = global_pose_dir / "pose.mp4"
                    global_mask = global_pose_dir / "mask.mp4"
                    with profile_span(
                        "dreamidv.dwpose.global",
                        requested_frames=requested_frames,
                        clip_count=len(clips),
                    ):
                        pose_worker.generate(
                            input_video=proxy,
                            pose_video=global_pose,
                            mask_video=global_mask,
                        )

                for clip in clips:
                    source_clip = temp_dir / (
                        f"source_{clip.index:04d}.{self.options.chunk_container}"
                    )
                    generated_clip = temp_dir / f"generated_{clip.index:04d}.mp4"
                    with profile_span(
                        "dreamidv.clip.extract_source",
                        clip_index=clip.index,
                        start_frame=clip.start_frame,
                    ):
                        self._extract_chunk(
                            ffmpeg=ffmpeg,
                            source=proxy,
                            destination=source_clip,
                            start_frame=clip.start_frame,
                        )
                    reference = self._select_reference(references, track, clip)
                    pose_video: Path | None = None
                    mask_video: Path | None = None
                    if pose_worker is not None:
                        pose_dir = temp_dir / f"pose_{clip.index:04d}"
                        pose_dir.mkdir(parents=True, exist_ok=True)
                        pose_video = pose_dir / (
                            f"pose.{self.options.chunk_container}"
                        )
                        mask_video = pose_dir / (
                            f"mask.{self.options.chunk_container}"
                        )
                        if global_pose is not None and global_mask is not None:
                            with profile_span(
                                "dreamidv.dwpose.extract_clip",
                                clip_index=clip.index,
                                start_frame=clip.start_frame,
                            ):
                                self._extract_chunk(
                                    ffmpeg=ffmpeg,
                                    source=global_pose,
                                    destination=pose_video,
                                    start_frame=clip.start_frame,
                                )
                                self._extract_chunk(
                                    ffmpeg=ffmpeg,
                                    source=global_mask,
                                    destination=mask_video,
                                    start_frame=clip.start_frame,
                                )
                        else:
                            console.print(
                                f"[cyan]DWPose:[/cyan] clip {clip.index + 1}/{len(clips)}"
                            )
                            with profile_span(
                                "dreamidv.dwpose.clip",
                                clip_index=clip.index,
                            ):
                                pose_worker.generate(
                                    input_video=source_clip,
                                    pose_video=pose_video,
                                    mask_video=mask_video,
                                )
                    prepared.append(
                        DreamIDVPreparedClip(
                            clip=clip,
                            source_clip=source_clip,
                            generated_clip=generated_clip,
                            source_reference=reference.path,
                            pose_video=pose_video,
                            mask_video=mask_video,
                        )
                    )
            finally:
                if pose_worker is not None:
                    with profile_span("dreamidv.dwpose.worker_close"):
                        pose_worker.close()
                    console.print(
                        "[green]DWPose completado; proceso cerrado y VRAM "
                        "liberada antes de Wan.[/green]"
                    )

            worker: DreamIDVPersistentClient | None = None
            worker_failed = False
            worker_restarts = 0
            worker_clip_reports: list[dict[str, Any]] = []
            if use_persistent_worker:
                if not self.options.precompute_pose:
                    raise RuntimeError(
                        "persistent_worker requiere precompute_pose=true para evitar "
                        "que DWPose y Wan compartan VRAM."
                    )
                with profile_span("dreamidv.worker.client_create"):
                    worker = DreamIDVPersistentClient(self.options, self.checkpoint)

            try:
                for item in prepared:
                    clip = item.clip
                    console.print(
                        f"[cyan]DreamID-V:[/cyan] clip {clip.index + 1}/{len(clips)}, "
                        f"inicio={clip.start_frame}, solape={clip.overlap_before}, "
                        f"ref={item.source_reference.name}"
                    )
                    payload = {
                        "clip_index": clip.index,
                        "clip_start_frame": clip.start_frame,
                        "clip_valid_frames": clip.valid_frames,
                        "clip_overlap_before": clip.overlap_before,
                        "input_video": str(item.source_clip.resolve()),
                        "source_reference": str(item.source_reference.resolve()),
                        "output_video": str(item.generated_clip.resolve()),
                        "frame_num": self.options.frame_num,
                        "seed": clip.seed,
                        "sample_steps": self.options.sample_steps,
                        "sample_shift": self.options.sample_shift,
                        "sample_solver": self.options.sample_solver,
                        "guide_scale": self.options.sample_guide_scale_img,
                        "offload_model": self.options.offload_model,
                    }
                    if item.pose_video is not None and item.mask_video is not None:
                        payload.update(
                            {
                                "pose_video": str(item.pose_video.resolve()),
                                "mask_video": str(item.mask_video.resolve()),
                            }
                        )

                    generated_by_worker = False
                    clip_restart_attempts = 0
                    while worker is not None and not generated_by_worker:
                        try:
                            response = worker.generate(payload)
                            if isinstance(response, Mapping):
                                metrics = response.get("metrics")
                                if isinstance(metrics, Mapping):
                                    worker_clip_reports.append(dict(metrics))
                            generated_by_worker = True
                        except Exception as exc:  # noqa: BLE001
                            worker.close()
                            worker = None
                            if clip_restart_attempts < self.options.worker_restart_attempts:
                                clip_restart_attempts += 1
                                worker_restarts += 1
                                console.print(
                                    "[yellow]Worker DreamID-V falló; se libera su proceso "
                                    f"y se reinicia ({clip_restart_attempts}/"
                                    f"{self.options.worker_restart_attempts} para este clip):"
                                    f"[/yellow] {exc}"
                                )
                                with profile_span(
                                    "dreamidv.worker.restart",
                                    clip_index=clip.index,
                                    restart_attempt=clip_restart_attempts,
                                ):
                                    worker = DreamIDVPersistentClient(
                                        self.options, self.checkpoint
                                    )
                                continue
                            if not self.options.worker_fallback:
                                raise
                            worker_failed = True
                            console.print(
                                "[yellow]Worker agotó los reinicios; se usa la CLI "
                                f"solo para este clip:[/yellow] {exc}"
                            )

                    if not generated_by_worker:
                        command = build_dreamidv_command(
                            self.options,
                            checkpoint=self.checkpoint,
                            input_video=item.source_clip,
                            source_reference=item.source_reference,
                            output_video=item.generated_clip,
                            seed=clip.seed,
                        )
                        with profile_span(
                            "dreamidv.clip.cli_fallback",
                            clip_index=clip.index,
                        ):
                            self.runner(
                                command,
                                cwd=self.options.repository_path,
                                env=_dreamidv_environment(self.options),
                                check=True,
                            )
                    if not item.generated_clip.is_file():
                        raise RuntimeError(
                            "DreamID-V terminó sin crear el vídeo esperado: "
                            f"{item.generated_clip}"
                        )
                    generated.append((clip, item.generated_clip, item.source_clip))
            finally:
                if worker is not None:
                    worker.close()

            with profile_span(
                "dreamidv.stitch_compose_encode",
                clip_count=len(generated),
                requested_frames=requested_frames,
            ):
                quality = self._stitch_compose_encode(
                    generated=generated,
                    track=track,
                    output=request.output_video,
                    width=proxy_meta.width,
                    height=proxy_meta.height,
                    request=request,
                    input_color=color,
                )

        return VideoSwapResult(
            output_video=request.output_video,
            metadata={
                "variant": self.options.variant,
                "size": self.options.size,
                "frame_num": self.options.frame_num,
                "sample_fps": self.options.sample_fps,
                "sample_steps": self.options.sample_steps,
                "chunks": len(clips),
                "chunk_overlap_frames": self.options.chunk_overlap_frames,
                "scene_aware_chunking": self.options.scene_aware_chunking,
                "requested_frames": requested_frames,
                "offload_model": self.options.offload_model,
                "t5_cpu": self.options.t5_cpu,
                "persistent_worker_requested": self.options.persistent_worker,
                "precompute_pose": self.options.precompute_pose,
                "precompute_pose_global": self.options.precompute_pose_global,
                "cache_context": self.options.cache_context,
                "cache_reference_latents": self.options.cache_reference_latents,
                "cuda_cleanup_mode": self.options.cuda_cleanup_mode,
                "offload_fallback": self.options.offload_fallback,
                "staged_offload": self.options.staged_offload,
                "offload_vae_during_dit": self.options.offload_vae_during_dit,
                "vae_dtype": self.options.vae_dtype,
                "stream_video_write": self.options.stream_video_write,
                "worker_clip_reports": worker_clip_reports,
                "worker_restart_attempts": self.options.worker_restart_attempts,
                "worker_restarts": worker_restarts,
                "persistent_worker_fallback": worker_failed,
                "analysis_gpu_released": analysis_gpu_released,
                "target_track": track.as_dict() if track is not None else None,
                "quality_metrics": quality,
                "clip_plan": [
                    {
                        "index": clip.index,
                        "start_frame": clip.start_frame,
                        "valid_frames": clip.valid_frames,
                        "overlap_before": clip.overlap_before,
                        "hard_cut_before": clip.hard_cut_before,
                        "seed": clip.seed,
                    }
                    for clip in clips
                ],
                "input_color": {
                    "pixel_format": color.pixel_format,
                    "primaries": color.color_primaries,
                    "transfer": color.color_transfer,
                    "space": color.color_space,
                    "range": color.color_range,
                    "hdr": color.hdr,
                    "hdr_policy": self.options.hdr_policy,
                },
            },
        )

    def _prepare_source_proxy(
        self,
        *,
        ffmpeg: Path,
        source: Path,
        destination: Path,
        requested_frames: int,
        hdr: bool,
    ) -> None:
        filters = [f"fps={self.options.sample_fps}"]
        if hdr and self.options.hdr_policy == "tonemap":
            filters.extend(
                [
                    "zscale=t=linear:npl=100",
                    "format=gbrpf32le",
                    "tonemap=tonemap=hable:desat=0",
                    "zscale=p=bt709:t=bt709:m=bt709:r=tv",
                    "format=yuv444p",
                ]
            )
        filters.append("tpad=stop_mode=clone:stop_duration=2")
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            ",".join(filters),
            "-frames:v",
            str(requested_frames),
            "-c:v",
            self.options.chunk_codec,
        ]
        command += self._lossless_codec_arguments()
        command += ["-pix_fmt", self.options.chunk_pix_fmt, str(destination)]
        try:
            self.runner(command, check=True)
        except subprocess.CalledProcessError:
            if not (hdr and self.options.hdr_policy == "tonemap"):
                raise
            console.print(
                "[yellow]El FFmpeg no soportÃ³ tonemap/zscale; se usa "
                "conversiÃ³n SDR bÃ¡sica.[/yellow]"
            )
            fallback = command.copy()
            filter_index = fallback.index("-vf") + 1
            fallback[filter_index] = (
                f"fps={self.options.sample_fps},format=yuv444p,"
                "tpad=stop_mode=clone:stop_duration=2"
            )
            self.runner(fallback, check=True)

    def _lossless_codec_arguments(self) -> list[str]:
        codec = self.options.chunk_codec.lower()
        if codec in {"libx264", "libx264rgb", "libx265"}:
            return ["-preset", "veryfast", "-crf", str(self.options.chunk_crf)]
        if codec == "ffv1":
            return ["-level", "3", "-coder", "1", "-context", "1"]
        return []

    def _extract_chunk(
        self,
        *,
        ffmpeg: Path,
        source: Path,
        destination: Path,
        start_frame: int,
    ) -> None:
        start = start_frame / self.options.sample_fps
        command = [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-ss",
            f"{start:.8f}",
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "tpad=stop_mode=clone:stop_duration=4",
            "-frames:v",
            str(self.options.frame_num),
            "-c:v",
            self.options.chunk_codec,
        ]
        command += self._lossless_codec_arguments()
        command += ["-pix_fmt", self.options.chunk_pix_fmt, str(destination)]
        self.runner(command, check=True)

    @staticmethod
    def _read_frames(path: Path, expected: int) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo leer el clip generado: {path}")
        frames: list[np.ndarray] = []
        try:
            while len(frames) < expected:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()
        if not frames:
            raise RuntimeError(f"El clip no contiene fotogramas: {path}")
        while len(frames) < expected:
            frames.append(frames[-1].copy())
        return frames[:expected]

    def _select_reference(
        self,
        references: Sequence[VideoReference],
        track: TargetTrack | None,
        clip: DreamIDVClip,
    ) -> VideoReference:
        if len(references) == 1 or track is None:
            return references[0]
        yaw, pitch = track.pose_for_range(clip.start_frame, clip.end_frame)
        return min(
            references,
            key=lambda ref: (
                ((ref.yaw - yaw) / 35.0) ** 2
                + ((ref.pitch - pitch) / 25.0) ** 2
                - 0.18 * float(ref.quality)
            ),
        )

    def _resolved_encoding(self, input_color) -> Any:
        encoding = self.encoding_config
        if encoding is None:
            return SimpleNamespace(
                codec="libx264",
                preset="slow",
                cq=16,
                fallback_codec="libx264",
                fallback_preset="slow",
                fallback_crf=16,
                pixel_format="yuv420p",
                color_primaries="bt709",
                color_transfer="bt709",
                color_space="bt709",
                color_range="tv",
            )
        values = {
            "color_primaries": getattr(encoding, "color_primaries", None),
            "color_transfer": getattr(encoding, "color_transfer", None),
            "color_space": getattr(encoding, "color_space", None),
            "color_range": getattr(encoding, "color_range", None),
        }
        if input_color.hdr and self.options.hdr_policy == "tonemap":
            values.update(
                {
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                    "color_space": "bt709",
                    "color_range": "tv",
                }
            )
        else:
            values = {
                "color_primaries": values["color_primaries"] or input_color.color_primaries,
                "color_transfer": values["color_transfer"] or input_color.color_transfer,
                "color_space": values["color_space"] or input_color.color_space,
                "color_range": values["color_range"] or input_color.color_range,
            }
        try:
            return replace(encoding, **values)
        except TypeError:
            payload = dict(vars(encoding))
            payload.update(values)
            return SimpleNamespace(**payload)

    def _stitch_compose_encode(
        self,
        *,
        generated: Sequence[tuple[DreamIDVClip, Path, Path]],
        track: TargetTrack | None,
        output: Path,
        width: int,
        height: int,
        request: VideoSwapRequest,
        input_color: Any,
    ) -> dict[str, Any]:
        if not generated:
            raise RuntimeError("DreamID-V no produjo clips para unir.")
        encoding = self._resolved_encoding(input_color)
        writer = RawFFmpegWriter(
            output,
            width,
            height,
            float(self.options.sample_fps),
            encoding,
        )
        compositor = TargetAwareCompositor(
            mask_expand=self.options.mask_expand,
            feather_ratio=self.options.mask_feather_ratio,
            temporal_smoothing=self.options.mask_temporal_smoothing,
            occlusion_strength=self.options.occlusion_strength,
            color_match=self.options.color_match,
        )
        quality = QualityAccumulator(
            analyzer=self.analyzer if self.options.benchmark_enabled else None,
            source_embedding=request.source_embedding,
            sample_interval=self.options.benchmark_interval,
        )
        pending: list[tuple[int, np.ndarray, np.ndarray, bool]] = []

        def emit(item: tuple[int, np.ndarray, np.ndarray, bool]) -> None:
            index, generated_frame, source_frame, boundary = item
            if generated_frame.shape[:2] != (height, width):
                generated_frame = cv2.resize(
                    generated_frame,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            if source_frame.shape[:2] != (height, width):
                source_frame = cv2.resize(
                    source_frame,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            target_frame = track.frame(index) if track is not None else None
            if track is None:
                output_frame = generated_frame
                alpha = np.ones((height, width), dtype=np.float32)
            else:
                output_frame, alpha, _ = compositor.compose(
                    source_frame,
                    generated_frame,
                    target_frame,
                )
            quality.observe(
                index,
                source_frame,
                output_frame,
                alpha,
                target_frame,
                chunk_boundary=boundary,
            )
            writer.write(output_frame)

        try:
            for clip, generated_path, source_path in generated:
                gen_frames = self._read_frames(generated_path, self.options.frame_num)[
                    : clip.valid_frames
                ]
                src_frames = self._read_frames(source_path, self.options.frame_num)[
                    : clip.valid_frames
                ]
                incoming = [
                    (
                        clip.start_frame + offset,
                        gen_frames[offset],
                        src_frames[offset],
                        False,
                    )
                    for offset in range(clip.valid_frames)
                ]
                if not pending:
                    pending = incoming
                    continue

                overlap = min(clip.overlap_before, len(pending), len(incoming))
                if clip.hard_cut_before:
                    for item in pending:
                        if item[0] < clip.start_frame:
                            emit(item)
                    pending = incoming
                    if pending:
                        first = pending[0]
                        pending[0] = (first[0], first[1], first[2], True)
                    continue

                flush_count = max(0, len(pending) - overlap)
                for item in pending[:flush_count]:
                    emit(item)
                tail = pending[flush_count:]
                blended: list[tuple[int, np.ndarray, np.ndarray, bool]] = []
                for offset in range(overlap):
                    previous_item = tail[offset]
                    next_item = incoming[offset]
                    source_frame = next_item[2]
                    previous_aligned = align_generated_to_source(previous_item[1], source_frame)
                    next_aligned = align_generated_to_source(next_item[1], source_frame)
                    phase = (offset + 1) / (overlap + 1)
                    alpha = 0.5 - 0.5 * math.cos(math.pi * phase)
                    frame = cv2.addWeighted(
                        previous_aligned,
                        1.0 - alpha,
                        next_aligned,
                        alpha,
                        0.0,
                    )
                    blended.append(
                        (
                            next_item[0],
                            frame,
                            source_frame,
                            offset == 0,
                        )
                    )
                pending = blended + incoming[overlap:]

            for item in pending:
                emit(item)
        finally:
            writer.close()
        report = quality.report()
        report["encoder_codec"] = writer.used_codec
        return report


class DreamIDVBackendFactory:
    def create(self, config: Any, model_path: Path) -> VideoModelBundle:
        model_path = model_path.expanduser().resolve()
        options = DreamIDVOptions.from_config(config)
        readiness = dreamidv_readiness(config, model_path, probe_environment=False)
        missing = [name for name, okay in readiness["checks"].items() if not okay]
        if missing:
            raise RuntimeError(
                "DreamID-V no estÃ¡ listo. Elementos faltantes: " + ", ".join(missing)
            )

        services = create_insightface_analysis_services(config)
        artifacts = [model_path, options.wan_checkpoint_dir]
        pose_models = options.repository_path / "pose" / "models"
        for name in ("dw-ll_ucoco_384.onnx", "yolox_l.onnx"):
            path = pose_models / name
            if path.is_file():
                artifacts.append(path)

        runtime = dict(services.runtime)
        runtime.update(
            {
                "repository_path": str(options.repository_path),
                "wan_checkpoint_dir": str(options.wan_checkpoint_dir),
                "dreamidv_checkpoint": str(model_path),
                "variant": options.variant,
                "size": options.size,
                "frame_num": options.frame_num,
                "sample_fps": options.sample_fps,
                "sample_steps": options.sample_steps,
                "chunk_overlap_frames": options.chunk_overlap_frames,
                "scene_aware_chunking": options.scene_aware_chunking,
                "persistent_worker": options.persistent_worker,
                "precompute_pose": options.precompute_pose,
                "precompute_pose_global": options.precompute_pose_global,
                "profile_worker": options.profile_worker,
                "profile_dit_forwards": options.profile_dit_forwards,
                "profile_worker_cprofile": options.profile_worker_cprofile,
                "profile_worker_cprofile_all": options.profile_worker_cprofile_all,
                "profile_detailed_clips": options.profile_detailed_clips,
                "worker_cprofile_top": options.worker_cprofile_top,
                "worker_heartbeat_seconds": options.worker_heartbeat_seconds,
                "cache_context": options.cache_context,
                "cache_reference_latents": options.cache_reference_latents,
                "cuda_cleanup_mode": options.cuda_cleanup_mode,
                "cuda_cleanup_reserved_ratio": options.cuda_cleanup_reserved_ratio,
                "offload_fallback": options.offload_fallback,
                "staged_offload": options.staged_offload,
                "offload_vae_during_dit": options.offload_vae_during_dit,
                "vae_dtype": options.vae_dtype,
                "stream_video_write": options.stream_video_write,
                "worker_restart_attempts": options.worker_restart_attempts,
                "release_analysis_gpu": options.release_analysis_gpu,
                "sdpa_backend_priority": options.sdpa_backend_priority,
                "sdpa_allow_math_fallback": options.sdpa_allow_math_fallback,
                "sdpa_padding_mode": options.sdpa_padding_mode,
                "sdpa_diagnostics": options.sdpa_diagnostics,
                "offload_model": options.offload_model,
                "t5_cpu": options.t5_cpu,
                "python_executable": options.python_executable,
            }
        )
        return VideoModelBundle(
            backend=BACKEND_NAME,
            analyzer=services.analyzer,
            processor=DreamIDVSubprocessBackend(
                model_path,
                options,
                analyzer=services.analyzer,
                identity_config=config.identity,
                tracking_config=config.tracking,
                encoding_config=config.encoding,
            ),
            providers=services.providers,
            runtime=runtime,
            capabilities=ModelCapabilities(
                generator="dreamid_v_wan_1.3b",
                native_output_size=720 if options.size == "1280*720" else 480,
                geometry_conditioning="pose_video_precomputed_separate_process",
                geometry_postprocess="target_track_mask_occlusion_aware",
                temporal_generation="video_diffusion_transformer_overlap_stitched",
            ),
            model_artifacts=tuple(artifacts),
        )


def dreamidv_readiness(
    config: Any,
    model_path: Path,
    *,
    probe_environment: bool = True,
) -> dict[str, Any]:
    model_path = model_path.expanduser().resolve()
    try:
        options = DreamIDVOptions.from_config(config)
    except (TypeError, ValueError) as exc:
        return {
            "ready": False,
            "error": str(exc),
            "checks": {"configuration": False},
        }

    python_ok = _python_exists(options.python_executable)
    script = options.repository_path / options.script_name
    package = options.repository_path / options.package_name
    pose_dir = options.repository_path / "pose" / "models"
    needs_dwpose = options.variant in {"faster", "dwpose"}
    wan_nonempty = options.wan_checkpoint_dir.is_dir() and any(
        options.wan_checkpoint_dir.iterdir()
    )
    checks = {
        "python_executable": python_ok,
        "repository": options.repository_path.is_dir(),
        "entry_script": script.is_file(),
        "runtime_package": package.is_dir(),
        "faster_context": (package / "context.pth").is_file()
        if options.variant == "faster"
        else True,
        "dreamidv_checkpoint": model_path.is_file(),
        "wan_checkpoint_directory": wan_nonempty,
        "dwpose_detector": (pose_dir / "dw-ll_ucoco_384.onnx").is_file()
        if needs_dwpose
        else True,
        "dwpose_pose_model": (pose_dir / "yolox_l.onnx").is_file()
        if needs_dwpose
        else True,
        "ffmpeg": select_ffmpeg() is not None,
        "persistent_worker_bridge": (
            Path(__file__).with_name("dreamidv_worker.py").is_file()
            if options.persistent_worker and options.variant in {"faster", "dwpose"}
            else True
        ),
        "pose_precompute_bridge": (
            Path(__file__).with_name("dreamidv_pose_worker.py").is_file()
            if (
                options.persistent_worker
                and options.precompute_pose
                and options.variant in {"faster", "dwpose"}
            )
            else True
        ),
        "native_sdpa_bridge": Path(__file__).with_name("dreamidv_sdpa.py").is_file(),
    }
    environment: dict[str, Any] | None = None
    if probe_environment and python_ok:
        environment = _probe_external_environment(options.python_executable)
        checks["python_dependencies"] = bool(environment.get("ready"))

    return {
        "ready": all(checks.values()),
        "variant": options.variant,
        "repository_path": str(options.repository_path),
        "entry_script": str(script),
        "dreamidv_checkpoint": str(model_path),
        "wan_checkpoint_dir": str(options.wan_checkpoint_dir),
        "python_executable": options.python_executable,
        "profile": {
            "size": options.size,
            "frame_num": options.frame_num,
            "sample_fps": options.sample_fps,
            "sample_steps": options.sample_steps,
            "offload_model": options.offload_model,
            "t5_cpu": options.t5_cpu,
            "chunking": options.chunking,
            "chunk_overlap_frames": options.chunk_overlap_frames,
            "scene_aware_chunking": options.scene_aware_chunking,
            "persistent_worker": options.persistent_worker,
            "precompute_pose": options.precompute_pose,
            "precompute_pose_global": options.precompute_pose_global,
            "profile_worker": options.profile_worker,
            "profile_dit_forwards": options.profile_dit_forwards,
            "profile_worker_cprofile": options.profile_worker_cprofile,
            "profile_worker_cprofile_all": options.profile_worker_cprofile_all,
            "profile_detailed_clips": options.profile_detailed_clips,
            "worker_cprofile_top": options.worker_cprofile_top,
            "worker_heartbeat_seconds": options.worker_heartbeat_seconds,
            "cache_context": options.cache_context,
            "cache_reference_latents": options.cache_reference_latents,
            "reference_latent_cache_size": options.reference_latent_cache_size,
            "cuda_cleanup_mode": options.cuda_cleanup_mode,
            "cuda_cleanup_reserved_ratio": options.cuda_cleanup_reserved_ratio,
            "offload_fallback": options.offload_fallback,
            "staged_offload": options.staged_offload,
            "offload_vae_during_dit": options.offload_vae_during_dit,
            "vae_dtype": options.vae_dtype,
            "stream_video_write": options.stream_video_write,
            "worker_restart_attempts": options.worker_restart_attempts,
            "release_analysis_gpu": options.release_analysis_gpu,
            "sdpa_backend_priority": options.sdpa_backend_priority,
            "sdpa_allow_math_fallback": options.sdpa_allow_math_fallback,
            "sdpa_padding_mode": options.sdpa_padding_mode,
            "sdpa_diagnostics": options.sdpa_diagnostics,
        },
        "checks": checks,
        "environment": environment,
    }


def _probe_external_environment(python_executable: str) -> dict[str, Any]:
    source = """
import json
payload = {"ready": False}
try:
    import torch
    import torchvision
    import cv2, decord, diffusers, numpy, onnxruntime, transformers
    payload.update({
        "ready": bool(torch.cuda.is_available()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available()
            else 0
        ),
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "decord": getattr(decord, "__version__", "unknown"),
        "diffusers": diffusers.__version__,
        "transformers": transformers.__version__,
        "onnxruntime": onnxruntime.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "sdpa_kernel_api": bool(
            hasattr(torch.nn, "attention")
            and hasattr(torch.nn.attention, "sdpa_kernel")
            and hasattr(torch.nn.attention, "SDPBackend")
        ),
        "sdpa_backends": (
            list(torch.nn.attention.SDPBackend.__members__.keys())
            if hasattr(torch.nn, "attention")
            and hasattr(torch.nn.attention, "SDPBackend")
            and hasattr(torch.nn.attention.SDPBackend, "__members__")
            else []
        ),
    })
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload))
"""
    try:
        completed = subprocess.run(
            [python_executable, "-c", source],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 - diagnÃ³stico, no inferencia
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {
            "ready": False,
            "error": completed.stderr.strip() or f"CÃ³digo de salida {completed.returncode}",
        }
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError:
        return {"ready": False, "error": lines[-1]}
    if completed.returncode != 0:
        payload["ready"] = False
        payload.setdefault("error", completed.stderr.strip())
    return payload

