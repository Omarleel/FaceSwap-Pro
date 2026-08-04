from __future__ import annotations

"""Worker persistente e instrumentado de DreamID-V.

El protocolo usa JSON Lines por stdin. Los mensajes de control se separan de la
salida normal del runtime oficial mediante tres prefijos:

* ``FACESWAP_RESULT``: respuesta a una solicitud.
* ``FACESWAP_PROFILE``: métricas que el proceso padre incorpora a profile.jsonl.
* ``FACESWAP_LOG``: problemas que el padre incorpora a logs.jsonl.

El perfilado CUDA usa eventos diferidos: no sincroniza después de cada forward,
por lo que conserva el paralelismo del runtime. Todos los eventos se resuelven
con una sola sincronización al terminar cada clip.
"""

import argparse
import cProfile
import contextlib
import gc
import importlib
import json
import os
import pstats
import sys
import threading
import time
import traceback
import types
import uuid
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

RESULT_PREFIX = "FACESWAP_RESULT "
PROFILE_PREFIX = "FACESWAP_PROFILE "
LOG_PREFIX = "FACESWAP_LOG "
_PRINT_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _emit(prefix: str, payload: Mapping[str, Any]) -> None:
    with _PRINT_LOCK:
        print(prefix + json.dumps(dict(payload), ensure_ascii=False, default=repr), flush=True)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--variant", choices=["faster", "dwpose"], required=True)
    parser.add_argument("--size", required=True)
    parser.add_argument("--sample-fps", type=int, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--dreamidv-checkpoint", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--t5-cpu", action="store_true")
    parser.add_argument("--profile-worker", action="store_true")
    parser.add_argument("--profile-dit-forwards", action="store_true")
    parser.add_argument("--profile-cprofile", action="store_true")
    parser.add_argument("--profile-cprofile-all", action="store_true")
    parser.add_argument("--profile-detailed-clips", type=int, default=1)
    parser.add_argument("--cprofile-top", type=int, default=80)
    parser.add_argument("--cache-context", action="store_true")
    parser.add_argument("--cache-reference-latents", action="store_true")
    parser.add_argument("--reference-latent-cache-size", type=int, default=8)
    parser.add_argument(
        "--cuda-cleanup-mode",
        choices=["adaptive", "always", "never"],
        default="adaptive",
    )
    parser.add_argument("--cuda-cleanup-reserved-ratio", type=float, default=0.82)
    parser.add_argument("--heartbeat-seconds", type=float, default=15.0)
    parser.add_argument("--offload-fallback", action="store_true")
    parser.add_argument("--staged-offload", action="store_true")
    parser.add_argument("--offload-vae-during-dit", action="store_true")
    parser.add_argument(
        "--vae-dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
    )
    parser.add_argument("--stream-video-write", action="store_true")
    parser.add_argument("--optimize-model-invariants", action="store_true")
    parser.add_argument("--optimize-scheduler-tensors", action="store_true")
    return parser.parse_args()


class WorkerProfiler:
    """Perfilador de baja interferencia con métricas de pared, CPU y CUDA."""

    def __init__(self, *, enabled: bool, device_id: int) -> None:
        self.enabled = enabled
        self.device_id = device_id
        self.process_id = os.getpid()
        self._request_fields: dict[str, Any] = {}
        self._pending_cuda: list[tuple[dict[str, Any], Any, Any]] = []
        self._counters: defaultdict[str, int] = defaultdict(int)
        self._durations_ns: defaultdict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        self._torch: Any | None = None

    def attach_torch(self, torch_module: Any) -> None:
        self._torch = torch_module

    def set_request(self, request: Mapping[str, Any]) -> str:
        request_id = str(request.get("request_id") or uuid.uuid4().hex[:16])
        self._request_fields = {
            "worker_request_id": request_id,
            "clip_index": request.get("clip_index"),
            "clip_start_frame": request.get("clip_start_frame"),
            "frame_num": request.get("frame_num"),
            "sample_steps": request.get("sample_steps"),
            "source_reference": request.get("source_reference"),
        }
        self._counters.clear()
        self._durations_ns.clear()
        return request_id

    def clear_request(self) -> None:
        self._request_fields = {}

    def _base(self) -> dict[str, Any]:
        return {
            "timestamp": _utc_now(),
            "thread": threading.current_thread().name,
            "process_id": self.process_id,
            "source": "dreamidv_worker",
            **self._request_fields,
        }

    def event(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return
        _emit(PROFILE_PREFIX, {**self._base(), "event": event, **fields})

    def log(self, level: str, message: str, **fields: Any) -> None:
        _emit(
            LOG_PREFIX,
            {
                **self._base(),
                "level": level.upper(),
                "message": message,
                **fields,
            },
        )

    def count(self, name: str, increment: int = 1) -> int:
        with self._lock:
            self._counters[name] += increment
            return self._counters[name]

    def gpu_memory(self) -> dict[str, Any]:
        torch = self._torch
        if torch is None or not torch.cuda.is_available():
            return {"cuda_available": False}
        try:
            device = torch.device(f"cuda:{self.device_id}")
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            return {
                "cuda_available": True,
                "cuda_device": str(device),
                "cuda_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "cuda_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "cuda_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "cuda_free_bytes": int(free_bytes),
                "cuda_total_bytes": int(total_bytes),
            }
        except Exception as exc:  # pragma: no cover - depende de la build CUDA
            return {"cuda_available": True, "cuda_metrics_error": str(exc)}

    def gpu_snapshot(self, stage: str, **fields: Any) -> dict[str, Any]:
        metrics = self.gpu_memory()
        self.event("dreamidv.gpu_memory", stage=stage, **metrics, **fields)
        return metrics

    @contextlib.contextmanager
    def span(self, name: str, *, cuda: bool = False, **fields: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        started_ns = time.perf_counter_ns()
        cpu_started_ns = time.thread_time_ns()
        start_event = end_event = None
        torch = self._torch
        if cuda and torch is not None and torch.cuda.is_available():
            try:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            except Exception:
                start_event = end_event = None

        status = "ok"
        error_type: str | None = None
        try:
            yield
        except BaseException as exc:
            status = "error"
            error_type = type(exc).__name__
            raise
        finally:
            ended_ns = time.perf_counter_ns()
            cpu_ended_ns = time.thread_time_ns()
            if end_event is not None:
                try:
                    end_event.record()
                except Exception:
                    start_event = end_event = None
            duration_ns = ended_ns - started_ns
            with self._lock:
                self._counters[f"span_count:{name}"] += 1
                self._durations_ns[name] += duration_ns
            record = {
                **self._base(),
                "event": "span",
                "name": name,
                "span_id": uuid.uuid4().hex[:16],
                "status": status,
                "error_type": error_type,
                "started_ns": started_ns,
                "duration_ns": duration_ns,
                "duration_seconds": duration_ns / 1_000_000_000,
                "thread_cpu_ns": cpu_ended_ns - cpu_started_ns,
                **fields,
            }
            if start_event is not None and end_event is not None:
                with self._lock:
                    self._pending_cuda.append((record, start_event, end_event))
            else:
                _emit(PROFILE_PREFIX, record)

    def flush_cuda(self) -> None:
        if not self.enabled:
            return
        torch = self._torch
        with self._lock:
            pending = self._pending_cuda
            self._pending_cuda = []
        if not pending:
            return
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception as exc:
                self.log("WARNING", "No se pudo sincronizar CUDA para el perfil", error=str(exc))
        for record, start_event, end_event in pending:
            try:
                record["cuda_duration_ms"] = float(start_event.elapsed_time(end_event))
                record["cuda_duration_ns"] = int(record["cuda_duration_ms"] * 1_000_000)
            except Exception as exc:
                record["cuda_timing_error"] = str(exc)
            _emit(PROFILE_PREFIX, record)

    def emit_cprofile(
        self, profiler: cProfile.Profile, *, scope: str, limit: int
    ) -> None:
        if not self.enabled:
            return
        stats = pstats.Stats(profiler)
        rows: list[tuple[float, tuple[str, int, str], tuple[Any, ...]]] = []
        for function, values in stats.stats.items():
            primitive_calls, total_calls, total_time, cumulative_time, callers = values
            del callers
            if total_time <= 0.0 and cumulative_time <= 0.0:
                continue
            rows.append(
                (
                    cumulative_time,
                    function,
                    (primitive_calls, total_calls, total_time, cumulative_time),
                )
            )
        rows.sort(key=lambda item: item[0], reverse=True)
        for _, function, values in rows[: max(1, int(limit))]:
            filename, line_number, function_name = function
            primitive_calls, total_calls, total_time, cumulative_time = values
            self.event(
                "cprofile_function",
                scope=scope,
                filename=filename,
                line_number=line_number,
                function=function_name,
                primitive_calls=primitive_calls,
                total_calls=total_calls,
                total_seconds=total_time,
                cumulative_seconds=cumulative_time,
            )

    def summary(self, **fields: Any) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            durations = dict(self._durations_ns)
        result = {
            "dit_forward_calls": counters.get("dit_forward", 0),
            "vae_encode_calls": counters.get("vae_encode", 0),
            "vae_decode_calls": counters.get("vae_decode", 0),
            "context_cache_hits": counters.get("context_cache_hit", 0),
            "context_cache_misses": counters.get("context_cache_miss", 0),
            "reference_latent_cache_hits": counters.get("reference_latent_cache_hit", 0),
            "reference_latent_cache_misses": counters.get("reference_latent_cache_miss", 0),
            "span_duration_ns": durations,
            **fields,
            **self.gpu_memory(),
        }
        self.event("dreamidv.clip_summary", **result)
        return result


def _tensor_descriptors(values: Any, *, limit: int = 8) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(value: Any) -> None:
        if len(result) >= limit or id(value) in visited:
            return
        visited.add(id(value))
        shape = getattr(value, "shape", None)
        dtype = getattr(value, "dtype", None)
        device = getattr(value, "device", None)
        if shape is not None and dtype is not None:
            try:
                result.append(
                    {
                        "shape": [int(item) for item in shape],
                        "dtype": str(dtype),
                        "device": str(device),
                    }
                )
            except Exception:
                pass
            return
        if isinstance(value, Mapping):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(values)
    return result


def _is_single_frame_input(values: Any) -> bool:
    descriptors = _tensor_descriptors(values)
    candidates = [item["shape"] for item in descriptors if len(item["shape"]) >= 4]
    return bool(candidates) and all(shape[-3] == 1 for shape in candidates)


def _move_tensors(value: Any, device: Any) -> Any:
    if hasattr(value, "to") and hasattr(value, "dtype"):
        return value.to(device)
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    return value


class WorkerRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.profiler = WorkerProfiler(enabled=args.profile_worker, device_id=args.device_id)
        self.profiler.event(
            "dreamidv.worker_start",
            variant=args.variant,
            size=args.size,
            sample_fps=args.sample_fps,
            python=sys.version,
        )
        repository = args.repository.expanduser().resolve()
        sys.path.insert(0, str(repository))
        self.repository = repository
        self.package_name = "dreamidv_wan_faster" if args.variant == "faster" else "dreamidv_wan"
        self._reference_latent_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._context_cache: Any | None = None
        self._context_cache_enabled = bool(args.cache_context)
        self._reference_cache_enabled = bool(args.cache_reference_latents)
        self._original_torch_load: Callable[..., Any] | None = None
        self._invariant_optimizer: Any | None = None
        self._scheduler_torch_proxy: Any | None = None

        with self.profiler.span("dreamidv.initialize.imports"):
            from faceswap_pro.dreamidv_sdpa import (
                install_attention_override,
                sdpa_runtime_summary,
            )

            install_attention_override(self.package_name)
            print(f"FaceSwap-Pro SDPA configurado: {sdpa_runtime_summary()}", flush=True)
            package = importlib.import_module(self.package_name)
            configs = importlib.import_module(f"{self.package_name}.configs")
            utils = importlib.import_module(f"{self.package_name}.utils.utils")
            import torch

            self.torch = torch
            self.profiler.attach_torch(torch)
            self.profiler.event(
                "dreamidv.runtime",
                torch_version=str(torch.__version__),
                cuda_version=str(getattr(torch.version, "cuda", None)),
                sdpa=sdpa_runtime_summary(),
            )

        self.size_configs = configs.SIZE_CONFIGS
        self.cache_video = utils.cache_video
        self.cfg = configs.WAN_CONFIGS["swapface"]
        self.cfg.sample_fps = args.sample_fps
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
        self.profiler.gpu_snapshot("before_pipeline_init")
        init_started = time.perf_counter()
        stop_init_heartbeat = threading.Event()
        init_heartbeat = self._start_heartbeat(
            stop_init_heartbeat,
            init_started,
            stage="pipeline_init",
            include_gpu=False,
        )
        try:
            with self.profiler.span("dreamidv.initialize.pipeline", cuda=True):
                self.pipeline = package.DreamIDV(
                    config=self.cfg,
                    checkpoint_dir=str(args.checkpoint_dir.resolve()),
                    dreamidv_ckpt=str(args.dreamidv_checkpoint.resolve()),
                    device_id=args.device_id,
                    rank=0,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_usp=False,
                    t5_cpu=args.t5_cpu,
                )
        finally:
            self.profiler.flush_cuda()
            stop_init_heartbeat.set()
            init_heartbeat.join(timeout=2.0)
        self.profiler.gpu_snapshot("after_pipeline_init")
        self._install_runtime_optimizations()
        self._install_context_cache()
        self._instrument_pipeline_components()
        self.profiler.event(
            "dreamidv.worker_ready",
            cache_context=self._context_cache_enabled,
            cache_reference_latents=self._reference_cache_enabled,
            cuda_cleanup_mode=args.cuda_cleanup_mode,
            staged_offload=bool(args.staged_offload),
            offload_vae_during_dit=bool(args.offload_vae_during_dit),
            vae_dtype=str(args.vae_dtype),
            stream_video_write=bool(args.stream_video_write),
            optimize_model_invariants=bool(args.optimize_model_invariants),
            optimize_scheduler_tensors=bool(args.optimize_scheduler_tensors),
        )

    def _detailed_profile_enabled(self) -> bool:
        clip_index = self.profiler._request_fields.get("clip_index")
        try:
            index = int(clip_index or 0)
        except (TypeError, ValueError):
            index = 0
        return index < max(0, int(self.args.profile_detailed_clips))

    def _install_runtime_optimizations(self) -> None:
        """Instala optimizaciones compatibles con el checkout oficial Faster.

        Se aplican dinámicamente para que también funcionen cuando DreamID-V ya
        estaba clonado antes de actualizar FaceSwap-Pro. Si el fork externo no
        expone la estructura esperada, el worker conserva la implementación
        original y registra el motivo en vez de impedir el arranque.
        """

        if self.args.variant != "faster":
            return
        # Liberar primero el DiT evita que la conversión FP32→BF16 del VAE
        # necesite coexistir con ambos juegos de pesos en una GPU de 16 GB.
        if self.args.staged_offload:
            self._move_dit("cpu", stage="before_vae_precision")
            self._release_cuda_cache(stage="before_vae_precision")
        self._configure_vae_precision()
        self._patch_temporal_vae_concat()
        self._patch_vae_wrappers()
        if self.args.optimize_model_invariants:
            try:
                from faceswap_pro.dreamidv_invariants import DreamIDVInvariantOptimizer

                self._invariant_optimizer = DreamIDVInvariantOptimizer(
                    package_name=self.package_name,
                    model=self.pipeline.model,
                    torch_module=self.torch,
                    attention_dtype=getattr(
                        self.pipeline, "param_dtype", self.torch.bfloat16
                    ),
                    event=self.profiler.event,
                )
                if not self._invariant_optimizer.install():
                    self._invariant_optimizer = None
                    self.profiler.log(
                        "WARNING",
                        "El checkout DreamID-V no expone las funciones invariantes esperadas",
                    )
            except Exception as exc:
                self._invariant_optimizer = None
                self.profiler.log(
                    "WARNING",
                    "No se pudieron instalar las cachés invariantes DreamID-V",
                    error=str(exc),
                )
        self._patch_faster_generate()

    def _resolved_vae_dtype(self) -> Any:
        requested = str(self.args.vae_dtype)
        mapping = {
            "bfloat16": self.torch.bfloat16,
            "float16": self.torch.float16,
            "float32": self.torch.float32,
        }
        if requested != "auto":
            return mapping[requested]
        pipeline_dtype = getattr(self.pipeline, "param_dtype", None)
        if pipeline_dtype in {self.torch.bfloat16, self.torch.float16}:
            return pipeline_dtype
        return self.torch.bfloat16 if self.torch.cuda.is_available() else self.torch.float32

    def _configure_vae_precision(self) -> None:
        vae = getattr(self.pipeline, "vae", None)
        model = getattr(vae, "model", None)
        if vae is None or model is None:
            self.profiler.log("WARNING", "No se encontró el VAE para configurar precisión")
            return
        dtype = self._resolved_vae_dtype()
        device = getattr(self.pipeline, "device", self.torch.device("cpu"))
        try:
            with self.profiler.span(
                "dreamidv.optimize.vae_precision",
                cuda=self.torch.cuda.is_available(),
                target_dtype=str(dtype),
            ):
                model.to(device=device, dtype=dtype)
                vae.dtype = dtype
                for name in ("mean", "std"):
                    value = getattr(vae, name, None)
                    if value is not None and hasattr(value, "to"):
                        setattr(vae, name, value.to(device=device, dtype=dtype))
                if getattr(vae, "mean", None) is not None and getattr(vae, "std", None) is not None:
                    vae.scale = [vae.mean, 1.0 / vae.std]
            self.profiler.event(
                "dreamidv.optimization",
                optimization="vae_precision",
                enabled=True,
                dtype=str(dtype),
            )
        except Exception as exc:
            self.profiler.log(
                "WARNING",
                "No se pudo convertir el VAE a la precisión optimizada; se conserva el dtype original",
                error=str(exc),
            )

    def _patch_temporal_vae_concat(self) -> None:
        vae = getattr(self.pipeline, "vae", None)
        model = getattr(vae, "model", None)
        required = ("encoder", "decoder", "conv1", "conv2", "clear_cache", "z_dim")
        if model is None or not all(hasattr(model, name) for name in required):
            self.profiler.log(
                "WARNING",
                "El VAE no expone la estructura oficial; no se cambia la concatenación temporal",
            )
            return
        if getattr(model, "_faceswap_concat_optimized", False):
            return
        torch = self.torch

        def optimized_encode(model_self: Any, x: Any, scale: Any) -> Any:
            model_self.clear_cache()
            temporal_chunks: list[Any] = []
            iterations = 1 + (int(x.shape[2]) - 1) // 4
            for index in range(iterations):
                model_self._enc_conv_idx = [0]
                if index == 0:
                    chunk = x[:, :, :1, :, :]
                else:
                    chunk = x[:, :, 1 + 4 * (index - 1) : 1 + 4 * index, :, :]
                temporal_chunks.append(
                    model_self.encoder(
                        chunk,
                        feat_cache=model_self._enc_feat_map,
                        feat_idx=model_self._enc_conv_idx,
                    )
                )
            out = temporal_chunks[0] if len(temporal_chunks) == 1 else torch.cat(temporal_chunks, 2)
            mu, _log_var = model_self.conv1(out).chunk(2, dim=1)
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, model_self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, model_self.z_dim, 1, 1, 1
                )
            else:
                mu = (mu - scale[0]) * scale[1]
            model_self.clear_cache()
            return mu

        def optimized_decode(model_self: Any, z: Any, scale: Any) -> Any:
            model_self.clear_cache()
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, model_self.z_dim, 1, 1, 1) + scale[0].view(
                    1, model_self.z_dim, 1, 1, 1
                )
            else:
                z = z / scale[1] + scale[0]
            x = model_self.conv2(z)
            temporal_chunks: list[Any] = []
            for index in range(int(z.shape[2])):
                model_self._conv_idx = [0]
                temporal_chunks.append(
                    model_self.decoder(
                        x[:, :, index : index + 1, :, :],
                        feat_cache=model_self._feat_map,
                        feat_idx=model_self._conv_idx,
                    )
                )
            out = temporal_chunks[0] if len(temporal_chunks) == 1 else torch.cat(temporal_chunks, 2)
            model_self.clear_cache()
            return out

        model.encode = types.MethodType(optimized_encode, model)
        model.decode = types.MethodType(optimized_decode, model)
        model._faceswap_concat_optimized = True
        self.profiler.event(
            "dreamidv.optimization",
            optimization="vae_temporal_list_concat",
            enabled=True,
        )

    def _autocast_context(self, dtype: Any) -> Any:
        enabled = self.torch.cuda.is_available() and dtype in {
            self.torch.bfloat16,
            self.torch.float16,
        }
        if not enabled:
            return contextlib.nullcontext()
        return self.torch.autocast(device_type="cuda", dtype=dtype)

    def _patch_vae_wrappers(self) -> None:
        vae = getattr(self.pipeline, "vae", None)
        model = getattr(vae, "model", None)
        if vae is None or model is None or not hasattr(vae, "scale"):
            return
        if getattr(vae, "_faceswap_precision_optimized", False):
            return
        runtime = self

        def optimized_encode(vae_self: Any, videos: Any, device: Any) -> list[Any]:
            dtype = getattr(vae_self, "dtype", runtime._resolved_vae_dtype())
            with runtime._autocast_context(dtype), runtime.torch.inference_mode():
                return [
                    vae_self.model.encode(
                        video.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True),
                        vae_self.scale,
                    ).squeeze(0)
                    for video in videos
                ]

        def optimized_decode(vae_self: Any, latents: Any) -> list[Any]:
            dtype = getattr(vae_self, "dtype", runtime._resolved_vae_dtype())
            device = getattr(runtime.pipeline, "device", runtime.torch.device("cpu"))
            with runtime._autocast_context(dtype), runtime.torch.inference_mode():
                return [
                    vae_self.model.decode(
                        latent.unsqueeze(0).to(device=device, dtype=dtype, non_blocking=True),
                        vae_self.scale,
                    ).clamp_(-1, 1).squeeze(0)
                    for latent in latents
                ]

        vae.encode = types.MethodType(optimized_encode, vae)
        vae.decode = types.MethodType(optimized_decode, vae)
        vae._faceswap_precision_optimized = True
        self.profiler.event(
            "dreamidv.optimization",
            optimization="vae_keep_low_precision",
            enabled=True,
            dtype=str(getattr(vae, "dtype", None)),
        )

    def _move_dit(self, target: Any, *, stage: str) -> None:
        model = getattr(self.pipeline, "model", None)
        if model is None:
            return
        target_device = self.torch.device(target)
        current_device = self._module_device(model)
        if current_device == target_device:
            self.profiler.event(
                "dreamidv.stage_move_skipped",
                component="dit",
                stage=stage,
                device=str(target_device),
            )
            return
        with self.profiler.span(
            "dreamidv.stage.dit_move",
            cuda=self.torch.cuda.is_available(),
            stage=stage,
            target_device=str(target_device),
        ):
            model.to(target_device)

    def _move_vae(self, target: Any, *, stage: str) -> None:
        vae = getattr(self.pipeline, "vae", None)
        model = getattr(vae, "model", None)
        if vae is None or model is None:
            return
        target_device = self.torch.device(target)
        dtype = getattr(vae, "dtype", self._resolved_vae_dtype())
        current_device = self._module_device(model)
        current_dtype = self._module_dtype(model)
        if current_device == target_device and current_dtype == dtype:
            self.profiler.event(
                "dreamidv.stage_move_skipped",
                component="vae",
                stage=stage,
                device=str(target_device),
                dtype=str(dtype),
            )
            return
        with self.profiler.span(
            "dreamidv.stage.vae_move",
            cuda=self.torch.cuda.is_available(),
            stage=stage,
            target_device=str(target_device),
            dtype=str(dtype),
        ):
            model.to(device=target_device, dtype=dtype)
            for name in ("mean", "std"):
                value = getattr(vae, name, None)
                if value is not None and hasattr(value, "to"):
                    setattr(vae, name, value.to(device=target_device, dtype=dtype))
            if getattr(vae, "mean", None) is not None and getattr(vae, "std", None) is not None:
                vae.scale = [vae.mean, 1.0 / vae.std]

    @staticmethod
    def _module_device(module: Any) -> Any | None:
        try:
            return next(module.parameters()).device
        except (StopIteration, AttributeError, TypeError):
            try:
                return next(module.buffers()).device
            except (StopIteration, AttributeError, TypeError):
                return None

    @staticmethod
    def _module_dtype(module: Any) -> Any | None:
        try:
            return next(module.parameters()).dtype
        except (StopIteration, AttributeError, TypeError):
            try:
                return next(module.buffers()).dtype
            except (StopIteration, AttributeError, TypeError):
                return None

    def _release_cuda_cache(self, *, stage: str) -> None:
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        self.profiler.gpu_snapshot(stage)

    def _patch_faster_generate(self) -> None:
        original = getattr(self.pipeline, "generate", None)
        if not callable(original) or getattr(original, "_faceswap_staged_optimized", False):
            return
        try:
            scheduler_module = importlib.import_module(
                f"{self.package_name}.utils.fm_solvers_unipc"
            )
            scheduler_class = scheduler_module.FlowUniPCMultistepScheduler
            if (
                self._invariant_optimizer is not None
                and self.args.optimize_scheduler_tensors
            ):
                self._scheduler_torch_proxy = (
                    self._invariant_optimizer.install_scheduler_proxy(scheduler_module)
                )
                self.profiler.event(
                    "dreamidv.optimization",
                    optimization="unipc_tensor_proxy",
                    enabled=True,
                )
        except Exception as exc:
            self.profiler.log(
                "WARNING",
                "No se pudo cargar el scheduler Faster; se conserva generate original",
                error=str(exc),
            )
            return
        runtime = self
        torch = self.torch

        def optimized_generate(
            pipeline_self: Any,
            input_prompt: str,
            paths: Any,
            size: Any = (1280, 720),
            frame_num: int = 81,
            shift: float = 5.0,
            sample_solver: str = "unipc",
            sampling_steps: int = 50,
            guide_scale_img: float = 5.0,
            n_prompt: str = "",
            seed: int = -1,
            offload_model: bool = True,
        ) -> Any:
            del input_prompt, n_prompt
            device = pipeline_self.device
            dtype = pipeline_self.param_dtype
            staged = bool(runtime.args.staged_offload)

            if staged:
                runtime._move_dit("cpu", stage="before_vae_encode")
                runtime._move_vae(device, stage="before_vae_encode")
                runtime._release_cuda_cache(stage="before_vae_encode")

            with runtime.profiler.span("dreamidv.stage.vae_preprocess", cuda=True):
                latents_ref = pipeline_self.load_image_latent_ref_ip_video(
                    paths, size, device, frame_num
                )
                latents_ref_video = latents_ref["video"].to(device=device, dtype=dtype)
                latents_ref_image = latents_ref["image"].to(device=device, dtype=dtype)
                mask = latents_ref["mask"].to(device=device, dtype=dtype)
                del latents_ref

            target_shape = (
                pipeline_self.vae.model.z_dim,
                latents_ref_video.shape[1],
                latents_ref_video.shape[2],
                latents_ref_video.shape[3],
            )
            seq_len = __import__("math").ceil(
                (target_shape[2] * target_shape[3])
                / (pipeline_self.patch_size[1] * pipeline_self.patch_size[2])
                * target_shape[1]
                / pipeline_self.sp_size
            ) * pipeline_self.sp_size

            if seed < 0:
                seed = __import__("random").randint(0, sys.maxsize)
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
            raw_context = torch.load(f"{runtime.package_name}/context.pth")
            conditioned = torch.concat([latents_ref_video, mask])
            zero_reference = torch.zeros_like(latents_ref_image)
            noise = [
                torch.randn(
                    *target_shape,
                    dtype=torch.float32,
                    device=device,
                    generator=generator,
                )
            ]

            if staged and runtime.args.offload_vae_during_dit:
                runtime._move_vae("cpu", stage="before_dit")
                runtime._release_cuda_cache(stage="after_vae_offload")
            runtime._move_dit(device, stage="before_dit")

            no_sync = getattr(pipeline_self.model, "no_sync", contextlib.nullcontext)
            with runtime._autocast_context(dtype), torch.inference_mode(), no_sync():
                if runtime._invariant_optimizer is not None:
                    with runtime.profiler.span(
                        "dreamidv.stage.context_encode_cached", cuda=True
                    ):
                        context = runtime._invariant_optimizer.encode_context(
                            raw_context, device=device
                        )
                else:
                    context = [
                        tensor.to(device=device, non_blocking=True)
                        for tensor in raw_context
                    ]
                args_conditioned = {
                    "context": context,
                    "seq_len": seq_len,
                    "y": [conditioned],
                    "img_ref": [latents_ref_image],
                }
                args_unconditioned = {
                    "context": context,
                    "seq_len": seq_len,
                    "y": [conditioned],
                    "img_ref": [zero_reference],
                }
                if sample_solver != "unipc":
                    raise NotImplementedError("DreamID-V Faster optimizado admite unipc.")
                scheduler = scheduler_class(
                    num_train_timesteps=pipeline_self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
                latents = noise
                for timestep in scheduler.timesteps:
                    timestep_batch = timestep.reshape(1)
                    positive = pipeline_self.model(
                        latents, t=timestep_batch, **args_conditioned
                    )[0]
                    negative = pipeline_self.model(
                        latents, t=timestep_batch, **args_unconditioned
                    )[0]
                    noise_prediction = positive + guide_scale_img * (positive - negative)
                    next_latent = scheduler.step(
                        noise_prediction.unsqueeze(0),
                        timestep,
                        latents[0].unsqueeze(0),
                        return_dict=False,
                        generator=generator,
                    )[0]
                    latents = [next_latent.squeeze(0)]
                    del positive, negative, noise_prediction, next_latent
                decoded_latents = latents

            del noise, scheduler, context, raw_context, conditioned, zero_reference
            del args_conditioned, args_unconditioned, latents_ref_video, latents_ref_image, mask
            if runtime._invariant_optimizer is not None:
                runtime._invariant_optimizer.clear_clip_caches()
            if staged or offload_model:
                runtime._move_dit("cpu", stage="before_vae_decode")
                runtime._release_cuda_cache(stage="after_dit_offload")
            if staged and runtime.args.offload_vae_during_dit:
                runtime._move_vae(device, stage="before_vae_decode")

            videos = pipeline_self.vae.decode(decoded_latents) if pipeline_self.rank == 0 else None
            del decoded_latents
            return videos[0] if pipeline_self.rank == 0 else None

        optimized_generate._faceswap_staged_optimized = True  # type: ignore[attr-defined]
        self.pipeline.generate = types.MethodType(optimized_generate, self.pipeline)
        self.profiler.event(
            "dreamidv.optimization",
            optimization="staged_model_residency",
            enabled=True,
            staged_offload=bool(self.args.staged_offload),
            offload_vae_during_dit=bool(self.args.offload_vae_during_dit),
            dit_move_outside_loop=True,
            inference_mode=True,
        )

    def _install_context_cache(self) -> None:
        if not self._context_cache_enabled:
            return
        context_path = (self.repository / self.package_name / "context.pth").resolve()
        if not context_path.is_file():
            self.profiler.log(
                "WARNING",
                "No se encontró context.pth; se desactiva su caché",
                context_path=str(context_path),
            )
            self._context_cache_enabled = False
            return
        self._original_torch_load = self.torch.load
        runtime = self

        def cached_torch_load(path: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                candidate = Path(os.fspath(path))
                if not candidate.is_absolute():
                    candidate = (Path.cwd() / candidate).resolve()
                else:
                    candidate = candidate.resolve()
            except (TypeError, ValueError, OSError):
                candidate = None
            if candidate != context_path or not runtime._context_cache_enabled:
                return runtime._original_torch_load(path, *args, **kwargs)
            if runtime._context_cache is not None:
                runtime.profiler.count("context_cache_hit")
                runtime.profiler.event("dreamidv.context_cache", cache_hit=True)
                return runtime._context_cache
            runtime.profiler.count("context_cache_miss")
            with runtime.profiler.span("dreamidv.context.load", cuda=True, cache_hit=False):
                loaded = runtime._original_torch_load(path, *args, **kwargs)
                # Mantener el contexto en CPU evita ocupar VRAM durante las fases
                # VAE. ``generate`` lo transfiere una sola vez al iniciar el DiT.
                loaded = _move_tensors(loaded, runtime.torch.device("cpu"))
                runtime._context_cache = loaded
            return runtime._context_cache

        self.torch.load = cached_torch_load

    def _instrument_pipeline_components(self) -> None:
        model = getattr(self.pipeline, "model", None)
        if (
            self.profiler.enabled
            and self.args.profile_dit_forwards
            and model is not None
            and callable(getattr(model, "forward", None))
        ):
            self._wrap_method(
                model,
                "forward",
                "dreamidv.dit.forward",
                counter="dit_forward",
                detailed=True,
            )

        vae = getattr(self.pipeline, "vae", None)
        if vae is not None:
            if (
                self.profiler.enabled or self._reference_cache_enabled
            ) and callable(getattr(vae, "encode", None)):
                self._wrap_vae_encode(vae)
            if self.profiler.enabled and callable(getattr(vae, "decode", None)):
                self._wrap_method(
                    vae,
                    "decode",
                    "dreamidv.vae.decode",
                    counter="vae_decode",
                    detailed=True,
                )

        if not self.profiler.enabled:
            return
        for attribute in ("clip", "image_encoder", "text_encoder", "t5"):
            component = getattr(self.pipeline, attribute, None)
            if component is None:
                continue
            for method_name in ("visual", "encode", "encode_image"):
                if callable(getattr(component, method_name, None)):
                    self._wrap_method(
                        component,
                        method_name,
                        f"dreamidv.{attribute}.{method_name}",
                        counter=f"{attribute}_{method_name}",
                        detailed=True,
                    )

    def _wrap_method(
        self,
        component: Any,
        method_name: str,
        span_name: str,
        *,
        counter: str,
        detailed: bool,
    ) -> None:
        original = getattr(component, method_name)
        if getattr(original, "_faceswap_profiled", False):
            return
        runtime = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            index = runtime.profiler.count(counter) - 1
            if counter == "dit_forward" and not runtime._detailed_profile_enabled():
                return original(*args, **kwargs)
            if not detailed:
                return original(*args, **kwargs)
            fields: dict[str, Any] = {"call_index": index}
            if counter == "dit_forward":
                fields.update(
                    {
                        "diffusion_step_index": index // 2,
                        "guidance_pass": "conditional" if index % 2 == 0 else "unconditional",
                    }
                )
            descriptors = _tensor_descriptors((args, kwargs))
            if descriptors:
                fields["input_tensors"] = descriptors
            with runtime.profiler.span(span_name, cuda=True, **fields):
                return original(*args, **kwargs)

        wrapped._faceswap_profiled = True  # type: ignore[attr-defined]
        setattr(component, method_name, wrapped)

    def _wrap_vae_encode(self, vae: Any) -> None:
        original = getattr(vae, "encode")
        runtime = self

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            index = runtime.profiler.count("vae_encode") - 1
            single_frame = _is_single_frame_input((args, kwargs))
            reference_path = runtime.profiler._request_fields.get("source_reference")
            key = None
            if runtime._reference_cache_enabled and single_frame and reference_path:
                path = Path(str(reference_path))
                try:
                    stat = path.stat()
                    fingerprint = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
                except OSError:
                    fingerprint = (str(path), None, None)
                key = (fingerprint, runtime.args.size)
                if key in runtime._reference_latent_cache:
                    runtime.profiler.count("reference_latent_cache_hit")
                    cached = runtime._reference_latent_cache.pop(key)
                    runtime._reference_latent_cache[key] = cached
                    with runtime.profiler.span(
                        "dreamidv.vae.encode",
                        cuda=False,
                        call_index=index,
                        input_kind="reference",
                        cache_hit=True,
                    ):
                        return cached

            runtime.profiler.count("reference_latent_cache_miss" if key else "vae_encode_uncached")
            descriptors = _tensor_descriptors((args, kwargs))
            with runtime.profiler.span(
                "dreamidv.vae.encode",
                cuda=True,
                call_index=index,
                input_kind="reference" if single_frame else "temporal",
                cache_hit=False,
                input_tensors=descriptors,
            ):
                result = original(*args, **kwargs)
            if key is not None:
                # La referencia es pequeña y se reutiliza, pero conservarla en
                # GPU reduce el margen del VAE. Se cachea en CPU y el pipeline la
                # mueve al dtype/device final cuando la consume.
                runtime._reference_latent_cache[key] = _move_tensors(
                    result, runtime.torch.device("cpu")
                )
                while len(runtime._reference_latent_cache) > max(
                    1, runtime.args.reference_latent_cache_size
                ):
                    runtime._reference_latent_cache.popitem(last=False)
            return result

        wrapped._faceswap_profiled = True  # type: ignore[attr-defined]
        setattr(vae, "encode", wrapped)

    def _drop_gpu_caches(self, *, disable_persistent_caches: bool) -> None:
        self._reference_latent_cache.clear()
        self._context_cache = None
        if disable_persistent_caches:
            self._reference_cache_enabled = False
            self._context_cache_enabled = False
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _cleanup_cuda(self, *, force: bool, offload_model: bool) -> dict[str, Any]:
        gc.collect()
        if not self.torch.cuda.is_available():
            return {"cuda_available": False, "cache_cleared": False}
        device = self.torch.device(f"cuda:{self.args.device_id}")
        allocated = int(self.torch.cuda.memory_allocated(device))
        reserved = int(self.torch.cuda.memory_reserved(device))
        free_bytes, total_bytes = self.torch.cuda.mem_get_info(device)
        inactive = max(0, reserved - allocated)
        reserved_ratio = reserved / max(int(total_bytes), 1)
        mode = self.args.cuda_cleanup_mode
        should_clear = force or mode == "always"
        if mode == "adaptive":
            should_clear = should_clear or (
                offload_model and inactive > 256 * 1024 * 1024
            ) or (
                reserved_ratio >= self.args.cuda_cleanup_reserved_ratio
                and inactive > 512 * 1024 * 1024
            )
        if mode == "never" and not force:
            should_clear = False
        if should_clear:
            self.torch.cuda.empty_cache()
        result = {
            "cuda_available": True,
            "cache_cleared": should_clear,
            "cuda_allocated_bytes_before_cleanup": allocated,
            "cuda_reserved_bytes_before_cleanup": reserved,
            "cuda_inactive_reserved_bytes": inactive,
            "cuda_reserved_ratio": reserved_ratio,
            "cuda_free_bytes_before_cleanup": int(free_bytes),
        }
        self.profiler.event("dreamidv.cuda_cleanup", **result)
        return result

    def _generate_once(
        self,
        request: Mapping[str, Any],
        *,
        input_video: Path,
        source_reference: Path,
        output_video: Path,
        pose_path: Path,
        mask_path: Path,
        offload_model: bool,
    ) -> None:
        if self.args.variant == "faster":
            ref_paths = [str(input_video), str(mask_path), str(source_reference)]
        else:
            ref_paths = [
                str(input_video),
                str(mask_path),
                str(source_reference),
                str(pose_path),
            ]

        video = None
        function_profiler = (
            cProfile.Profile()
            if self.profiler.enabled
            and self.args.profile_cprofile
            and (self.args.profile_cprofile_all or self._detailed_profile_enabled())
            else None
        )
        if function_profiler is not None:
            function_profiler.enable()
        try:
            with self.profiler.span(
                "dreamidv.pipeline.generate",
                cuda=True,
                offload_model=offload_model,
            ):
                video = self.pipeline.generate(
                    "chang face",
                    ref_paths,
                    size=self.size_configs[self.args.size],
                    frame_num=int(request["frame_num"]),
                    shift=float(request["sample_shift"]),
                    sample_solver=str(request["sample_solver"]),
                    sampling_steps=int(request["sample_steps"]),
                    guide_scale_img=float(request["guide_scale"]),
                    seed=int(request["seed"]),
                    offload_model=offload_model,
                )
        finally:
            if function_profiler is not None:
                function_profiler.disable()
                self.profiler.emit_cprofile(
                    function_profiler,
                    scope=(
                        "dreamidv.pipeline.generate."
                        f"clip_{request.get('clip_index', 'unknown')}"
                    ),
                    limit=self.args.cprofile_top,
                )
        try:
            with self.profiler.span("dreamidv.video.write", cuda=True):
                if self.args.stream_video_write:
                    self._write_video_streaming(video, output_video)
                else:
                    self.cache_video(
                        tensor=video[None],
                        save_file=str(output_video),
                        fps=self.cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )
        finally:
            del video

    def _write_video_streaming(self, video: Any, output_video: Path) -> None:
        """Codifica por fotograma sin materializar una copia FP32 del clip entero."""

        import imageio.v2 as imageio

        if video is None or getattr(video, "ndim", None) != 4:
            raise RuntimeError("DreamID-V devolvió un tensor de vídeo inválido.")
        writer = imageio.get_writer(
            str(output_video),
            fps=self.cfg.sample_fps,
            codec="libx264",
            quality=8,
        )
        try:
            for frame_index in range(int(video.shape[1])):
                frame = video[:, frame_index]
                frame = (
                    frame.clamp(-1, 1)
                    .add(1)
                    .mul(127.5)
                    .permute(1, 2, 0)
                    .to(dtype=self.torch.uint8, device="cpu", non_blocking=False)
                    .numpy()
                )
                writer.append_data(frame)
        finally:
            writer.close()

    def generate(self, request: dict[str, Any]) -> dict[str, Any]:
        request_id = self.profiler.set_request(request)
        input_video = Path(request["input_video"]).resolve()
        source_reference = Path(request["source_reference"]).resolve()
        output_video = Path(request["output_video"]).resolve()
        pose_path = Path(request["pose_video"]).resolve()
        mask_path = Path(request["mask_video"]).resolve()
        output_video.parent.mkdir(parents=True, exist_ok=True)

        missing = [str(path) for path in (pose_path, mask_path) if not path.is_file()]
        if missing:
            raise RuntimeError("Faltan artefactos DWPose precalculados: " + ", ".join(missing))

        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
        self.profiler.gpu_snapshot("clip_start")
        started = time.perf_counter()
        stop_heartbeat = threading.Event()
        heartbeat = self._start_heartbeat(
            stop_heartbeat,
            started,
            stage="clip_generate",
            include_gpu=True,
        )
        requested_offload = bool(request["offload_model"])
        actual_offload = requested_offload
        oom_fallback_used = False
        try:
            try:
                self._generate_once(
                    request,
                    input_video=input_video,
                    source_reference=source_reference,
                    output_video=output_video,
                    pose_path=pose_path,
                    mask_path=mask_path,
                    offload_model=requested_offload,
                )
            except self.torch.cuda.OutOfMemoryError as exc:
                if requested_offload or not self.args.offload_fallback:
                    raise
                oom_fallback_used = True
                actual_offload = True
                self.profiler.log(
                    "WARNING",
                    "CUDA OOM con el modelo residente; se reintenta el clip con offload_model=true",
                    error=str(exc),
                    **self.profiler.gpu_memory(),
                )
                try:
                    output_video.unlink(missing_ok=True)
                except OSError:
                    pass
                self._drop_gpu_caches(disable_persistent_caches=True)
                self._cleanup_cuda(force=True, offload_model=True)
                self._generate_once(
                    request,
                    input_video=input_video,
                    source_reference=source_reference,
                    output_video=output_video,
                    pose_path=pose_path,
                    mask_path=mask_path,
                    offload_model=True,
                )
        finally:
            self.profiler.flush_cuda()
            stop_heartbeat.set()
            heartbeat.join(timeout=2.0)

        cleanup = self._cleanup_cuda(force=False, offload_model=actual_offload)
        elapsed = time.perf_counter() - started
        self.profiler.gpu_snapshot("clip_end")
        optimization_metrics: dict[str, Any] = {}
        if self._invariant_optimizer is not None:
            optimization_metrics["model_invariants"] = dict(
                self._invariant_optimizer.summary()
            )
        if self._scheduler_torch_proxy is not None:
            optimization_metrics["scheduler_tensors"] = dict(
                self._scheduler_torch_proxy.summary()
            )
        summary = self.profiler.summary(
            request_id=request_id,
            elapsed_seconds=elapsed,
            requested_offload_model=requested_offload,
            actual_offload_model=actual_offload,
            oom_fallback_used=oom_fallback_used,
            cuda_cleanup=cleanup,
            optimization_metrics=optimization_metrics,
        )
        self.profiler.clear_request()
        return summary

    def _start_heartbeat(
        self,
        stop_event: threading.Event,
        started: float,
        *,
        stage: str,
        include_gpu: bool,
    ) -> threading.Thread:
        interval = max(0.0, float(self.args.heartbeat_seconds))

        def run() -> None:
            if interval <= 0.0 or not self.profiler.enabled:
                return
            sequence = 0
            while not stop_event.wait(interval):
                sequence += 1
                metrics = self.profiler.gpu_memory() if include_gpu else {}
                self.profiler.event(
                    "dreamidv.heartbeat",
                    stage=stage,
                    sequence=sequence,
                    elapsed_seconds=time.perf_counter() - started,
                    **metrics,
                )

        thread = threading.Thread(target=run, name="dreamidv-heartbeat", daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        self.profiler.flush_cuda()
        self.profiler.gpu_snapshot("worker_shutdown")
        self.profiler.event("dreamidv.worker_end")


def _reply(payload: dict[str, Any]) -> None:
    _emit(RESULT_PREFIX, payload)


def main() -> int:
    args = _arguments()
    os.chdir(args.repository)
    runtime: WorkerRuntime | None = None
    try:
        runtime = WorkerRuntime(args)
    except Exception as exc:  # noqa: BLE001 - reporta al proceso padre
        _emit(
            LOG_PREFIX,
            {
                "timestamp": _utc_now(),
                "process_id": os.getpid(),
                "source": "dreamidv_worker",
                "level": "ERROR",
                "message": "Falló la inicialización de DreamID-V",
                "exception_type": type(exc).__name__,
                "exception": traceback.format_exc(),
            },
        )
        _reply({"ok": False, "error": f"Inicialización: {type(exc).__name__}: {exc}"})
        traceback.print_exc()
        return 1

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw)
            if request.get("command") == "shutdown":
                runtime.close()
                _reply({"ok": True, "shutdown": True})
                return 0
            metrics = runtime.generate(request)
            _reply(
                {
                    "ok": True,
                    "output_video": request["output_video"],
                    "metrics": metrics,
                }
            )
        except Exception as exc:  # noqa: BLE001 - protocolo de proceso externo
            try:
                runtime.profiler.flush_cuda()
                runtime._cleanup_cuda(force=True, offload_model=True)
            except Exception:
                pass
            runtime.profiler.log(
                "ERROR",
                "Falló el procesamiento de un clip DreamID-V",
                exception_type=type(exc).__name__,
                exception=traceback.format_exc(),
                request_id=request.get("request_id"),
                **runtime.profiler.gpu_memory(),
            )
            traceback.print_exc()
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            runtime.profiler.clear_request()
    runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
