from __future__ import annotations

"""SDPA nativo y diagnosticable para DreamID-V.

DreamID-V entrega Q/K/V como ``[B, L, H, D]`` y longitudes reales por lote.
La implementación upstream descartaba esas longitudes en el fallback de PyTorch,
lo que podía mezclar padding con contenido y además impedir una selección clara
del kernel. Este módulo:

* conserva las semánticas de q_lens/k_lens sin crear una máscara LxS gigante;
* ofrece una máscara booleana compacta ``[B, 1, 1, S]`` cuando se solicita;
* prueba backends SDPA de uno en uno, por lo que el backend informado es el que
  realmente ejecutó la operación;
* deja MATH como último recurso y permite desactivarlo para fallar rápido;
* aplica q_scale y softmax_scale, omitidos por el fallback upstream.
"""

import os
import sys
import threading
import warnings
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - compatibilidad con PyTorch antiguo
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

__all__ = [
    "attention",
    "flash_attention",
    "install_attention_override",
    "sdpa_runtime_summary",
]

_HALF_DTYPES = (torch.float16, torch.bfloat16)
_BACKEND_CACHE: dict[tuple[object, ...], str] = {}
_REPORTED: set[tuple[object, ...]] = set()
_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _Settings:
    priority: tuple[str, ...]
    allow_math: bool
    diagnostics: bool
    padding_mode: str


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "si", "sí"}


def _settings() -> _Settings:
    raw = os.environ.get(
        "FACESWAP_SDPA_BACKENDS",
        "cudnn,flash,efficient,math" if os.name == "nt" else "flash,cudnn,efficient,math",
    )
    aliases = {
        "flash_attention": "flash",
        "flash-attention": "flash",
        "mem_efficient": "efficient",
        "memory_efficient": "efficient",
        "efficient_attention": "efficient",
        "cudnn_attention": "cudnn",
    }
    priority: list[str] = []
    for item in raw.split(","):
        name = aliases.get(item.strip().lower(), item.strip().lower())
        if name in {"flash", "cudnn", "efficient", "math"} and name not in priority:
            priority.append(name)
    allow_math = _env_bool("FACESWAP_SDPA_ALLOW_MATH", False)
    if not allow_math:
        priority = [name for name in priority if name != "math"]
    elif "math" not in priority:
        priority.append("math")
    if not priority:
        raise RuntimeError("FACESWAP_SDPA_BACKENDS no contiene ningún backend utilizable.")

    padding_mode = os.environ.get("FACESWAP_SDPA_PADDING_MODE", "ragged").strip().lower()
    if padding_mode not in {"ragged", "mask"}:
        raise RuntimeError("FACESWAP_SDPA_PADDING_MODE debe ser 'ragged' o 'mask'.")
    return _Settings(
        priority=tuple(priority),
        allow_math=allow_math,
        diagnostics=_env_bool("FACESWAP_SDPA_DIAGNOSTICS", True),
        padding_mode=padding_mode,
    )


def _backend_enum(name: str):
    if SDPBackend is None:
        return None
    attribute = {
        "flash": "FLASH_ATTENTION",
        "cudnn": "CUDNN_ATTENTION",
        "efficient": "EFFICIENT_ATTENTION",
        "math": "MATH",
    }[name]
    return getattr(SDPBackend, attribute, None)


def _normalize_lengths(
    lengths: torch.Tensor | Sequence[int] | None,
    *,
    batch: int,
    maximum: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    del device
    # Las longitudes son metadatos diminutos. Mantenerlas en CPU evita una
    # sincronización CUDA por bloque de atención al inspeccionarlas.
    if lengths is None:
        return torch.full((batch,), maximum, dtype=torch.long)
    if isinstance(lengths, torch.Tensor):
        result = lengths.detach().to(device="cpu", dtype=torch.long).flatten()
    else:
        result = torch.as_tensor(lengths, dtype=torch.long).flatten()
    if result.numel() == 1 and batch > 1:
        result = result.expand(batch)
    if result.numel() != batch:
        raise ValueError(f"Se esperaban {batch} longitudes y se recibieron {result.numel()}.")
    return result.clamp_(min=0, max=maximum)


def _build_padding_mask(
    k_lens: torch.Tensor, maximum: int, *, device: torch.device
) -> torch.Tensor | None:
    """Máscara SDPA compacta; True significa que la clave participa."""

    if bool(torch.all(k_lens == maximum)):
        return None
    positions = torch.arange(maximum, device=device)
    lengths = k_lens.to(device=device, non_blocking=True)
    return (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(1)


def _zero_padded_queries(output: torch.Tensor, q_lens: torch.Tensor) -> torch.Tensor:
    length = output.size(-2)
    if bool(torch.all(q_lens == length)):
        return output
    positions = torch.arange(length, device=output.device)
    lengths = q_lens.to(device=output.device, non_blocking=True)
    valid = (positions.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).unsqueeze(-1)
    return output * valid.to(dtype=output.dtype)


def _call_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    causal: bool,
    softmax_scale: float | None,
    backend: str,
) -> torch.Tensor:
    enum = _backend_enum(backend)
    if enum is None or sdpa_kernel is None:
        if backend != "math":
            raise RuntimeError(f"PyTorch no expone el backend SDPA {backend}.")
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )

    with sdpa_kernel(enum):
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=causal,
            scale=softmax_scale,
        )


def _signature(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    has_mask: bool,
    causal: bool,
    priority: tuple[str, ...],
) -> tuple[object, ...]:
    return (
        q.device.type,
        q.device.index,
        str(q.dtype),
        q.size(-1),
        q.size(-2),
        k.size(-2),
        has_mask,
        causal,
        priority,
    )


def _short_error(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return text[:320] if text else type(exc).__name__


def _report_backend(
    *,
    signature: tuple[object, ...],
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: torch.Tensor | None,
    rejected: Sequence[str],
    settings: _Settings,
) -> None:
    if not settings.diagnostics:
        return
    with _CACHE_LOCK:
        if signature in _REPORTED:
            return
        _REPORTED.add(signature)
    gpu = "cpu"
    if q.is_cuda:
        try:
            gpu = torch.cuda.get_device_name(q.device)
        except Exception:  # pragma: no cover - diagnóstico de mejor esfuerzo
            gpu = f"cuda:{q.device.index}"
    mask = "none" if attn_mask is None else f"{tuple(attn_mask.shape)}"
    print(
        "FaceSwap-Pro SDPA: "
        f"backend={backend.upper()}, torch={torch.__version__}, gpu={gpu}, "
        f"dtype={q.dtype}, q={tuple(q.shape)}, k={tuple(k.shape)}, mask={mask}",
        flush=True,
    )
    if rejected:
        print("FaceSwap-Pro SDPA descartados: " + " | ".join(rejected), flush=True)


def _run_selected_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    causal: bool,
    softmax_scale: float | None,
    settings: _Settings,
) -> torch.Tensor:
    signature = _signature(
        q,
        k,
        has_mask=attn_mask is not None,
        causal=causal,
        priority=settings.priority,
    )
    with _CACHE_LOCK:
        cached = _BACKEND_CACHE.get(signature)
    if cached is not None:
        try:
            return _call_sdpa(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                causal=causal,
                softmax_scale=softmax_scale,
                backend=cached,
            )
        except (RuntimeError, NotImplementedError):
            with _CACHE_LOCK:
                _BACKEND_CACHE.pop(signature, None)

    rejected: list[str] = []
    for backend in settings.priority:
        if backend == "math" and not settings.allow_math:
            continue
        try:
            # Los backends no compatibles suelen emitir warnings antes de lanzar.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                output = _call_sdpa(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    causal=causal,
                    softmax_scale=softmax_scale,
                    backend=backend,
                )
            with _CACHE_LOCK:
                _BACKEND_CACHE[signature] = backend
            warning_text = "; ".join(_short_error(item.message) for item in caught)
            if warning_text:
                rejected.append(f"{backend}: {warning_text}")
            _report_backend(
                signature=signature,
                backend=backend,
                q=q,
                k=k,
                attn_mask=attn_mask,
                rejected=rejected,
                settings=settings,
            )
            return output
        except (RuntimeError, NotImplementedError, ValueError) as exc:
            rejected.append(f"{backend}: {_short_error(exc)}")

    detail = " | ".join(rejected) or "sin diagnóstico del runtime"
    math_hint = (
        " El fallback MATH está desactivado para evitar ejecuciones de muchas horas."
        if not settings.allow_math
        else ""
    )
    raise RuntimeError(
        "Ningún kernel SDPA fusionado pudo ejecutar DreamID-V. "
        f"Backends probados: {', '.join(settings.priority)}. {detail}.{math_hint}"
    )


def _native_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    q_lens: torch.Tensor | Sequence[int] | None,
    k_lens: torch.Tensor | Sequence[int] | None,
    dropout_p: float,
    softmax_scale: float | None,
    q_scale: float | None,
    causal: bool,
    window_size: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q, K y V deben tener forma [B, L, H, D].")
    if q.size(0) != k.size(0) or k.size(0) != v.size(0):
        raise ValueError("Q, K y V deben compartir el tamaño de lote.")
    if k.size(1) != v.size(1):
        raise ValueError("K y V deben compartir la longitud de secuencia.")
    if window_size != (-1, -1):
        raise RuntimeError(
            "La ruta SDPA nativa optimizada solo admite atención global; "
            "DreamID-V Faster usa window_size=(-1, -1)."
        )

    settings = _settings()
    output_dtype = q.dtype
    target_dtype = q.dtype if q.dtype in _HALF_DTYPES else dtype
    if target_dtype not in _HALF_DTYPES:
        target_dtype = torch.bfloat16

    # SDPA espera [B, H, L, D]. Los contiguous evitan layouts que excluyen kernels.
    qh = q.transpose(1, 2).to(dtype=target_dtype).contiguous()
    kh = k.transpose(1, 2).to(dtype=target_dtype).contiguous()
    vh = v.transpose(1, 2).to(dtype=target_dtype).contiguous()
    if q_scale is not None:
        qh = qh * float(q_scale)

    batch, _, q_max, _ = qh.shape
    k_max = kh.size(-2)
    q_lengths = _normalize_lengths(q_lens, batch=batch, maximum=q_max)
    k_lengths = _normalize_lengths(k_lens, batch=batch, maximum=k_max)

    # Caso habitual de DreamID-V (B=1): recortar padding mantiene la semántica
    # exacta y deja a cuDNN/Flash trabajar sin una máscara densa.
    q_unique = torch.unique(q_lengths)
    k_unique = torch.unique(k_lengths)
    if q_unique.numel() == 1 and k_unique.numel() == 1:
        q_valid = int(q_unique.item())
        k_valid = int(k_unique.item())
        if q_valid == 0 or k_valid == 0:
            return torch.zeros_like(q).to(dtype=output_dtype)
        core = _run_selected_sdpa(
            qh[..., :q_valid, :],
            kh[..., :k_valid, :],
            vh[..., :k_valid, :],
            attn_mask=None,
            dropout_p=dropout_p,
            causal=causal,
            softmax_scale=softmax_scale,
            settings=settings,
        )
        if q_valid != q_max:
            padded = torch.zeros(
                (*core.shape[:-2], q_max, core.size(-1)),
                device=core.device,
                dtype=core.dtype,
            )
            padded[..., :q_valid, :] = core
            core = padded
        return core.transpose(1, 2).contiguous().to(dtype=output_dtype)

    if settings.padding_mode == "ragged" or causal:
        # Variable-length real: ejecutar cada muestra recortada evita una máscara
        # [B,H,L,S] y conserva la posibilidad de kernel fusionado.
        output = torch.zeros(
            (batch, qh.size(1), q_max, vh.size(-1)),
            device=qh.device,
            dtype=target_dtype,
        )
        for index in range(batch):
            q_valid = int(q_lengths[index].item())
            k_valid = int(k_lengths[index].item())
            if q_valid == 0 or k_valid == 0:
                continue
            sample = _run_selected_sdpa(
                qh[index : index + 1, :, :q_valid, :],
                kh[index : index + 1, :, :k_valid, :],
                vh[index : index + 1, :, :k_valid, :],
                attn_mask=None,
                dropout_p=dropout_p,
                causal=causal,
                softmax_scale=softmax_scale,
                settings=settings,
            )
            output[index : index + 1, :, :q_valid, :] = sample
    else:
        # Alternativa compacta y correcta: True = participa en SDPA.
        attn_mask = _build_padding_mask(k_lengths, k_max, device=kh.device)
        output = _run_selected_sdpa(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            causal=False,
            softmax_scale=softmax_scale,
            settings=settings,
        )
        output = _zero_padded_queries(output, q_lengths)

    return output.transpose(1, 2).contiguous().to(dtype=output_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    del deterministic, fa_version
    return _native_attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        dropout_p=float(dropout_p),
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=bool(causal),
        window_size=tuple(window_size),
        dtype=dtype,
    )


def flash_attention(*args, **kwargs):
    """Compatibilidad con model.py upstream, sin depender del paquete flash-attn."""

    return attention(*args, **kwargs)


def install_attention_override(package_name: str) -> None:
    """Hace que imports relativos de DreamID-V usen este módulo vendorizado."""

    target = f"{package_name}.modules.attention"
    sys.modules[target] = sys.modules[__name__]


def sdpa_runtime_summary() -> str:
    settings = _settings()
    return (
        f"priority={','.join(settings.priority)}, "
        f"math={'on' if settings.allow_math else 'off'}, "
        f"padding={settings.padding_mode}, diagnostics={'on' if settings.diagnostics else 'off'}"
    )


def _reset_backend_cache_for_tests() -> None:
    with _CACHE_LOCK:
        _BACKEND_CACHE.clear()
        _REPORTED.clear()
