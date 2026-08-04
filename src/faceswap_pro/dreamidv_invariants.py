from __future__ import annotations

"""Optimizaciones invariantes de DreamID-V sin modificar pasos ni pesos.

Las rutas oficiales reconstruyen muchos tensores pequeños y transformaciones
constantes en cada bloque/pasada. Este módulo instala cachés acotadas y un proxy
local de ``torch.tensor`` sobre los módulos upstream. Las operaciones numéricas
principales se mantienen intactas; solo se reutilizan resultados deterministas.
"""

import importlib
import threading
import types
import weakref
from collections import OrderedDict
from typing import Any, Callable, Mapping, Sequence


EventCallback = Callable[..., None]


def _safe_tensor_version(tensor: Any) -> tuple[str, int | None]:
    """Devuelve una versión utilizable sin romper ``inference_mode``.

    Los tensores creados por ``torch.inference_mode`` no tienen contador de
    versión y PyTorch lanza ``RuntimeError`` incluso al leer ``_version``. En
    ese caso las cachés se limitan a identidad viva + metadatos y se vacían al
    terminar cada clip. Para tensores normales se conserva la invalidación por
    mutación mediante el contador de versión.
    """

    try:
        return ("tracked", int(tensor._version))
    except RuntimeError as exc:
        if "Inference tensors do not track version counter" in str(exc):
            return ("inference", None)
        return ("unavailable", None)
    except Exception:
        return ("unavailable", None)


def _tensor_key(tensor: Any) -> tuple[object, ...]:
    try:
        pointer = int(tensor.data_ptr())
    except Exception:
        pointer = id(tensor)
    return (
        pointer,
        tuple(getattr(tensor, "shape", ())),
        str(getattr(tensor, "dtype", None)),
        str(getattr(tensor, "device", None)),
        _safe_tensor_version(tensor),
    )


class OptimizedTorchProxy:
    """Proxy por módulo que acelera ``torch.tensor`` sin tocar torch global."""

    def __init__(
        self,
        torch_module: Any,
        *,
        cache_size: int = 128,
        event: EventCallback | None = None,
        scope: str = "unknown",
    ) -> None:
        self._torch = torch_module
        self._cache_size = max(8, int(cache_size))
        self._cache: OrderedDict[tuple[object, ...], Any] = OrderedDict()
        self._event = event
        self._scope = scope
        self._hits = 0
        self._stacked = 0
        self._empty_cat_elisions = 0
        self._stack_hook: Callable[..., Any | None] | None = None
        self._lock = threading.RLock()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    @staticmethod
    def _device_key(device: Any) -> str:
        return str(device) if device is not None else "default"

    @staticmethod
    def _scalar_sequence(data: Any) -> tuple[Any, ...] | None:
        if not isinstance(data, (list, tuple)) or len(data) > 32:
            return None
        values: list[Any] = []
        for item in data:
            if isinstance(item, (bool, int, float, complex)):
                values.append(item)
                continue
            # Los grid_sizes upstream contienen tensores escalares CPU.
            if hasattr(item, "numel") and int(item.numel()) == 1:
                try:
                    values.append(item.item())
                    continue
                except Exception:
                    return None
            return None
        return tuple(values)

    def cat(self, tensors: Any, *args: Any, **kwargs: Any) -> Any:
        """Evita copias cuando concatenar tensores vacíos no cambia el valor."""

        if isinstance(tensors, (list, tuple)) and tensors:
            nonempty = [item for item in tensors if int(item.numel()) > 0]
            if len(nonempty) == 1 and all(hasattr(item, "numel") for item in tensors):
                candidate = nonempty[0]
                # Verificar la forma esperada para no alterar concatenaciones con
                # dimensiones vacías incompatibles.
                dim = kwargs.get("dim", args[0] if args else 0)
                dim = int(dim) % candidate.ndim
                expected = list(candidate.shape)
                expected[dim] = sum(int(item.shape[dim]) for item in tensors)
                if tuple(expected) == tuple(candidate.shape):
                    self._empty_cat_elisions += 1
                    return candidate
        return self._torch.cat(tensors, *args, **kwargs)

    def stack(self, tensors: Any, *args: Any, **kwargs: Any) -> Any:
        if self._stack_hook is not None:
            replacement = self._stack_hook(tensors, *args, **kwargs)
            if replacement is not None:
                return replacement
        return self._torch.stack(tensors, *args, **kwargs)

    def tensor(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        torch = self._torch
        requires_grad = bool(kwargs.get("requires_grad", False))
        pin_memory = bool(kwargs.get("pin_memory", False))
        device = kwargs.get("device")
        dtype = kwargs.get("dtype")

        # UniPC crea tensores desde listas de escalares CUDA. torch.tensor(...)
        # fuerza una ruta host; stack conserva los valores en el dispositivo.
        if (
            isinstance(data, (list, tuple))
            and data
            and all(isinstance(item, torch.Tensor) and item.numel() == 1 for item in data)
            and not pin_memory
        ):
            result = torch.stack([item.reshape(()) for item in data])
            if device is not None or dtype is not None:
                result = result.to(
                    device=device if device is not None else result.device,
                    dtype=dtype if dtype is not None else result.dtype,
                )
            if requires_grad:
                result = result.detach().requires_grad_(True)
            self._stacked += 1
            return result

        values = self._scalar_sequence(data)
        if values is not None and not requires_grad and not pin_memory:
            key = (
                values,
                str(dtype),
                self._device_key(device),
                tuple(args),
                tuple(
                    sorted(
                        (name, repr(value))
                        for name, value in kwargs.items()
                        if name not in {"device", "dtype"}
                    )
                ),
            )
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return cached
            result = torch.tensor(data, *args, **kwargs)
            with self._lock:
                self._cache[key] = result
                self._cache.move_to_end(key)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return result

        return torch.tensor(data, *args, **kwargs)

    def summary(self) -> Mapping[str, Any]:
        return {
            "scope": self._scope,
            "constant_cache_entries": len(self._cache),
            "constant_cache_hits": self._hits,
            "tensor_lists_stacked": self._stacked,
            "empty_cat_elisions": self._empty_cat_elisions,
        }


class _ForwardLRU:
    """LRU por identidad viva del almacenamiento raíz del tensor.

    Las vistas nuevas de un mismo tensor (por ejemplo ``transpose`` de la
    referencia o ``reshape`` del timestep) pueden reutilizar el resultado. El
    ``weakref`` al tensor raíz evita confundir contenido nuevo cuando el allocator
    CUDA recicla una dirección entre clips.
    """

    def __init__(self, *, max_entries: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self.values: OrderedDict[
            tuple[object, ...], tuple[Any, Any]
        ] = OrderedDict()
        self.hits = 0

    @staticmethod
    def _root(tensor: Any) -> Any:
        root = tensor
        seen: set[int] = set()
        while True:
            base = getattr(root, "_base", None)
            if base is None or id(base) in seen:
                return root
            seen.add(id(root))
            root = base

    @classmethod
    def _key(cls, tensor: Any) -> tuple[Any, tuple[object, ...]]:
        root = cls._root(tensor)
        version = _safe_tensor_version(root)
        owner = root
        # En inference_mode PyTorch no registra relaciones de vista en ``_base``.
        # El almacenamiento sí permanece compartido, por lo que permite que una
        # vista equivalente reutilice la caché sin consultar el contador ausente.
        if version[0] == "inference":
            try:
                owner = tensor.untyped_storage()
            except Exception:
                owner = root
        metadata = (
            tuple(getattr(tensor, "shape", ())),
            tuple(getattr(tensor, "stride", lambda: ())()),
            int(getattr(tensor, "storage_offset", lambda: 0)()),
            str(getattr(tensor, "dtype", None)),
            str(getattr(tensor, "device", None)),
            version,
        )
        return owner, (id(owner), *metadata)

    def get(self, tensor: Any) -> Any | None:
        root, key = self._key(tensor)
        entry = self.values.get(key)
        if entry is None:
            return None
        reference, result = entry
        if reference() is not root:
            self.values.pop(key, None)
            return None
        self.values.move_to_end(key)
        self.hits += 1
        return result

    def put(self, tensor: Any, result: Any) -> None:
        root, key = self._key(tensor)
        try:
            reference = weakref.ref(root)
        except TypeError:
            return
        self.values[key] = (reference, result)
        self.values.move_to_end(key)
        # Eliminar entradas cuyo almacenamiento raíz ya murió para no retener
        # resultados GPU de clips anteriores.
        for stale_key, (stale_ref, _) in list(self.values.items()):
            if stale_ref() is None:
                self.values.pop(stale_key, None)
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)

    def clear(self) -> None:
        self.values.clear()


class DreamIDVInvariantOptimizer:
    """Instala cachés de RoPE, tiempo, referencia y contexto codificado."""

    def __init__(
        self,
        *,
        package_name: str,
        model: Any,
        torch_module: Any,
        attention_dtype: Any,
        event: EventCallback | None = None,
    ) -> None:
        self.package_name = package_name
        self.model = model
        self.torch = torch_module
        self.attention_dtype = attention_dtype
        self.event = event
        self.model_module = importlib.import_module(f"{package_name}.modules.model")
        self.model_torch_proxy = OptimizedTorchProxy(
            torch_module, event=event, scope="dreamidv_model"
        )
        self.model_torch_proxy._stack_hook = self._stack_active_context
        self.model_module.torch = self.model_torch_proxy
        self._rope_grid_cache: OrderedDict[tuple[object, ...], tuple[tuple[int, int, int], ...]] = OrderedDict()
        self._rope_multiplier_cache: OrderedDict[tuple[object, ...], Any] = OrderedDict()
        self._sinusoid_basis_cache: OrderedDict[tuple[object, ...], Any] = OrderedDict()
        self._sinusoid_output_cache = _ForwardLRU(max_entries=32)
        self._encoded_context_cpu: Any | None = None
        self._encoded_context_key: tuple[object, ...] | None = None
        self._active_encoded_context_gpu: Any | None = None
        self._module_caches: dict[str, _ForwardLRU] = {}
        self._cross_attention_cache_count = 0
        self._installed = False

    def _stack_active_context(self, tensors: Any, *args: Any, **kwargs: Any) -> Any | None:
        active = self._active_encoded_context_gpu
        if not isinstance(active, self.torch.Tensor):
            return None
        if not isinstance(tensors, (list, tuple)) or len(tensors) != active.size(0):
            return None
        dim = kwargs.get("dim", args[0] if args else 0)
        if int(dim) != 0:
            return None
        for index, tensor in enumerate(tensors):
            if not isinstance(tensor, self.torch.Tensor):
                return None
            expected = active[index]
            if (
                tensor.shape != expected.shape
                or tensor.dtype != expected.dtype
                or tensor.device != expected.device
                or tensor.stride() != expected.stride()
                or tensor.data_ptr() != expected.data_ptr()
            ):
                return None
        return active

    def install(self) -> bool:
        if self._installed:
            return True
        required = ("rope_apply", "sinusoidal_embedding_1d")
        if not all(callable(getattr(self.model_module, name, None)) for name in required):
            return False
        self._patch_rope_apply()
        self._patch_sinusoidal_embedding()
        self._patch_module_forward(getattr(self.model, "ref_conv", None), "ref_conv", 2)
        self._patch_module_forward(getattr(self.model, "time_embedding", None), "time_embedding", 32)
        self._patch_module_forward(getattr(self.model, "time_projection", None), "time_projection", 32)
        self._patch_cross_attention_context()
        self._patch_text_embedding_identity()
        self._installed = True
        if self.event is not None:
            self.event(
                "dreamidv.optimization",
                optimization="model_invariant_cache",
                enabled=True,
                rope_multiplier_cache=True,
                sinusoidal_cache=True,
                reference_projection_cache=True,
                time_embedding_cache=True,
                preencoded_context=True,
                cross_attention_kv_cache=self._cross_attention_cache_count,
                local_torch_tensor_proxy=True,
            )
        return True

    def install_scheduler_proxy(self, scheduler_module: Any) -> OptimizedTorchProxy:
        proxy = OptimizedTorchProxy(
            self.torch, event=self.event, scope="dreamidv_unipc", cache_size=128
        )
        scheduler_module.torch = proxy
        return proxy

    def clear_clip_caches(self) -> None:
        """Libera resultados GPU dependientes del clip tras la difusión.

        El contexto codificado en CPU permanece porque es constante durante la
        sesión. Las entradas basadas en identidad de tensores de inferencia no
        sobreviven al clip, evitando reutilización accidental entre generaciones.
        """

        self._active_encoded_context_gpu = None
        self._rope_grid_cache.clear()
        self._rope_multiplier_cache.clear()
        self._sinusoid_output_cache.clear()
        for cache in self._module_caches.values():
            cache.clear()

    def _grid_values(self, grid_sizes: Any) -> tuple[tuple[int, int, int], ...]:
        key = _tensor_key(grid_sizes)
        cached = self._rope_grid_cache.get(key)
        if cached is not None:
            self._rope_grid_cache.move_to_end(key)
            return cached
        values = tuple(tuple(int(value) for value in row) for row in grid_sizes.tolist())
        self._rope_grid_cache[key] = values
        self._rope_grid_cache.move_to_end(key)
        while len(self._rope_grid_cache) > 128:
            self._rope_grid_cache.popitem(last=False)
        return values

    def _rope_multiplier(self, freqs: Any, f: int, h: int, w: int, c: int) -> Any:
        key = (*_tensor_key(freqs), f, h, w, c)
        cached = self._rope_multiplier_cache.get(key)
        if cached is not None:
            self._rope_multiplier_cache.move_to_end(key)
            return cached
        parts = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        multiplier = self.torch.cat(
            [
                parts[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        self._rope_multiplier_cache[key] = multiplier
        self._rope_multiplier_cache.move_to_end(key)
        while len(self._rope_multiplier_cache) > 8:
            self._rope_multiplier_cache.popitem(last=False)
        return multiplier

    def _patch_rope_apply(self) -> None:
        optimizer = self
        torch = self.torch

        def optimized_rope_apply(x: Any, grid_sizes: Any, freqs: Any) -> Any:
            n, c = x.size(2), x.size(3) // 2
            output = []
            for index, (f, h, w) in enumerate(optimizer._grid_values(grid_sizes)):
                seq_len = f * h * w
                x_i = torch.view_as_complex(
                    x[index, :seq_len].reshape(seq_len, n, -1, 2)
                )
                multiplier = optimizer._rope_multiplier(freqs, f, h, w, c)
                x_i = torch.view_as_real(x_i * multiplier).flatten(2)
                if seq_len < x.size(1):
                    tail = x[index, seq_len:]
                    if tail.dtype != x_i.dtype:
                        tail = tail.to(dtype=x_i.dtype)
                    x_i = torch.cat([x_i, tail])
                if x_i.dtype != optimizer.attention_dtype:
                    x_i = x_i.to(dtype=optimizer.attention_dtype)
                output.append(x_i)
            if len(output) == 1:
                return output[0].unsqueeze(0)
            return torch.stack(output)

        optimized_rope_apply._faceswap_optimized = True  # type: ignore[attr-defined]
        self.model_module.rope_apply = optimized_rope_apply

    def _patch_sinusoidal_embedding(self) -> None:
        optimizer = self
        torch = self.torch

        def optimized_sinusoidal_embedding_1d(dim: int, position: Any) -> Any:
            if dim % 2 != 0:
                raise AssertionError("dim debe ser par")
            cached = optimizer._sinusoid_output_cache.get(position)
            if cached is not None:
                return cached
            half = dim // 2
            position64 = position.type(torch.float64)
            basis_key = (half, str(position64.device))
            basis = optimizer._sinusoid_basis_cache.get(basis_key)
            if basis is None:
                basis = torch.pow(
                    10000,
                    -torch.arange(half, device=position64.device, dtype=torch.float64).div(half),
                )
                optimizer._sinusoid_basis_cache[basis_key] = basis
                while len(optimizer._sinusoid_basis_cache) > 8:
                    optimizer._sinusoid_basis_cache.popitem(last=False)
            sinusoid = torch.outer(position64, basis)
            result = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
            optimizer._sinusoid_output_cache.put(position, result)
            return result

        optimized_sinusoidal_embedding_1d._faceswap_optimized = True  # type: ignore[attr-defined]
        self.model_module.sinusoidal_embedding_1d = optimized_sinusoidal_embedding_1d

    def _patch_module_forward(self, module: Any, name: str, max_entries: int) -> None:
        if module is None or not callable(getattr(module, "forward", None)):
            return
        original = module.forward
        if getattr(original, "_faceswap_invariant_cached", False):
            return
        cache = _ForwardLRU(max_entries=max_entries)
        self._module_caches[name] = cache

        def wrapped(module_self: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            if args or kwargs or not isinstance(value, self.torch.Tensor):
                return original(value, *args, **kwargs)
            cached = cache.get(value)
            if cached is not None:
                return cached
            result = original(value)
            cache.put(value, result)
            return result

        wrapped._faceswap_invariant_cached = True  # type: ignore[attr-defined]
        module.forward = types.MethodType(wrapped, module)

    def _patch_cross_attention_context(self) -> None:
        """Cachea K/V del contexto fijo una vez por bloque y por clip.

        Cada bloque tiene pesos distintos, por lo que conserva su propia caché.
        Solo se parchean las ramas de cross-attention; Q y toda la autoatención
        siguen calculándose normalmente porque dependen del latente de cada paso.
        """

        blocks = getattr(self.model, "blocks", ())
        for index, block in enumerate(blocks):
            cross = getattr(block, "cross_attn", None)
            if cross is None:
                continue
            installed = 0
            for attribute in ("k", "v", "norm_k"):
                module = getattr(cross, attribute, None)
                if module is None or not callable(getattr(module, "forward", None)):
                    continue
                name = f"cross_attention.{index}.{attribute}"
                self._patch_module_forward(module, name, 2)
                installed += 1
            if installed:
                self._cross_attention_cache_count += installed

    def _patch_text_embedding_identity(self) -> None:
        module = getattr(self.model, "text_embedding", None)
        if module is None or not callable(getattr(module, "forward", None)):
            return
        original = module.forward
        model_dim = int(getattr(self.model, "dim", -1))
        text_len = int(getattr(self.model, "text_len", -1))

        def wrapped(module_self: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
            if (
                isinstance(value, self.torch.Tensor)
                and value.ndim == 3
                and value.size(-1) == model_dim
                and (text_len <= 0 or value.size(-2) == text_len)
            ):
                active = self._active_encoded_context_gpu
                if (
                    isinstance(active, self.torch.Tensor)
                    and active.shape == value.shape
                    and active.device == value.device
                    and active.dtype == value.dtype
                ):
                    return active
                return value
            return original(value, *args, **kwargs)

        wrapped._faceswap_preencoded_context = True  # type: ignore[attr-defined]
        module.forward = types.MethodType(wrapped, module)

    def encode_context(self, context: Sequence[Any], *, device: Any) -> list[Any]:
        """Codifica el contexto fijo una vez y lo guarda en CPU entre clips."""

        if not context:
            return []
        key = tuple(_tensor_key(tensor) for tensor in context)
        if self._encoded_context_cpu is None or self._encoded_context_key != key:
            text_len = int(getattr(self.model, "text_len", 512))
            padded = self.torch.stack(
                [
                    self.torch.cat(
                        [
                            tensor.to(device=device, non_blocking=True),
                            tensor.new_zeros(max(0, text_len - tensor.size(0)), tensor.size(1)).to(
                                device=device, non_blocking=True
                            ),
                        ]
                    )[:text_len]
                    for tensor in context
                ]
            )
            encoded = self.model.text_embedding(padded)
            self._encoded_context_cpu = encoded.detach().to(device="cpu")
            self._encoded_context_key = key
        encoded_gpu = self._encoded_context_cpu.to(device=device, non_blocking=True)
        self._active_encoded_context_gpu = encoded_gpu
        return [item for item in encoded_gpu]

    def summary(self) -> Mapping[str, Any]:
        return {
            "rope_grid_entries": len(self._rope_grid_cache),
            "rope_multiplier_entries": len(self._rope_multiplier_cache),
            "sinusoid_entries": len(self._sinusoid_output_cache.values),
            "encoded_context_cached": self._encoded_context_cpu is not None,
            "cross_attention_cache_modules": self._cross_attention_cache_count,
            "module_cache_hits": {
                name: cache.hits for name, cache in self._module_caches.items()
            },
            "model_tensor_proxy": dict(self.model_torch_proxy.summary()),
        }
