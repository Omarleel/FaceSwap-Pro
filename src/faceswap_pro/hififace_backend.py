from __future__ import annotations

import importlib
import importlib.util
import sys
import threading
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from .alignment import align_face
from .insightface_backend import create_insightface_analysis_services
from .modeling import FaceData, ModelBundle, ModelCapabilities, SwapResult

BACKEND_NAME = "hififace_3dmm"


class HifiFaceRuntime(Protocol):
    """Contrato mínimo del generador 3DMM.

    El runtime recibe rostros ya alineados en BGR y devuelve una imagen BGR junto
    con una máscara opcional. De esta forma el pipeline no depende de PyTorch ni de
    la estructura interna de un repositorio concreto.
    """

    @property
    def output_size(self) -> int: ...

    def infer(
        self,
        source_bgr: np.ndarray,
        target_bgr: np.ndarray,
        *,
        shape_rate: float,
        identity_rate: float,
    ) -> tuple[np.ndarray, np.ndarray | None]: ...


class XuehyHifiFaceRuntime:
    """Adaptador diferido para la implementación MIT ``xuehy/HiFiFace-pytorch``.

    El código de terceros no se copia dentro del proyecto. Se carga desde una ruta
    externa para mantener límites de licencia y permitir actualizar el runtime sin
    modificar el dominio de FaceSwap-Pro.
    """

    def __init__(
        self,
        *,
        repository_path: Path,
        checkpoint_directory: Path,
        checkpoint_iteration: int | None,
        f_3d_checkpoint_path: Path,
        f_id_checkpoint_path: Path,
        bfm_folder: Path,
        hrnet_path: Path | None,
        device: str,
        use_fp16: bool,
        output_size: int = 256,
    ) -> None:
        self._repository_path = repository_path.resolve()
        self._checkpoint_directory = checkpoint_directory.resolve()
        self._checkpoint_iteration = checkpoint_iteration
        self._device_name = device
        self._use_fp16 = bool(use_fp16)
        self._output_size = int(output_size)
        if self._output_size != 256:
            raise ValueError(
                "La implementación xuehy/HiFiFace-pytorch incluida en este adaptador "
                "usa checkpoints nativos de 256×256; output_size debe ser 256."
            )
        self._lock = threading.Lock()

        if not self._repository_path.is_dir():
            raise FileNotFoundError(
                f"No existe el repositorio externo de HifiFace: {self._repository_path}"
            )
        if not self._checkpoint_directory.is_dir():
            raise FileNotFoundError(
                f"No existe el directorio del checkpoint HifiFace: "
                f"{self._checkpoint_directory}"
            )
        generator_name = (
            "generator.pth"
            if self._checkpoint_iteration is None
            else f"generator_{self._checkpoint_iteration}.pth"
        )
        generator_checkpoint = self._checkpoint_directory / generator_name
        if not generator_checkpoint.is_file():
            raise FileNotFoundError(
                f"No existe el checkpoint generador esperado: {generator_checkpoint}"
            )
        for required in (f_3d_checkpoint_path, f_id_checkpoint_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        if not bfm_folder.is_dir():
            raise FileNotFoundError(bfm_folder)
        if hrnet_path is not None and not hrnet_path.is_file():
            raise FileNotFoundError(hrnet_path)

        self._prepare_import_path()
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "HifiFace requiere PyTorch. Instala el extra hififace3d y una build "
                "CUDA de PyTorch compatible con tu sistema."
            ) from exc

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Se solicitó {device}, pero torch.cuda.is_available() es False."
            )

        # El repositorio importa LPIPS incluso para inferencia y su constructor
        # intenta preparar VGG. HifiFace no usa esa pérdida con is_training=False,
        # por lo que instalamos un shim antes de importar ``models.model``.
        class _InferenceOnlyLPIPS(torch.nn.Module):
            def forward(self, first, second):
                batch = int(first.shape[0])
                return torch.zeros((batch, 1, 1, 1), device=first.device)

        lpips_module = sys.modules.get("lpips")
        if lpips_module is None:
            lpips_module = types.ModuleType("lpips")
            sys.modules["lpips"] = lpips_module
        lpips_module.LPIPS = lambda *args, **kwargs: _InferenceOnlyLPIPS()

        try:
            config_module = importlib.import_module("configs.train_config")
            model_module = importlib.import_module("models.model")
        except Exception as exc:
            raise RuntimeError(
                "No se pudo importar la implementación externa de HifiFace. "
                "Comprueba repository_path y sus dependencias."
            ) from exc

        TrainConfig = getattr(config_module, "TrainConfig")
        HifiFace = getattr(model_module, "HifiFace")

        # Algunas revisiones del repositorio solo inicializan ``use_ddp`` durante
        # entrenamiento, pero ``setup`` lo consulta también en inferencia.
        class _InferenceHifiFace(HifiFace):
            def setup(self, selected_device):
                self.use_ddp = False
                return super().setup(selected_device)

        options = TrainConfig()
        options.use_ddp = False
        # Estas opciones solo controlan pérdidas de entrenamiento; se desactivan para
        # evitar cargar HRNet cuando el checkpoint no lo necesita en inferencia.
        options.eye_hm_loss = False
        options.mouth_hm_loss = False
        identity_config = dict(options.identity_extractor_config)
        identity_config.update(
            {
                "f_3d_checkpoint_path": str(f_3d_checkpoint_path.resolve()),
                "f_id_checkpoint_path": str(f_id_checkpoint_path.resolve()),
                "bfm_folder": str(bfm_folder.resolve()),
            }
        )
        if hrnet_path is not None:
            identity_config["hrnet_path"] = str(hrnet_path.resolve())

        checkpoint = (str(self._checkpoint_directory), self._checkpoint_iteration)
        try:
            model = _InferenceHifiFace(
                identity_config,
                is_training=False,
                device=device,
                load_checkpoint=checkpoint,
            )
            model.eval()
        except Exception as exc:
            raise RuntimeError(
                "HifiFace no pudo cargar el generador o los modelos 3DMM auxiliares."
            ) from exc

        self._torch = torch
        self._model = model
        self._device = torch.device(device)

    @property
    def output_size(self) -> int:
        return self._output_size

    def _prepare_import_path(self) -> None:
        root = str(self._repository_path)
        if root not in sys.path:
            sys.path.insert(0, root)

        # Los nombres absolutos ``models`` y ``configs`` usados por el proyecto de
        # terceros son genéricos. Si ya apuntan a otro paquete, fallar es más seguro
        # que ejecutar silenciosamente código incorrecto.
        for package in ("models", "configs"):
            spec = importlib.util.find_spec(package)
            if spec is None or spec.origin is None:
                continue
            origin = Path(spec.origin).resolve()
            if self._repository_path not in origin.parents:
                raise RuntimeError(
                    f"El módulo global {package!r} ya apunta a {origin}; inicia "
                    "FaceSwap-Pro en un entorno limpio para cargar HifiFace."
                )

    def _to_tensor(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (self._output_size, self._output_size),
            interpolation=cv2.INTER_CUBIC,
        )
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
        return tensor.unsqueeze(0).to(self._device, dtype=self._torch.float32) / 255.0

    def infer(
        self,
        source_bgr: np.ndarray,
        target_bgr: np.ndarray,
        *,
        shape_rate: float,
        identity_rate: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        torch = self._torch
        source = self._to_tensor(source_bgr)
        target = self._to_tensor(target_bgr)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self._use_fp16 and self._device.type == "cuda"
            else nullcontext()
        )
        with self._lock, torch.inference_mode(), autocast:
            output = self._model.forward(
                source,
                target,
                shape_rate=float(shape_rate),
                id_rate=float(identity_rate),
            )

        if isinstance(output, (tuple, list)):
            generated = output[0]
            predicted_mask = output[1] if len(output) > 1 else None
        else:
            generated = output
            predicted_mask = None

        generated = generated.detach().float().clamp(0.0, 1.0)[0]
        rgb = (generated.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        mask = None
        if predicted_mask is not None:
            predicted_mask = predicted_mask.detach().float().clamp(0.0, 1.0)
            while predicted_mask.ndim > 2:
                predicted_mask = predicted_mask[0]
            mask = predicted_mask.cpu().numpy().astype(np.float32)
            mask = cv2.resize(
                mask,
                (self._output_size, self._output_size),
                interpolation=cv2.INTER_LINEAR,
            )[..., None]
        return np.ascontiguousarray(bgr), mask


class HifiFace3DMMSwapper:
    def __init__(
        self,
        runtime: HifiFaceRuntime,
        *,
        shape_rate: float = 1.0,
        identity_rate: float = 1.0,
        mask_dilate_ratio: float = 0.025,
        mask_blur_ratio: float = 0.018,
        iterations: int = 1,
    ) -> None:
        self._runtime = runtime
        self._shape_rate = float(np.clip(shape_rate, 0.0, 1.5))
        self._identity_rate = float(np.clip(identity_rate, 0.0, 1.5))
        self._mask_dilate_ratio = max(0.0, float(mask_dilate_ratio))
        self._mask_blur_ratio = max(0.0, float(mask_blur_ratio))
        self._iterations = max(1, min(4, int(iterations)))
        self._source_cache_key: tuple[int, int] | None = None
        self._source_crop: np.ndarray | None = None

    def _aligned_source(self, source_face: FaceData) -> np.ndarray:
        if source_face.reference_image is None:
            raise RuntimeError(
                "HifiFace necesita la imagen de referencia del rostro de origen."
            )
        key = (id(source_face.reference_image), self._runtime.output_size)
        if self._source_crop is None or self._source_cache_key != key:
            self._source_crop, _ = align_face(
                source_face.reference_image,
                source_face.kps,
                self._runtime.output_size,
                template="hififace",
            )
            self._source_cache_key = key
        return self._source_crop

    def _refine_mask(self, mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        result = np.asarray(mask, dtype=np.float32)
        if result.ndim == 2:
            result = result[..., None]
        size = min(result.shape[:2])
        dilate = int(round(size * self._mask_dilate_ratio))
        if dilate > 0:
            kernel_size = dilate * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            )
            # i_r ya contiene la fusión semántica interna de HifiFace. Dilatar
            # garantiza que el blend exterior no vuelva a recortar esa transición.
            result = cv2.dilate(result[..., 0], kernel)[..., None]
        blur = int(round(size * self._mask_blur_ratio))
        if blur > 0:
            blur = blur * 2 + 1
            result = cv2.GaussianBlur(result[..., 0], (blur, blur), 0)[..., None]
        return np.clip(result, 0.0, 1.0)

    def swap(
        self,
        frame: np.ndarray,
        target_face: FaceData,
        source_face: FaceData,
    ) -> SwapResult:
        source_crop = self._aligned_source(source_face)
        target_crop, affine = align_face(
            frame,
            target_face.kps,
            self._runtime.output_size,
            template="hififace",
        )
        generated = target_crop
        mask = None
        for _ in range(self._iterations):
            generated, mask = self._runtime.infer(
                source_crop,
                generated,
                shape_rate=self._shape_rate,
                identity_rate=self._identity_rate,
            )
        if generated.shape[:2] != (self._runtime.output_size, self._runtime.output_size):
            generated = cv2.resize(
                generated,
                (self._runtime.output_size, self._runtime.output_size),
                interpolation=cv2.INTER_CUBIC,
            )
        return SwapResult(
            crop=np.ascontiguousarray(generated),
            affine=affine,
            mask=self._refine_mask(mask),
            mask_mode="replace" if mask is not None else "multiply",
            metadata={
                "generator": "hififace",
                "geometry_conditioning": "3dmm_internal",
                "native_output_size": self._runtime.output_size,
                "shape_rate": self._shape_rate,
                "identity_rate": self._identity_rate,
                "iterations": self._iterations,
                "semantic_fusion_internal": True,
            },
        )


def _path_option(options: dict[str, Any], name: str, *, required: bool = True) -> Path | None:
    value = options.get(name)
    if value in (None, ""):
        if required:
            raise ValueError(f"Falta engine.options.{name} para el backend {BACKEND_NAME}.")
        return None
    return Path(str(value))


class HifiFace3DMMBackendFactory:
    def create(self, config: Any, model_path: Path) -> ModelBundle:
        options = dict(config.engine.options)
        services = create_insightface_analysis_services(config)

        # ``model_path`` es el único origen de verdad para el checkpoint principal.
        # La CLI lo resuelve desde --model-path o engine.options.model_path.
        checkpoint_directory = Path(model_path)
        checkpoint_iteration_value = options.get("checkpoint_iteration")
        checkpoint_iteration = (
            None
            if checkpoint_iteration_value in (None, "")
            else int(checkpoint_iteration_value)
        )
        repository_path = _path_option(options, "repository_path")
        f_3d_checkpoint = _path_option(options, "f_3d_checkpoint_path")
        f_id_checkpoint = _path_option(options, "f_id_checkpoint_path")
        bfm_folder = _path_option(options, "bfm_folder")
        hrnet_path = _path_option(options, "hrnet_path", required=False)
        device = str(options.get("device", "cuda:0"))

        runtime = XuehyHifiFaceRuntime(
            repository_path=repository_path,
            checkpoint_directory=checkpoint_directory,
            checkpoint_iteration=checkpoint_iteration,
            f_3d_checkpoint_path=f_3d_checkpoint,
            f_id_checkpoint_path=f_id_checkpoint,
            bfm_folder=bfm_folder,
            hrnet_path=hrnet_path,
            device=device,
            use_fp16=bool(options.get("use_fp16", True)),
            output_size=int(options.get("output_size", 256)),
        )
        swapper = HifiFace3DMMSwapper(
            runtime,
            shape_rate=float(options.get("shape_rate", 1.0)),
            identity_rate=float(options.get("identity_rate", 1.0)),
            mask_dilate_ratio=float(options.get("mask_dilate_ratio", 0.025)),
            mask_blur_ratio=float(options.get("mask_blur_ratio", 0.018)),
            iterations=int(options.get("iterations", 1)),
        )

        runtime_info = dict(services.runtime)
        runtime_info.update(
            {
                "generator": "hififace_3dmm",
                "implementation": "xuehy/HiFiFace-pytorch",
                "repository_path": str(repository_path),
                "checkpoint_directory": str(checkpoint_directory),
                "checkpoint_iteration": checkpoint_iteration,
                "device": device,
                "native_output_size": runtime.output_size,
                "geometry_conditioning": "3dmm_internal",
            }
        )
        artifacts = [
            checkpoint_directory,
            f_3d_checkpoint,
            f_id_checkpoint,
            bfm_folder,
        ]
        if hrnet_path is not None:
            artifacts.append(hrnet_path)

        return ModelBundle(
            backend=BACKEND_NAME,
            analyzer=services.analyzer,
            swapper=swapper,
            providers=services.providers,
            runtime=runtime_info,
            capabilities=ModelCapabilities(
                generator="hififace",
                native_output_size=runtime.output_size,
                geometry_conditioning="3dmm_internal",
                geometry_postprocess="learned_semantic_mask",
                temporal_generation="frame_independent",
            ),
            model_artifacts=tuple(artifacts),
        )
