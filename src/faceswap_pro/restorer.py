from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .modeling import FaceRestorer


class IdentityFaceRestorer:
    """Implementación nula; evita condicionales en el blender."""

    enabled = False

    def restore(self, bgr: np.ndarray) -> np.ndarray:
        return bgr

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        return self.restore(bgr)


class OnnxFaceRestorer:
    """Adaptador ONNX para restauradores RGB BCHW de una entrada y una salida."""

    enabled = True

    def __init__(
        self,
        model_path: Path,
        input_size: int,
        output_range: str,
        providers: tuple[Any, ...] | list[Any],
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Restaurador ONNX no encontrado: {model_path}")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ONNX Runtime es necesario para habilitar el restaurador facial."
            ) from exc
        self.input_size = input_size
        self.output_range = output_range
        self.session = ort.InferenceSession(str(model_path), providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def restore(self, bgr: np.ndarray) -> np.ndarray:
        original_size = (bgr.shape[1], bgr.shape[0])
        image = cv2.resize(
            bgr,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_CUBIC,
        )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(rgb, (2, 0, 1))[None]
        output = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        output = np.transpose(output, (1, 2, 0))
        mode = self.output_range
        if mode == "auto":
            mode = "minus_one_one" if float(output.min()) < -0.05 else "zero_one"
        if mode == "minus_one_one":
            output = (output + 1.0) * 0.5
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return cv2.resize(output, original_size, interpolation=cv2.INTER_LANCZOS4)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        return self.restore(bgr)


def build_face_restorer(config, providers) -> FaceRestorer:
    if not config.enabled:
        return IdentityFaceRestorer()
    return OnnxFaceRestorer(
        model_path=config.model_path,
        input_size=config.input_size,
        output_range=config.output_range,
        providers=providers,
    )


# Alias de compatibilidad para integraciones anteriores.
class OptionalFaceRestorer:
    def __new__(
        cls,
        enabled: bool,
        model_path: Path,
        input_size: int,
        output_range: str,
        providers,
    ) -> FaceRestorer:
        config = type(
            "RestorerSettings",
            (),
            {
                "enabled": enabled,
                "model_path": model_path,
                "input_size": input_size,
                "output_range": output_range,
            },
        )
        return build_face_restorer(config, providers)
