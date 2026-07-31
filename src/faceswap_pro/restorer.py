from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class OptionalFaceRestorer:
    """Adaptador ONNX genérico para restauradores RGB BCHW de una entrada y una salida."""

    def __init__(self, enabled: bool, model_path: Path, input_size: int, output_range: str, providers):
        self.enabled = enabled
        self.input_size = input_size
        self.output_range = output_range
        self.session = None
        self.input_name = None
        self.output_name = None
        if enabled:
            if not model_path.is_file():
                raise FileNotFoundError(f"Restaurador ONNX no encontrado: {model_path}")
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        if self.session is None:
            return bgr
        original_size = (bgr.shape[1], bgr.shape[0])
        image = cv2.resize(bgr, (self.input_size, self.input_size), interpolation=cv2.INTER_CUBIC)
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
