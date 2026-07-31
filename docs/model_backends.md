# Añadir o cambiar un backend de modelos

El pipeline trabaja con objetos neutrales definidos en `faceswap_pro.modeling` y no
con clases de InsightFace. Un backend nuevo debe proporcionar únicamente:

- un `FaceAnalyzer`, que transforma la salida de detección/reconocimiento en `FaceData`;
- un `FaceSwapper`, que devuelve `SwapResult`;
- una fábrica que construya ambos servicios y los entregue en un `ModelBundle`.

## Contratos

```python
from faceswap_pro.modeling import (
    DetectionStats,
    FaceData,
    ModelBundle,
    SwapResult,
    register_model_backend,
)


class MyAnalyzer:
    def find_faces(self, image):
        # Ejecutar el detector del framework y convertir cada resultado a FaceData.
        return []

    def analyze(self, frame, previous_bbox, full_scan):
        faces = []
        return faces, DetectionStats(
            detected=len(faces),
            recognized=len(faces),
            full_scan=full_scan,
        )


class MySwapper:
    def swap(self, frame, target_face: FaceData, source_face: FaceData):
        crop, affine = my_model.run(frame, target_face, source_face)
        return SwapResult(crop=crop, affine=affine)


def build_my_backend(config, model_path):
    options = config.engine.options
    return ModelBundle(
        backend="my_backend",
        analyzer=MyAnalyzer(),
        swapper=MySwapper(),
        providers=(options.get("provider", "CPU"),),
        runtime={"framework": "my-framework"},
    )


register_model_backend("my_backend", build_my_backend)
```

Guarda el código, por ejemplo, en `my_backend.py`. El motor importa los módulos
indicados en `engine.plugins`; al importarse, el módulo ejecuta el registro. Después
selecciona el backend desde YAML:

```yaml
engine:
  backend: my_backend
  plugins:
    - my_backend
  options:
    provider: CUDA
```

El módulo debe estar instalado o ser importable desde el directorio de ejecución.
No es necesario modificar `engine.py`, `pipeline.py` ni la CLI.

La CLI conserva `--swapper-model` y también acepta su alias genérico `--model`.

## Responsabilidades

- `modeling.py`: contratos, datos neutrales y registro de fábricas.
- `insightface_backend.py`: único lugar que conoce `FaceAnalysis` e `INSwapper`.
- `engine.py`: raíz de composición y proveedores de ONNX Runtime.
- `pipeline.py`: orquestación del caso de uso mediante `ModelBundle`.
- `identity.py`, `tracking.py` y `parallel_pipeline.py`: lógica de dominio sobre
  `FaceData`, sin dependencias de un framework concreto.

Así, un backend nuevo se agrega mediante extensión y registro, sin modificar el
pipeline existente.
