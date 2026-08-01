# Backends de modelo y nombres precisos

## Separación de responsabilidades

El pipeline depende de contratos pequeños:

```python
class FaceAnalyzer(Protocol):
    def find_faces(self, image): ...
    def analyze(self, frame, previous_bbox, full_scan): ...

class FaceSwapper(Protocol):
    def swap(self, frame, target_face, source_face) -> SwapResult: ...
```

Cada fábrica devuelve un `ModelBundle` con analizador, generador y capacidades.
`pipeline.py` no conoce InsightFace, MediaPipe ni PyTorch.

## Capacidades declarativas

```python
ModelCapabilities(
    generator="hififace",
    native_output_size=256,
    geometry_conditioning="3dmm_internal",
    geometry_postprocess="learned_semantic_mask",
    temporal_generation="frame_independent",
)
```

Un backend solo es marcado `truly_3d_aware` cuando la geometría condiciona al
modelo durante la generación. Una malla aplicada después se declara únicamente como
`geometry_postprocess`.

## Backends integrados

### `insightface_inswapper`

- generador: `inswapper_128.onnx`;
- salida nativa: 128×128;
- condicionamiento geométrico interno: ninguno.

### `insightface_inswapper_mediapipe_mesh`

- generador: el backend base, normalmente INSwapper;
- MediaPipe estima una malla después de la generación;
- realiza warp triangular, máscara de contorno y atenuación por pose;
- no se presenta como generador 3D-aware.

### `hififace_3dmm`

- reutiliza InsightFace solo para detección, reconocimiento y tracking;
- alinea source y target con la plantilla del runtime HifiFace;
- HifiFace combina forma/identidad 3DMM del source con pose y expresión del target;
- Semantic Facial Fusion produce el rostro final y su máscara;
- el adaptador externo queda detrás de `HifiFaceRuntime`.

## Alias obsoleto

`mediapipe_3d_hybrid` permanece registrado para configuraciones existentes. Su fábrica
solo delega al backend de malla y emite `DeprecationWarning`. Los perfiles nuevos no
usan ese nombre.

## Añadir otro generador

```python
from faceswap_pro.modeling import (
    ModelBundle,
    ModelCapabilities,
    register_model_backend,
)


def build(config, model_path):
    return ModelBundle(
        backend="my_generator",
        analyzer=my_analyzer,
        swapper=my_swapper,
        capabilities=ModelCapabilities(
            generator="my_model",
            native_output_size=512,
            geometry_conditioning="none",
        ),
        model_artifacts=(model_path,),
    )


register_model_backend("my_generator", build)
```

Selecciona el módulo desde `engine.plugins`. No necesitas modificar tracking, video,
blend o manifiestos.
