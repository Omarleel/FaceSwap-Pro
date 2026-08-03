# FaceSwap-Pro

Pipeline local y modular de reemplazo facial con GPU NVIDIA. Mantiene los flujos
por fotograma existentes y añade un backend temporal nativo para **DreamID-V sobre
Wan 2.1 1.3B**. La etiqueta visible es configurable por perfil; el perfil DreamID-V
para RTX 5070 Ti no dibuja marcas sobre los fotogramas y conserva un manifiesto JSON
separado.

## Backends incluidos

| Backend | Generador | Geometría | Estado |
|---|---|---|---|
| `insightface_inswapper` | INSwapper 128 | Ninguna dentro del generador | Compatible y rápido |
| `insightface_inswapper_mediapipe_mesh` | INSwapper 128 | Postproceso por malla MediaPipe | Compatible, no 3D-aware generativo |
| `hififace_3dmm` | HifiFace 256 | Identidad condicionada por 3DMM dentro del generador | 3DMM-aware real |
| `dreamid_v` | DreamID-V + Wan 2.1 1.3B | Diffusion Transformer temporal sobre clips | Backend de vídeo, 480p/720p |

El alias histórico `mediapipe_3d_hybrid` permanece para no romper YAML antiguos,
pero emite una advertencia de obsolescencia. El nombre era impreciso porque MediaPipe
solo deformaba la salida 2D después de generarla.

## Arquitectura

```text
FaceAnalyzer ───────────────┐
                           ├─► tracking y selección de identidad
FaceGenerator / FaceSwapper┘
        │
        ├─ INSwapper 128
        ├─ INSwapper + postproceso por malla
        └─ HifiFace + extractor 3DMM + Semantic Facial Fusion
                                      │
                                      ▼
                      máscara aprendida / composición ROI
                                      │
                                      ▼
                              NVENC + manifiesto

VideoSwapBackend
        └─ DreamID-V Faster + Wan 2.1 1.3B
                    │
                    ├─ banco de referencias por pose/calidad
                    ├─ tracking real del sujeto objetivo
                    ├─ ventanas solapadas + worker persistente
                    ├─ stitching guiado por flujo
                    ├─ composición selectiva con oclusiones
                    └─ codificación final + métricas + C2PA/manifiesto
```

Las capacidades del modelo se declaran mediante `ModelCapabilities`:

- `geometry_conditioning`: geometría usada dentro del generador;
- `geometry_postprocess`: corrección aplicada después de generar;
- `truly_3d_aware`: verdadero únicamente para condicionamiento interno conocido.

## Instalación base

El entorno existente con ONNX Runtime GPU continúa funcionando. Instala el proyecto:

```powershell
python -m pip install -e .
faceswap-pro doctor
```

Para el backend opcional por malla:

```powershell
python -m pip install -e ".[mesh]"
```

## Preparar HifiFace 3DMM

La integración usa la implementación externa MIT `xuehy/HiFiFace-pytorch`. El código
y los pesos no se redistribuyen dentro de este ZIP.

1. Instala juntos una build CUDA de PyTorch y su `torchvision` compatible con tu controlador y GPU.
2. Instala las dependencias de soporte:

```powershell
python -m pip install -e ".[hififace3d]"
```

3. Clona la implementación. En Windows no uses un `git clone` normal: el
repositorio contiene `AdaptiveWingLoss/aux.py` y `AUX` es un nombre reservado
por NTFS. El instalador `scripts/setup_windows.ps1` ya aplica automáticamente
un checkout compatible. Para hacerlo manualmente:

```powershell
git clone --depth 1 --no-checkout `
  https://github.com/xuehy/HiFiFace-pytorch.git `
  .\third_party\HiFiFace-pytorch

git -C .\third_party\HiFiFace-pytorch config core.protectNTFS false
git -C .\third_party\HiFiFace-pytorch sparse-checkout set --no-cone `
  "/*" "!/AdaptiveWingLoss/aux.py"
git -C .\third_party\HiFiFace-pytorch checkout --force
```

4. Descarga desde las fuentes indicadas por esa implementación:

```text
models/hififace/
├── standard_model/
│   └── generator_320000.pth
└── aux/
    ├── Deep3DFaceRecon/epoch_20_new.pth
    ├── arcface/ms1mv3_arcface_r100_fp16_backbone.pth
    └── BFM/
        ├── 01_MorphableModel.mat
        ├── BFM_exp_idx.mat
        ├── BFM_front_idx.mat
        ├── BFM_model_front.mat
        ├── Exp_Pca.bin
        ├── facemodel_info.mat
        ├── select_vertex_id.mat
        ├── similarity_Lm3D_all.mat
        └── std_exp.txt
```

BFM requiere obtener los archivos bajo sus propias condiciones. Usa únicamente
checkpoints de una fuente confiable: los checkpoints PyTorch pueden contener datos
serializados ejecutables.

5. Valida todo el perfil antes de procesar:

```powershell
faceswap-pro doctor --config .\config\quality_3dmm.yaml
```

El bloque `hififace_3dmm.ready` debe ser `true`.

## Preparar DreamID-V para RTX 5070 Ti

DreamID-V se ejecuta en un entorno Python separado para que sus dependencias no
entren en conflicto con ONNX Runtime e InsightFace. Consulta
[`docs/dreamidv_5070ti.md`](docs/dreamidv_5070ti.md) para la estructura completa.
El perfil incluido usa `dreamidv_faster.pth`, 832×480, 49 fotogramas, 16 pasos,
offloading de modelo y T5 en CPU.

Valida primero los archivos y la GPU:

```powershell
faceswap-pro doctor --config .\config\quality_dreamidv.yaml
```

Ejecuta:

```powershell
faceswap-pro run --config .\config\quality_dreamidv.yaml
```

El backend divide vídeos largos en ventanas `4n+1` solapadas, desplaza fronteras a
cortes de escena y mantiene el runtime cargado mediante un worker persistente. Un
fallback automático conserva compatibilidad con la CLI oficial. La referencia
objetivo se usa para seguir a la persona correcta y la salida se recompone sobre el
vídeo original únicamente dentro de su máscara temporal.

Cada ejecución temporal genera también métricas JSON, una hoja visual entrada/salida,
hashes cacheados de modelos y C2PA opcional. Consulta
[`docs/mejoras_calidad_2026.md`](docs/mejoras_calidad_2026.md) para el mapa completo.

## Ejecutar

### Flujo actual, sin cambios

```powershell
faceswap-pro run --config .\config\quality.yaml
```

### INSwapper asistido por malla

```powershell
faceswap-pro run --config .\config\quality_mesh_assisted.yaml
```

### Generación condicionada por 3DMM

```powershell
faceswap-pro run --config .\config\quality_3dmm.yaml
```

`quality_3d.yaml` es un alias legible del perfil `quality_3dmm.yaml`.

## Rutas personalizadas

`--model-path` admite archivos o directorios:

```powershell
# Archivo ONNX
faceswap-pro run `
  --model-path .\models\inswapper_128.onnx `
  --config .\config\quality.yaml

# Directorio de checkpoint HifiFace
faceswap-pro run `
  --model-path .\models\hififace\standard_model `
  --config .\config\quality_3dmm.yaml
```

Los alias antiguos `--model` y `--swapper-model` continúan disponibles.

## Qué significa “3D-aware” en este proyecto

`hififace_3dmm` no es un simple warper. El generador obtiene coeficientes de forma
3DMM de source y target, conserva la identidad/forma del source y combina expresión
y pose del target antes de sintetizar. Después, Semantic Facial Fusion predice una
máscara y conserva iluminación, fondo y oclusiones dentro del propio modelo.

Esto es **condicionamiento generativo por 3DMM**, no un avatar 3D explícito con rig,
texturas editables o render físico. El perfil sigue siendo independiente por frame;
la consistencia temporal neuronal queda como una extensión separada futura.

## Pruebas

```powershell
pytest -q
```

La suite cubre contratos, configuración, tracking, ventanas DreamID-V solapadas,
banco multi-referencia, composición selectiva, métricas visuales, caché de hashes,
C2PA, compatibilidad del flujo anterior, HifiFace y el adaptador 3DMM.

## Uso responsable

Procesa únicamente material propio o autorizado. El perfil DreamID-V no añade una
marca visible, pero mantiene un manifiesto JSON, intenta incrustar C2PA y conserva
los frames originales cuando el sujeto es ambiguo o no está localizado. Revisa por
separado las licencias del código, los pesos, Wan 2.1, BFM e InsightFace.
