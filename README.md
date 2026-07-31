# FaceSwap-Pro 0.1.0

Pipeline local de reemplazo facial optimizado para GPU NVIDIA. Incluye detección y reconocimiento con InsightFace, tracking temporal, flujo óptico, composición por ROI, decodificación NVDEC, codificación NVENC, audio original, métricas y etiqueta visible `CONTENIDO SINTÉTICO · IA`.

## Estructura del proyecto

```text
FaceSwap-Pro/
├── inputs/
│   ├── videos/
│   │   └── input.mp4
│   ├── source_faces/
│   │   ├── frente.jpg
│   │   ├── tres_cuartos.jpg
│   │   └── perfil.jpg
│   └── target_faces/
│       └── target.jpg
├── models/
│   ├── inswapper_128.onnx
│   └── face_restorer.onnx       # opcional
├── outputs/
│   ├── videos/
│   └── manifests/
├── config/
│   ├── max_speed.yaml
│   └── quality.yaml
├── scripts/
├── docs/
│   └── model_backends.md
├── src/
│   └── faceswap_pro/
│       ├── modeling.py              # contratos y registro de backends
│       └── insightface_backend.py   # adaptador predeterminado
└── tests/
```

## Instalación automática en Windows con Conda

Requisitos:

- Windows 11 de 64 bits;
- Anaconda o Miniconda;
- controlador NVIDIA actualizado;
- WinGet/App Installer.

Desde PowerShell, en la carpeta del proyecto:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Para reconstruir completamente el entorno:

```powershell
conda deactivate
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Recreate
```

Después:

```powershell
conda activate FaceSwap-Pro
faceswap-pro doctor
```

El diagnóstico esperado incluye:

```json
"cuda_provider_ok": true,
"h264_nvenc": true,
"cuda_hw_decode": true
```

## Preparar archivos

Copia el video de entrada:

```text
inputs/videos/input.mp4
```

Copia las fotografías de la identidad que deseas insertar:

```text
inputs/source_faces/
```

Copia la referencia del sujeto objetivo del video:

```text
inputs/target_faces/target.jpg
```

Coloca el modelo:

```text
models/inswapper_128.onnx
```

También puedes crear o reparar la estructura mediante:

```powershell
faceswap-pro init
```

## Ejecutar

Con las rutas predeterminadas:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_example.ps1
```

O directamente:

```powershell
faceswap-pro run
```

La salida se genera automáticamente, por ejemplo:

```text
outputs/videos/input_faceswap_20260731_173500.mp4
outputs/manifests/input_faceswap_20260731_173500.manifest.json
```

## Usar rutas personalizadas

```powershell
faceswap-pro run `
  --input "D:\Videos\escena.mp4" `
  --source-dir "D:\Rostros\origen" `
  --target-ref "D:\Rostros\objetivo.jpg" `
  --swapper-model ".\models\inswapper_128.onnx" `
  --config ".\config\max_speed.yaml"
```

Para indicar una salida exacta:

```powershell
faceswap-pro run `
  --output ".\outputs\videos\resultado_final.mp4"
```

Para cambiar solo las carpetas de salida:

```powershell
faceswap-pro run `
  --output-dir ".\renders\videos" `
  --manifest-dir ".\renders\metadata"
```

## Perfiles

Máxima velocidad:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_example.ps1 `
  -Config ".\config\max_speed.yaml"
```

Mayor calidad y seguimiento más frecuente:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_example.ps1 `
  -Config ".\config\quality.yaml"
```

## Pipeline

```text
FFmpeg/NVDEC → cola de lectura
                   ↓
SCRFD + ArcFace + tracking
                   ↓
INSwapper en GPU
                   ↓
blend ROI + color + etiqueta en CPU paralelo
                   ↓
FFmpeg/NVENC → outputs/videos
                   ↓
manifiesto y métricas → outputs/manifests
```

## Arquitectura SOLID y cambio de modelo

El flujo principal ya no depende directamente de InsightFace. La creación del modelo
se concentra en una fábrica y el pipeline recibe un `ModelBundle` compuesto por dos
interfaces pequeñas:

- `FaceAnalyzer`: detección y reconocimiento convertidos a `FaceData`;
- `FaceSwapper`: generación del recorte y su transformación mediante `SwapResult`.

La implementación actual vive en `insightface_backend.py`. Para usar otro framework o
modelo, crea un adaptador, regístralo con `register_model_backend(...)` y selecciónalo
en el perfil:

```yaml
engine:
  backend: my_backend
  plugins:
    - my_backend
  options:
    provider: CUDA
```

El archivo del modelo puede indicarse con cualquiera de estos aliases:

```powershell
faceswap-pro run --model ".\models\my_model.onnx"
faceswap-pro run --swapper-model ".\models\my_model.onnx"
```

Consulta [`docs/model_backends.md`](docs/model_backends.md) para ver un adaptador
completo. El pipeline, el tracking, la identidad y el blend no necesitan cambios al
añadir un backend nuevo.

El manifiesto registra hashes SHA-256 de las entradas y del modelo, proveedores ONNX Runtime, backend de decodificación, codec de salida, FPS efectivo y estadísticas por etapa.

## Supervisión

```powershell
nvidia-smi `
  --query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw `
  --format=csv `
  -l 1
```

## Límites

`inswapper_128.onnx` genera una cara de 128×128. El blend a mayor resolución mejora la integración, pero no crea detalle facial real que el modelo no haya generado. La velocidad depende de la resolución, el códec de entrada, el número de rostros y la frecuencia de redetección.

Este proyecto no redistribuye modelos. Verifica las licencias correspondientes y utiliza material para el que tengas permiso, conservando la etiqueta visible de contenido sintético.
