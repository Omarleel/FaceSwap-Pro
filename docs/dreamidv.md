# DreamID-V

El backend `dreamid_v` procesa clips completos con DreamID-V Faster y Wan 2.1.
Puede usar el mismo entorno Conda que FaceSwap-Pro, siempre que se instalen las
dependencias de forma selectiva mediante `scripts/setup_dreamidv_windows.ps1`.
Los pesos de DreamID-V y Wan no se redistribuyen en este proyecto.

## Estructura esperada

```text
FaceSwap-Pro/
├── third_party/DreamID-V/
│   ├── generate_dreamidv_faster.py
│   ├── dreamidv_wan_faster/context.pth
│   └── pose/models/
│       ├── dw-ll_ucoco_384.onnx
│       └── yolox_l.onnx
└── models/dreamidv/
    ├── dreamidv_faster.pth
    └── Wan2.1-T2V-1.3B/
        └── ... checkpoint base ...
```

## Preparación en Windows

Con el entorno Conda `FaceSwap-Pro` activo:

```powershell
Set-ExecutionPolicy -Scope Process Bypass -Force
.\scripts\setup_dreamidv_windows.ps1
```

El script:

1. conserva PyTorch si CUDA ya funciona;
2. deja una sola variante de OpenCV;
3. deja únicamente `onnxruntime-gpu`;
4. instala las dependencias compatibles de DreamID-V;
5. clona DreamID-V y aplica el wrapper de atención de PyTorch/FlashAttention;
6. descarga DreamID-V Faster, Wan 2.1 y DWPose;
7. ejecuta el diagnóstico final.

Comprueba manualmente:

```powershell
faceswap-pro doctor --config .\config\quality_dreamidv.yaml
```

`configured_backend_ready` debe ser `true`.

## Ejecución

Perfil de calidad:

```powershell
faceswap-pro run `
  --input .\inputs\videos\input.mp4 `
  --source-dir .\inputs\source_faces `
  --target-ref .\inputs\target_faces\target.jpg `
  --config .\config\quality_dreamidv.yaml `
  --output .\outputs\videos\resultado_dreamidv.mp4
```

Perfil rápido para vídeos largos o validaciones:

```powershell
faceswap-pro run `
  --input .\inputs\videos\input.mp4 `
  --source-dir .\inputs\source_faces `
  --target-ref .\inputs\target_faces\target.jpg `
  --config .\config\speed_dreamidv.yaml `
  --output .\outputs\videos\resultado_dreamidv_rapido.mp4
```

El perfil rápido usa 8 pasos y 9 fotogramas de solape. Es más veloz, pero puede
perder algo de detalle frente al perfil de 16 pasos.

## Flujo de GPU optimizado

1. **Banco de identidad y tracking.** InsightFace construye las referencias,
   identifica al actor y mantiene hasta dos apariciones simultáneas para cubrir el
   rostro directo y su reflexión.
2. **Liberación de InsightFace.** Tras finalizar el tracking, el backend elimina
   las sesiones ONNX de InsightFace antes de cargar los modelos pesados.
3. **Fase DWPose aislada.** Todos los clips se extraen y se procesan con un worker
   DWPose persistente. Los modelos ONNX se cargan una sola vez.
4. **Cierre completo de DWPose.** El proceso se termina y Windows libera sus
   buffers CUDA antes de iniciar Wan.
5. **Fase DreamID-V persistente.** Wan y DreamID-V se cargan una vez. Cada clip usa
   `pose.mp4` y `mask.mp4` precalculados, sin volver a inicializar DWPose.
6. **Limpieza entre clips.** El worker elimina tensores temporales y ejecuta
   `torch.cuda.empty_cache()` e `ipc_collect()` cuando están disponibles.
7. **Reinicio controlado.** Si el worker falla, se cierra para liberar toda su VRAM
   y se vuelve a crear según `worker_restart_attempts`. Con
   `worker_fallback: false`, la ejecución se detiene en lugar de caer a una CLI
   mucho más lenta.
8. **Stitching y composición.** Las ventanas se mezclan en los solapes y el vídeo
   original se conserva fuera de las máscaras del actor y su reflexión.

## Parámetros de memoria

```yaml
persistent_worker: true
precompute_pose: true
worker_restart_attempts: 1
worker_fallback: false
release_analysis_gpu: true
offload_model: true
t5_cpu: true
```

En Windows se evita `expandable_segments`, porque varias builds de PyTorch no lo
soportan. El backend usa `max_split_size_mb` y un umbral de recolección para reducir
fragmentación.

## Perfiles incluidos

### `quality_dreamidv.yaml`

```yaml
frame_num: 49
sample_fps: 16
sample_steps: 16
chunk_overlap_frames: 17
```

### `speed_dreamidv.yaml`

```yaml
frame_num: 49
sample_fps: 16
sample_steps: 8
chunk_overlap_frames: 9
benchmark_enabled: false
```

## Mensajes que no son errores

Estos avisos no bloquean por sí solos:

- `FutureWarning` de `torch.cuda.amp.autocast`;
- nodos ONNX asignados al CPU para operaciones de forma;
- aviso de máscara de padding al usar SDPA.

Sí debe detenerse la ejecución ante `CUDA out of memory`, archivos de máscara
ausentes o un worker que agote sus reinicios.

## Salidas

```text
outputs/videos/resultado_dreamidv.mp4
outputs/manifests/resultado_dreamidv.manifest.json
outputs/manifests/resultado_dreamidv.quality.json
outputs/manifests/resultado_dreamidv.quality-contact-sheet.jpg
```

Cuando `release_analysis_gpu: true`, las métricas que requieren volver a ejecutar
InsightFace pueden quedar sin muestras; las métricas temporales, composición,
fronteras y codificación siguen disponibles.

## Límites

- DreamID-V Faster sigue siendo un modelo de difusión pesado. El aislamiento de
  fases evita la degradación extrema causada por el fallback CLI, pero el tiempo
  final depende de pasos, duración, resolución y GPU.
- El perfil de 8 pasos prioriza velocidad; revisa el resultado antes de una entrega.
- La selección multi-referencia elige una imagen por clip, no fusiona varias
  referencias dentro de una misma pasada.
- Procesa únicamente material propio o autorizado y conserva el manifiesto de
  procedencia generado por FaceSwap-Pro.
