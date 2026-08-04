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
5. clona DreamID-V e instala el adaptador SDPA nativo de FaceSwap-Pro;
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

El perfil rápido usa 8 pasos y 5 fotogramas de solape. Es más veloz, pero puede
perder algo de detalle frente al perfil de 16 pasos.

## Flujo de GPU optimizado

1. **Banco de identidad y tracking.** InsightFace construye las referencias,
   identifica al actor y mantiene hasta dos apariciones simultáneas para cubrir el
   rostro directo y su reflexión.
2. **Liberación de InsightFace.** Tras finalizar el tracking, el backend elimina
   las sesiones ONNX de InsightFace antes de cargar los modelos pesados.
3. **DWPose global y aislado.** DWPose se ejecuta una sola vez sobre el proxy
   completo. Después se recortan `pose` y `mask` con los mismos índices de cada
   ventana, por lo que los fotogramas de solape no vuelven a inferirse.
4. **Cierre completo de DWPose.** El proceso se termina y Windows libera sus
   buffers CUDA antes de iniciar Wan.
5. **Fase DreamID-V persistente.** El worker permanece activo entre clips y
   reutiliza `context.pth` y los latentes VAE de las referencias repetidas.
   En GPU de 16 GB alterna DiT y VAE por etapas para que los pesos pesados no
   compitan por VRAM.
6. **VAE BF16 y concatenación lineal.** El VAE oficial se crea en FP32. El worker
   lo convierte al dtype configurado —BF16 en los perfiles de 16 GB—, conserva
   los latentes en baja precisión y acumula los bloques temporales en una lista
   antes de concatenarlos una sola vez.
7. **Limpieza por cambio de etapa.** `empty_cache()` se ejecuta después de mover
   un módulo pesado a CPU, tras un OOM o cuando existe memoria reservada inactiva
   relevante. No se vacía el allocator después de cada operación.
8. **Reinicio controlado.** Si el worker falla, se cierra para liberar toda su VRAM
   y se vuelve a crear según `worker_restart_attempts`. Con
   `worker_fallback: false`, la ejecución se detiene en lugar de caer a una CLI
   mucho más lenta.
9. **Stitching y composición.** Las ventanas se mezclan en los solapes y el vídeo
   original se conserva fuera de las máscaras del actor y su reflexión.



## Perfilado interno del subproceso

El worker ya no aparece como una única espera opaca del proceso principal. Emite
telemetría estructurada que el cliente incorpora al mismo `logs/profile.jsonl` y
`logs/logs.jsonl` de la ejecución. Cada evento conserva `worker_request_id`, índice
de clip, frame inicial, PID del worker y referencia utilizada.

Configuración recomendada:

```yaml
profile_worker: true
profile_dit_forwards: true
profile_worker_cprofile: true
profile_worker_cprofile_all: false
profile_detailed_clips: 1
worker_cprofile_top: 80
worker_heartbeat_seconds: 15
```

Operaciones principales registradas:

- `dreamidv.initialize.pipeline`: carga real de Wan y DreamID-V;
- `dreamidv.context.load` y `dreamidv.context_cache`;
- `dreamidv.vae.encode` diferenciando referencia y vídeo temporal;
- `dreamidv.dit.forward` por paso y pasada condicional/no condicionada;
- `dreamidv.vae.decode` y `dreamidv.video.write`;
- `dreamidv.gpu_memory`, `dreamidv.cuda_cleanup` y
  `dreamidv.clip_summary`;
- `cprofile_function` con las funciones Python más costosas de cada clip;
- `dreamidv.heartbeat` durante generaciones largas.

Los tiempos CUDA se obtienen con eventos diferidos y una sola sincronización al
final del clip. Esto evita serializar cada forward únicamente para medirlo. En
`logs/logs.jsonl` se registran OOM, reintentos con offload, fallos de sincronización
y tracebacks originados dentro del worker.

## SDPA nativo y selección real del kernel

El adaptador `faceswap_pro.dreamidv_sdpa` reemplaza únicamente la función de
atención de DreamID-V Faster; no modifica los pesos. Corrige cuatro problemas del
fallback upstream:

1. respeta `q_lens` y `k_lens`;
2. aplica `q_scale` y `softmax_scale`;
3. recorta padding por muestra para no construir una matriz `L×S` gigantesca;
4. fuerza un backend por vez y registra cuál ejecutó realmente la operación.

Configuración recomendada para una RTX 5060 Ti/5070 Ti de 16 GB:

```yaml
sdpa_backend_priority: [cudnn, flash, efficient, math]
sdpa_allow_math_fallback: false
sdpa_padding_mode: ragged
sdpa_diagnostics: true
```

`ragged` recorta cada secuencia a su longitud válida y suele conservar la ruta
fusionada. `mask` usa una máscara booleana compacta `[B, 1, 1, S]`, donde `true`
significa que la clave participa en SDPA. La máscara se ofrece para validación o
lotes variables; DreamID-V normalmente procesa lote 1 y se beneficia más de
`ragged`.

Antes de lanzar un vídeo largo se puede comprobar el soporte fusionado con:

```powershell
python .\scripts\probe_dreamidv_sdpa.py --seq-len 2048
```

La prueba sintética no sustituye la selección con las longitudes reales, pero evita
iniciar DreamID-V cuando la instalación no expone ningún kernel fusionado.

Al comenzar la primera atención aparecerá una línea como:

```text
FaceSwap-Pro SDPA: backend=CUDNN, torch=..., gpu=..., dtype=..., q=..., k=..., mask=none
```

Ese backend no es una estimación: la llamada se ejecuta dentro de un contexto que
habilita únicamente ese kernel. Si cuDNN falla, se prueba Flash y luego Efficient.
Con `sdpa_allow_math_fallback: false`, si ninguno funciona la ejecución termina con
un diagnóstico inmediato. Activa MATH solo para depuración de clips diminutos.

## Parámetros de memoria

```yaml
persistent_worker: true
precompute_pose: true
precompute_pose_global: true
worker_restart_attempts: 1
worker_fallback: false
release_analysis_gpu: true

# Residencia por etapas para GPU de 16 GB.
offload_model: true
offload_fallback: true
staged_offload: true
offload_vae_during_dit: true
vae_dtype: bfloat16
stream_video_write: true
t5_cpu: true

# Cachés y gestión de memoria del worker.
cache_context: true
cache_reference_latents: true
reference_latent_cache_size: 8
cuda_cleanup_mode: adaptive
cuda_cleanup_reserved_ratio: 0.82
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
chunk_overlap_frames: 5
benchmark_enabled: false
```

Sube el solape cuando una escena muestre uniones visibles entre ventanas.

## Mensajes que no son errores

Estos avisos no bloquean por sí solos:

- `FutureWarning` de `torch.cuda.amp.autocast`;
- nodos ONNX asignados al CPU para operaciones de forma;
- mensajes de diagnóstico que descartan un backend fusionado y prueban el siguiente.

Sí debe detenerse la ejecución ante `CUDA out of memory`, archivos de máscara
ausentes, un worker que agote sus reinicios o el mensaje `Ningún kernel SDPA
fusionado pudo ejecutar DreamID-V`.

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
