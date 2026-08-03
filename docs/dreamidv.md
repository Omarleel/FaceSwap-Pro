# DreamID-V

El backend `dreamid_v` procesa clips completos en un entorno Python aislado y deja
el entorno principal para InsightFace, tracking, composición, codificación y
proveniencia. Los pesos de DreamID-V y Wan no se redistribuyen en este proyecto.

## Estructura esperada

```text
FaceSwap-Pro/
├── .venv-dreamidv/
│   └── Scripts/python.exe
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

## Preparación

1. Clona DreamID-V en `third_party/DreamID-V`.
2. Crea un entorno separado e instala allí PyTorch CUDA y los requisitos del
   checkout de DreamID-V.
3. Comprueba que ese entorno importe también `decord`.
4. Descarga el checkpoint DreamID-V, el checkpoint base Wan y los dos modelos
   DWPose en las rutas del árbol anterior.
5. Ejecuta:

```powershell
faceswap-pro doctor --config .\config\quality_dreamidv.yaml
```

El diagnóstico comprueba el checkout, los checkpoints, DWPose, FFmpeg, el bridge del
worker persistente y las dependencias del entorno externo. También muestra el estado
de `c2patool` cuando C2PA está habilitado.

## Ejecución

```powershell
faceswap-pro run `
  --input .\inputs\videos\input.mp4 `
  --source-dir .\inputs\source_faces `
  --target-ref .\inputs\target_faces\target.jpg `
  --config .\config\quality_dreamidv.yaml
```

## Flujo implementado

1. **Banco de identidad.** Analiza todas las fotos fuente, calcula un embedding
   ponderado y conserva referencias izquierda, frontal y derecha según pose,
   nitidez, exposición, tamaño y confianza de detección.
2. **Proxy sin pérdida.** Normaliza la línea temporal a `sample_fps` usando
   H.264 lossless 4:4:4 por defecto. En HDR aplica tonemapping explícito a BT.709,
   o rechaza/pasa el material según `hdr_policy`.
3. **Tracking del objetivo.** Compara cada rostro con `--target-ref`, mantiene un
   track temporal y marca frames ambiguos cuando dos personas tienen similitud
   demasiado próxima. En esos frames no aplica el reemplazo.
4. **Ventanas solapadas.** Planifica clips `4n+1` con solapamiento. Cerca de un
   corte de escena desplaza el inicio de ventana al corte y evita mezclar planos.
5. **Worker persistente.** Para `faster` y `dwpose`, carga DreamID-V una vez y
   procesa todos los clips mediante JSON Lines. Si falla y `worker_fallback` está
   activo, vuelve a la CLI oficial por clip.
6. **Stitching temporal.** Alinea las dos hipótesis del solapamiento al movimiento
   del vídeo fuente y las mezcla con una curva cosenoidal. Las semillas se derivan
   del frame absoluto para que el plan sea reproducible.
7. **Composición selectiva.** Conserva el vídeo original fuera de la máscara del
   sujeto objetivo, estabiliza esa máscara con flujo óptico, iguala color y restaura
   oclusores probables como manos, cabello cruzado, gafas u objetos.
8. **Codificación final real.** Aplica el bloque `encoding` del YAML y después
   recupera el audio original. Ya no depende silenciosamente del códec interno usado
   por DreamID-V.
9. **Evaluación.** Escribe un JSON de métricas y una hoja JPEG entrada/salida con
   muestras normalizadas de toda la línea temporal.
10. **Proveniencia.** Crea manifiesto JSON con hashes cacheados y, cuando está
    disponible, incrusta credenciales C2PA mediante `c2patool`.

## Archivos de salida

Para `video_faceswap_....mp4` se generan normalmente:

```text
outputs/videos/video_faceswap_....mp4
outputs/manifests/video_faceswap_....manifest.json
outputs/manifests/video_faceswap_....quality.json
outputs/manifests/video_faceswap_....quality-contact-sheet.jpg
outputs/manifests/video_faceswap_....c2pa-definition.json   # si C2PA se solicita
outputs/manifests/.hash-cache/model-sha256.json
```

Las métricas incluyen similitud de identidad, error normalizado de landmarks,
cambio fuera de máscara, delta temporal y salto relativo en fronteras de clips.
Un valor `null` significa que no hubo suficientes muestras válidas, no que la
métrica haya sido aprobada.

## Parámetros principales del perfil de 16 GB

```yaml
frame_num: 49
sample_fps: 16
sample_steps: 16
chunk_overlap_frames: 17
scene_aware_chunking: true
chunk_crf: 0
chunk_pix_fmt: yuv444p
persistent_worker: true
reference_bank_size: 6
benchmark_enabled: true
hdr_policy: tonemap
```

`frame_num` debe tener forma `4n+1`. Para subir calidad, modifica una sola dimensión
por prueba: primero pasos, luego `frame_num=81` o `size="1280*720"`. Combinar 720p y
81 frames puede superar 16 GB según versiones, offloading y fragmentación de VRAM.

## C2PA

El perfil activa C2PA en modo no bloqueante:

```yaml
c2pa_enabled: true
c2pa_required: false
c2pa_tool: c2patool
```

Sin `c2patool`, el vídeo sigue generándose y el manifiesto registra `skipped`. Para
un flujo de entrega, instala `c2patool`, configura un certificado y una clave propios
y usa `c2pa_required: true`:

```yaml
c2pa_sign_cert: certificates/signing-cert.pem
c2pa_private_key: certificates/signing-key.pem
c2pa_required: true
```

La credencial de desarrollo de la herramienta sirve para pruebas, pero no representa
una firma de producción confiable.

## Límites honestos

- El detector de oclusiones incluido es temporal y heurístico; no sustituye una red
  semántica entrenada para manos, cabello, accesorios y transparencias.
- La selección multi-referencia elige una imagen por clip. No modifica el modelo
  DreamID-V para fusionar varias referencias dentro de una misma pasada.
- El tracking evita cambiar a otra persona y conserva frames ambiguos originales,
  pero planos con rostros muy pequeños o fuertemente ocultos pueden quedar sin swap.
- Las métricas automáticas sirven para regresión y comparación A/B. La aceptación
  final todavía requiere revisar el vídeo y la hoja visual.
