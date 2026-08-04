# Informe de validación de la implementación

Fecha de validación del paquete: 2026-08-03.

## Resultado automatizado

```text
109 passed
```

La suite incluye las pruebas existentes y regresiones nuevas para planificación de
clips, tracking multiinstancia del actor y su reflexión, reconocimiento reservado
por trayectoria, doble composición por frame, máscara DreamID-V multiinstancia antes de difusión, banco
multi-referencia, hoja visual, hashes cacheados, metadatos de color y C2PA.

## Prueba FFmpeg real

Se codificó un vídeo sintético H.264 con `RawFFmpegWriter` y se verificó mediante
FFprobe:

- 12 fotogramas a 12 FPS;
- resolución 160×96;
- `yuv420p`;
- rango limitado;
- matriz, transferencia y primarias BT.709.

## Validaciones no ejecutables dentro de este paquete

No se realizó inferencia perceptual DreamID-V porque el ZIP no contiene los
checkpoints DreamID-V/Wan ni el entorno CUDA externo. Tampoco se ejecutó una firma
C2PA real porque `c2patool` no está instalado en el entorno de validación. Ambos
flujos tienen pruebas de contrato y fallback, pero deben comprobarse en la RTX 5070
Ti con los artefactos reales.
